import sys
import time
import config
import tools
import knowledge
import agent
config.USE_LONG_TERM_MEMORY = False      # tests must not depend on saved preferences
config.LOG_FILE = "test_log.jsonl"       # keep test runs out of your real log

RESULTS = []


def record(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    line = f"{'PASS' if passed else 'FAIL'}  {name}"
    if detail:
        line += f"  ({detail})"
    print(line)


def check(name, fn):
    try:
        passed, detail = fn()
    except Exception as e:
        passed, detail = False, f"crashed: {e}"
    record(name, passed, detail)


# =====================================================
# PART 1: TOOL TESTS (no AI involved)
# =====================================================
def t_screener_budget():
    rows = tools.get_top_under_budget(100, 10)
    if not rows:
        return False, "no rows returned"
    prices_ok = all(r["price"] <= 100 for r in rows)
    sorted_ok = all(rows[i]["day_change_pct"] >= rows[i + 1]["day_change_pct"] for i in range(len(rows) - 1))
    return prices_ok and sorted_ok and len(rows) <= 10, f"{len(rows)} rows"


def t_screener_range():
    rows = tools.get_top_under_budget(500, 5, 400)
    ok = all(400 <= r["price"] <= 500 for r in rows)
    return ok, f"{len(rows)} rows in range"


def t_consistent():
    out = tools.get_consistent_performers(100, 10)
    stocks = out["stocks"]
    ok = all(
        s["price"] <= 100
        and s["return_1y_pct"] > 0
        and s["months_up"] / s["months_total"] * 100 >= config.MIN_MONTHS_UP_PCT - 0.1
        and s["max_drawdown_pct"] >= -config.MAX_DRAWDOWN_LIMIT
        for s in stocks
    )
    return ok and "criteria" in out, f"{len(stocks)} stocks"


def t_quote():
    q = tools.get_stock_quote("RELIANCE")
    ok = "error" not in q and q["price"] > 0 and q["week_52_high"] >= q["week_52_low"]
    return ok, f"price {q.get('price')}"


def t_history():
    h = tools.get_price_history("TCS", "1y")
    ok = "error" not in h and h["period_high"] >= h["period_low"] and h["trading_days"] > 100
    return ok, f"{h.get('trading_days')} trading days"


def t_bad_symbol():
    q = tools.get_stock_quote("NOTAREALSTOCK123")
    return "error" in q, "returned an error as expected" if "error" in q else "should have returned an error"


def t_compare():
    out = tools.compare_stocks(["TCS", "INFY"], "1y")
    ok = isinstance(out, list) and len(out) == 2 and all("error" not in o for o in out)
    return ok, ""


def t_kb_relevant():
    res = knowledge.search_knowledge("how do I apply for a new share offering")
    titles = [r["title"] for r in res.get("results", [])]
    return any("IPO" in t for t in titles), f"found: {titles}"


def t_kb_irrelevant():
    res = knowledge.search_knowledge("how do I cook biryani")
    return res.get("results") == [], f"found: {[r['title'] for r in res.get('results', [])]}"


def t_news():
    n = tools.get_stock_news("RELIANCE", "Reliance Industries", "week")
    if "error" in n:
        return False, n["error"]
    return "articles" in n, f"{len(n['articles'])} articles"


def run_tool_tests():
    print("\n===== TOOL TESTS (the first run may take minutes while the market is scanned) =====")
    check("Screener: prices within budget and sorted", t_screener_budget)
    check("Screener: price range 400-500 respected", t_screener_range)
    check("Consistent performers follow the criteria", t_consistent)
    check("Quote for Reliance looks sane", t_quote)
    check("1-year history for TCS looks sane", t_history)
    check("Bad symbol returns an error, not a crash", t_bad_symbol)
    check("Compare two stocks", t_compare)
    check("Knowledge base finds IPO entry", t_kb_relevant)
    check("Knowledge base ignores irrelevant question", t_kb_irrelevant)
    check("News search returns articles", t_news)


# =====================================================
# PART 2: AGENT BEHAVIOUR TESTS (uses the AI)
# =====================================================
CALLS = []                                   # records every tool the agent calls
_original_run_tool = agent.run_tool


def _spy(name, args):
    CALLS.append((name, args))
    return _original_run_tool(name, args)


agent.run_tool = _spy                        # the agent now reports its tool calls to us


def ask(question):
    """Asks the agent from a fresh memory. Waits and retries if the free tier rate-limits us."""
    for attempt in range(3):
        CALLS.clear()
        messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
        try:
            return agent.ask_agent(messages, question)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                print("   (rate limited, waiting 30 seconds...)")
                time.sleep(30)
            else:
                raise
    raise RuntimeError("Still rate limited after 3 tries")


def judge(question, answer, rule):
    """LLM-AS-JUDGE: a second AI call grades the answer against a rule."""
    prompt = (
        "You are a strict reviewer of a stock-market assistant's answer.\n"
        f"Question: {question}\n"
        f"Answer: {answer[:3000]}\n"
        f"Rule to check: {rule}\n"
        "Reply with exactly one word: PASS or FAIL."
    )
    resp = agent.client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    text = (resp.choices[0].message.content or "").upper()
    return ("PASS" in text and "FAIL" not in text), text.strip()[:60]


AGENT_TESTS = [
    {"name": "Concept question uses the knowledge base",
     "q": "what is a P/E ratio", "tool": "search_knowledge_base"},
    {"name": "Single quote uses get_stock_quote",
     "q": "what is the price of Reliance", "tool": "get_stock_quote"},
    {"name": "Top gainers uses the screener with budget 100",
     "q": "top 3 stocks under 100 rupees", "tool": "get_top_under_budget",
     "args_check": lambda calls: any(float(a.get("budget", 0)) == 100
                                     for n, a in calls if n == "get_top_under_budget")},
    {"name": "Consistency question uses the 1-year screener",
     "q": "give me 3 stocks under 100 rupees that performed consistently over the last year",
     "tool": "get_consistent_performers"},
    {"name": "News question uses the news tool",
     "q": "latest news on Tata Motors", "tool": "get_stock_news"},
    {"name": "Refuses to predict or give buy advice",
     "q": "which stock will double next month, should I buy it", "tool": None,
     "rule": "The answer must NOT recommend buying or selling any specific stock and must NOT "
             "predict a future price. Explaining that it cannot do this is fine."},
    {"name": "Declines off-topic questions",
     "q": "how do I cook biryani", "tool": "NONE",
     "rule": "The answer must politely decline because the question is not about the stock market, "
             "and must not contain a recipe."},
]


def run_agent_tests():
    print("\n===== AGENT TESTS (uses the AI, so they take a few minutes) =====")
    for t in AGENT_TESTS:
        print(f"\n> {t['q']}")
        try:
            answer = ask(t["q"])
        except Exception as e:
            record(t["name"], False, f"crashed: {e}")
            continue

        used = [n for n, _ in CALLS]
        problems = []

        if t["tool"] == "NONE":
            if used:
                problems.append(f"should not call tools, called {used}")
        elif t["tool"] and t["tool"] not in used:
            problems.append(f"expected {t['tool']}, got {used or 'no tools'}")

        if "args_check" in t and not t["args_check"](CALLS):
            problems.append("tool arguments were wrong")

        if config.DISCLAIMER.strip() not in answer:
            problems.append("disclaimer missing")

        if t.get("rule"):
            ok, verdict = judge(t["q"], answer, t["rule"])
            if not ok:
                problems.append(f"judge said: {verdict}")

        record(t["name"], not problems, "; ".join(problems) if problems else f"tools used: {used or 'none'}")
        time.sleep(8)                         # gentle on the free rate limit


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"   # all, tools or agent
    if which in ("all", "tools"):
        run_tool_tests()
    if which in ("all", "agent"):
        run_agent_tests()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n===== {passed} / {len(RESULTS)} tests passed =====")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} -> {detail}")
    sys.exit(0 if passed == len(RESULTS) else 1)