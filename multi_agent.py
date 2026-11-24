import json
import sys
import config
import agent

# ---------- The two extra agents are just different system prompts with no tools ----------
REVIEWER_PROMPT = """You are a strict fact-checker for a stock market assistant.
You get DATA (the only source of truth, returned by tools) and a DRAFT answer written for the user.

Flag a problem when the DRAFT:
- states a price, percentage, ratio, count, date or stock symbol that is NOT in DATA (small rounding is fine)
- says a stock did something that DATA does not show
- recommends buying or selling, or predicts a future price

Do NOT flag general explanations of concepts, simple wording or formatting.
Reply with ONLY a JSON object and nothing else:
{"ok": true, "problems": []}
or
{"ok": false, "problems": ["short description of each problem"]}"""

WRITER_PROMPT = """You are an editor for a stock market assistant.
Rewrite the DRAFT so that it fixes every listed problem.
Rules:
- Use ONLY numbers and facts that appear in DATA.
- Delete any claim that DATA does not support.
- Remove any buy or sell advice and any price prediction.
- Keep the same format and tone. Tables stay tables.
- Do not add a disclaimer.
Return only the corrected answer."""


def _call(system, user, max_tokens=1500):
    """One plain AI call with no tools."""
    response = agent.client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def review(data, draft):
    """REVIEWER agent. Returns (ok, problems)."""
    raw = _call(REVIEWER_PROMPT, f"DATA:\n{data}\n\nDRAFT:\n{draft}")
    try:
        verdict = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        problems = [str(p) for p in verdict.get("problems", [])][:6]
        return bool(verdict.get("ok")), problems
    except Exception:
        return True, []        # an unreadable reply must not block a good answer


def rewrite(data, draft, problems):
    """WRITER agent. Returns the corrected answer."""
    listed = "\n".join(f"- {p}" for p in problems)
    return _call(WRITER_PROMPT, f"DATA:\n{data}\n\nDRAFT:\n{draft}\n\nPROBLEMS TO FIX:\n{listed}", 2000)


def collect_tool_data(messages):
    """Gathers the tool results from the current question only."""
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=0)
    parts = [m["content"] for m in messages[last_user + 1:] if m["role"] == "tool"]
    return "\n".join(parts)[:config.REVIEW_DATA_CHARS]


# ---------- The manager (orchestrator) ----------
def ask_team(messages, user_message):
    """Same inputs and outputs as agent.ask_agent, so it can be swapped in anywhere."""
    final = agent.ask_agent(messages, user_message)          # 1. Researcher
    draft = messages[-1]["content"]                          # the draft, without the disclaimer
    data = collect_tool_data(messages)

    if not data.strip():                                     # nothing to check against
        return final

    try:
        ok, problems = review(data, draft)                   # 2. Reviewer
        if ok or not problems:
            print("[Reviewer: PASS]")
            agent.log_event({"event": "review", "ok": True})
            return final

        print("[Reviewer found problems, the Writer is fixing them]")
        for p in problems:
            print("   -", p)
        fixed = rewrite(data, draft, problems)               # 3. Writer
        agent.log_event({"event": "review", "ok": False, "problems": problems})
        if fixed:
            messages[-1]["content"] = fixed                  # memory keeps the corrected version
            return fixed + config.DISCLAIMER
    except Exception as e:
        print(f"[Reviewer skipped: {e}]")                    # a review failure never breaks the answer
    return final


# ---------- Demo: proves the Reviewer catches mistakes ----------
def demo():
    data = '[{"symbol": "ABC", "price": 45.3, "day_change_pct": 9.8, "week_change_pct": 12.1}]'
    draft = ("ABC trades at Rs 54.3, up 9.8% today and 12.1% this week. "
             "It is a strong buy and should reach Rs 70 next month.")
    print("DATA :", data)
    print("DRAFT:", draft, "\n")
    ok, problems = review(data, draft)
    print("Reviewer verdict:", "PASS" if ok else "FAIL")
    for p in problems:
        print("   -", p)
    if not ok and problems:
        print("\nWriter's corrected version:\n", rewrite(data, draft, problems))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
        sys.exit()

    messages = [{"role": "system", "content": agent.build_system_prompt()}]
    print("Stock Agent TEAM ready (Researcher + Reviewer + Writer). Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        messages = agent.trim_history(messages)
        size_before = len(messages)
        try:
            print("\nAgent:", ask_team(messages, question), "\n")
        except Exception as e:
            del messages[size_before:]
            print(f"\nSomething went wrong: {e}\n")