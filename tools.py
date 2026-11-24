import os
import time
import pandas as pd
import yfinance as yf
import config


# ---------- Helpers ----------
def load_universe():
    """Reads the NSE CSV and returns a list like ['RELIANCE.NS', 'TCS.NS', ...]"""
    df = pd.read_csv(config.UNIVERSE_CSV)
    df.columns = df.columns.str.strip()
    df = df[df["SERIES"].str.strip() == "EQ"]
    return [s.strip() + ".NS" for s in df["SYMBOL"]]


def _clean_symbol(symbol):
    """'reliance' -> 'RELIANCE.NS'. Leaves .NS / .BO endings alone."""
    s = str(symbol).strip().upper()
    if not (s.endswith(".NS") or s.endswith(".BO")):
        s += ".NS"
    return s


def _short(sym):
    return sym.replace(".NS", "").replace(".BO", "")


def _ttl_cache(seconds):
    """Remembers a function's results in memory for `seconds`. Errors are never remembered."""
    def decorator(fn):
        store = {}

        def wrapper(*args):
            now = time.time()
            if args in store and now - store[args][0] < seconds:
                return store[args][1]
            result = fn(*args)
            if not (isinstance(result, dict) and "error" in result):
                store[args] = (now, result)
            return result
        return wrapper
    return decorator


def _file_age_minutes(path):
    return (time.time() - os.path.getmtime(path)) / 60


def _download_batch(batch, period=None):
    """Downloads one batch. Tries twice, because Yahoo sometimes rate-limits."""
    period = period or config.HISTORY_PERIOD
    for attempt in range(2):
        try:
            data = yf.download(batch, period=period, interval="1d",
                               group_by="ticker", progress=False, threads=True)
            if data is not None and not data.empty:
                return data
        except Exception:
            pass
        time.sleep(3)
    return None


# ---------- 1-WEEK market scan with disk cache ----------
def _scan_all_stocks():
    """Scans every NSE stock once (1 week of data) and returns a DataFrame."""
    symbols = load_universe()
    total = len(symbols)
    print(f"Scanning {total} stocks (one-time, results will be saved)...")
    rows = []
    failed_batches = 0

    for i in range(0, total, config.BATCH_SIZE):
        batch = symbols[i:i + config.BATCH_SIZE]
        print(f"  checked {min(i + config.BATCH_SIZE, total)} / {total}")
        data = _download_batch(batch)
        if data is None:
            failed_batches += 1
            continue

        for sym in batch:
            try:
                d = data[sym].dropna()
                if len(d) < 2:
                    continue
                last, prev, first = d.iloc[-1], d.iloc[-2], d.iloc[0]
                if last["Volume"] < config.MIN_VOLUME:
                    continue
                price = float(last["Close"])
                rows.append({
                    "symbol": _short(sym),
                    "price": round(price, 2),
                    "day_change_pct": round((price / prev["Close"] - 1) * 100, 2),
                    "week_change_pct": round((price / first["Close"] - 1) * 100, 2),
                    "volume": int(last["Volume"]),
                })
            except Exception:
                continue

    df = pd.DataFrame(rows, columns=["symbol", "price", "day_change_pct", "week_change_pct", "volume"])

    if failed_batches > 2:
        print(f"WARNING: {failed_batches} batches failed, so results are incomplete and will NOT be saved.")
    elif not df.empty:
        df.to_csv(config.CACHE_FILE, index=False)
        print(f"Saved {len(df)} stocks to {config.CACHE_FILE}")
    return df


def _load_market_data():
    """Uses the saved file if it is fresh. Otherwise scans again."""
    if os.path.exists(config.CACHE_FILE):
        age = _file_age_minutes(config.CACHE_FILE)
        if age < config.CACHE_MINUTES:
            print(f"Using saved market data ({age:.0f} min old).")
            return pd.read_csv(config.CACHE_FILE, dtype={"symbol": str})
    return _scan_all_stocks()


# ---------- 1-YEAR market scan with disk cache ----------
YEAR_COLUMNS = ["symbol", "price", "return_1y_pct", "months_up", "months_total",
                "months_up_pct", "volatility_pct", "max_drawdown_pct",
                "above_200dma", "consistency_score"]


def _scan_year_metrics():
    """Downloads 1 year of data for every liquid stock and computes consistency metrics."""
    market = _load_market_data()          # gives us the list of liquid stocks
    if market.empty:
        return pd.DataFrame(columns=YEAR_COLUMNS)

    symbols = [s + ".NS" for s in market["symbol"]]
    total = len(symbols)
    print(f"Analysing 1 year of data for {total} stocks (one-time, results will be saved)...")
    rows = []
    failed_batches = 0

    for i in range(0, total, config.BATCH_SIZE):
        batch = symbols[i:i + config.BATCH_SIZE]
        print(f"  analysed {min(i + config.BATCH_SIZE, total)} / {total}")
        data = _download_batch(batch, "1y")
        if data is None:
            failed_batches += 1
            continue

        for sym in batch:
            try:
                close = data[sym].dropna()["Close"]
                if len(close) < 200:          # skips new listings with too little history
                    continue
                price = float(close.iloc[-1])
                daily = close.pct_change().dropna()
                volatility = float(daily.std() * (252 ** 0.5) * 100)
                if volatility == 0:
                    continue
                ret = (price / float(close.iloc[0]) - 1) * 100

                # roughly-monthly checkpoints: every 21 trading days, counting back from today
                idx = list(range(len(close) - 1, -1, -21))[:13][::-1]
                monthly = close.iloc[idx].pct_change().dropna()
                months_total = int(len(monthly))
                months_up = int((monthly > 0).sum())
                if months_total < 10:
                    continue

                drawdown = float(((close / close.cummax()) - 1).min() * 100)
                above_200 = bool(price > close.tail(200).mean())

                rows.append({
                    "symbol": _short(sym),
                    "price": round(price, 2),
                    "return_1y_pct": round(ret, 2),
                    "months_up": months_up,
                    "months_total": months_total,
                    "months_up_pct": round(months_up / months_total * 100, 1),
                    "volatility_pct": round(volatility, 2),
                    "max_drawdown_pct": round(drawdown, 2),
                    "above_200dma": above_200,
                    "consistency_score": round(ret / volatility, 2),
                })
            except Exception:
                continue

    df = pd.DataFrame(rows, columns=YEAR_COLUMNS)

    if failed_batches > 2:
        print(f"WARNING: {failed_batches} batches failed, so results are incomplete and will NOT be saved.")
    elif not df.empty:
        df.to_csv(config.YEAR_CACHE_FILE, index=False)
        print(f"Saved {len(df)} stocks to {config.YEAR_CACHE_FILE}")
    return df


def _load_year_data():
    """Uses the saved 1-year file if it is fresh. Otherwise rebuilds it."""
    if os.path.exists(config.YEAR_CACHE_FILE):
        age_hours = _file_age_minutes(config.YEAR_CACHE_FILE) / 60
        if age_hours < config.YEAR_CACHE_HOURS:
            print(f"Using saved 1-year data ({age_hours:.1f} hours old).")
            return pd.read_csv(config.YEAR_CACHE_FILE, dtype={"symbol": str})
    return _scan_year_metrics()


# ---------- TOOL 1: today's top gainers ----------
def get_top_under_budget(budget, top_n=config.DEFAULT_TOP_N, min_price=config.MIN_PRICE):
    """Top stocks priced between `min_price` and `budget`, ranked by today's % gain."""
    df = _load_market_data()
    if df.empty:
        return []
    df = df[(df["price"] <= budget) & (df["price"] >= min_price)]
    if df.empty:
        return []
    df = df.sort_values("day_change_pct", ascending=False)
    return df.head(top_n).to_dict(orient="records")


# ---------- TOOL 2: quote for ONE stock ----------
@_ttl_cache(300)   # remembered for 5 minutes
def get_stock_quote(symbol):
    """Price, day change, ~52-week high/low, plus P/E and market cap when available."""
    sym = _clean_symbol(symbol)
    try:
        t = yf.Ticker(sym)
        h = t.history(period="1y")
        if h.empty or len(h) < 2:
            return {"error": f"No data found for '{symbol}'. Ask the user for the exact NSE symbol."}

        price = float(h["Close"].iloc[-1])
        prev = float(h["Close"].iloc[-2])
        result = {
            "symbol": _short(sym),
            "price": round(price, 2),
            "day_change_pct": round((price / prev - 1) * 100, 2),
            "week_52_high": round(float(h["High"].max()), 2),
            "week_52_low": round(float(h["Low"].min()), 2),
            "volume": int(h["Volume"].iloc[-1]),
        }

        try:
            info = t.info
            result["name"] = info.get("longName")
            result["sector"] = info.get("sector")
            pe = info.get("trailingPE")
            result["pe_ratio"] = round(pe, 2) if pe else None
            mc = info.get("marketCap")
            result["market_cap_crore"] = round(mc / 1e7) if mc else None
        except Exception:
            pass

        return result
    except Exception as e:
        return {"error": str(e)}


# ---------- TOOL 3: performance over a period for ONE stock ----------
VALID_PERIODS = ("1mo", "3mo", "6mo", "1y")


@_ttl_cache(300)
def get_price_history(symbol, period="1y"):
    """Return, high, low, volatility and biggest fall over 1mo / 3mo / 6mo / 1y."""
    if period not in VALID_PERIODS:
        return {"error": f"period must be one of {VALID_PERIODS}"}
    sym = _clean_symbol(symbol)
    try:
        h = yf.Ticker(sym).history(period=period)
        if h.empty or len(h) < 5:
            return {"error": f"Not enough data for '{symbol}'. Ask the user for the exact NSE symbol."}

        close = h["Close"]
        daily = close.pct_change().dropna()
        return {
            "symbol": _short(sym),
            "period": period,
            "start_price": round(float(close.iloc[0]), 2),
            "end_price": round(float(close.iloc[-1]), 2),
            "return_pct": round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
            "period_high": round(float(h["High"].max()), 2),
            "period_low": round(float(h["Low"].min()), 2),
            "volatility_pct": round(float(daily.std() * (252 ** 0.5) * 100), 2),
            "max_drawdown_pct": round(float(((close / close.cummax()) - 1).min() * 100), 2),
            "trading_days": int(len(close)),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------- TOOL 4: compare 2 to 5 stocks ----------
def compare_stocks(symbols, period="1y"):
    """Quote + period performance for each stock, side by side."""
    if not isinstance(symbols, list) or len(symbols) < 2:
        return {"error": "Give at least 2 stock symbols to compare."}
    out = []
    for s in symbols[:5]:
        quote = get_stock_quote(s)
        history = get_price_history(s, period)
        if "error" in quote or "error" in history:
            out.append({"symbol": str(s).upper(), "error": quote.get("error") or history.get("error")})
            continue
        out.append({
            "symbol": quote["symbol"],
            "price": quote["price"],
            "pe_ratio": quote.get("pe_ratio"),
            "market_cap_crore": quote.get("market_cap_crore"),
            "week_52_high": quote["week_52_high"],
            "week_52_low": quote["week_52_low"],
            "period": period,
            "return_pct": history["return_pct"],
            "volatility_pct": history["volatility_pct"],
            "max_drawdown_pct": history["max_drawdown_pct"],
        })
    return out


# ---------- TOOL 5: consistent 1-year performers ----------
def get_consistent_performers(budget, top_n=config.DEFAULT_TOP_N, min_price=config.MIN_PRICE):
    """Stocks between min_price and budget that performed steadily over the past year."""
    criteria = (f"1-year return above 0%; at least {config.MIN_MONTHS_UP_PCT}% of monthly periods positive; "
                f"max drawdown no worse than -{config.MAX_DRAWDOWN_LIMIT}%; price above 200-day average. "
                f"Ranked by consistency score (1-year return divided by volatility). "
                f"New listings with under about 200 trading days of data are excluded.")

    df = _load_year_data()
    if df.empty:
        return {"criteria": criteria, "stocks": []}

    df = df[(df["price"] <= budget) & (df["price"] >= min_price)]
    df = df[(df["return_1y_pct"] > 0)
            & (df["months_up_pct"] >= config.MIN_MONTHS_UP_PCT)
            & (df["max_drawdown_pct"] >= -config.MAX_DRAWDOWN_LIMIT)
            & (df["above_200dma"])]
    df = df.sort_values("consistency_score", ascending=False).head(top_n)

    cols = ["symbol", "price", "return_1y_pct", "months_up", "months_total",
            "volatility_pct", "max_drawdown_pct", "consistency_score"]
    return {"criteria": criteria, "stocks": df[cols].to_dict(orient="records")}


# ---------- Web search helper (DuckDuckGo via the free `ddgs` package) ----------
NEWS_PERIODS = {"day": "d", "week": "w", "month": "m"}


def _ddgs_call(kind, query, **kwargs):
    """Runs a DuckDuckGo search. Returns (results, error_message)."""
    try:
        from ddgs import DDGS
    except ImportError:
        return None, "Search library not installed. Run: pip install ddgs"
    try:
        client = DDGS()
        search = client.news if kind == "news" else client.text
        return search(query=query, **kwargs), None
    except Exception as e:
        return None, f"Search failed: {e}"


# ---------- TOOL 6: recent news about ONE stock ----------
@_ttl_cache(600)   # remembered for 10 minutes
def get_stock_news(symbol, company_name="", period="week"):
    """Recent news headlines for a stock. period: day, week or month."""
    if period not in NEWS_PERIODS:
        period = "week"
    name = (company_name or "").strip() or _short(_clean_symbol(symbol))
    query = f"{name} share stock India"[:200]
    limit = NEWS_PERIODS[period]

    results, err = _ddgs_call("news", query, region="in-en",
                              timelimit=limit, max_results=config.NEWS_MAX_RESULTS)
    if not err and not results:          # try again without the region filter
        results, err = _ddgs_call("news", query, timelimit=limit,
                                  max_results=config.NEWS_MAX_RESULTS)
    if err:
        return {"error": err}

    articles = [{
        "title": r.get("title"),
        "source": r.get("source"),
        "date": r.get("date"),
        "summary": (r.get("body") or "")[:250],
        "url": r.get("url") or r.get("href"),
    } for r in (results or [])]
    return {"query_used": query, "period": period, "articles": articles}


# ---------- TOOL 7: general web search ----------
@_ttl_cache(600)
def search_web(query):
    """General web search for current facts, rules, or topics the other tools can't cover."""
    query = str(query).strip()[:200]
    if not query:
        return {"error": "Empty search query."}

    results, err = _ddgs_call("text", query, region="in-en", max_results=config.SEARCH_MAX_RESULTS)
    if err:
        return {"error": err}

    items = [{
        "title": r.get("title"),
        "summary": (r.get("body") or "")[:250],
        "url": r.get("href") or r.get("url"),
    } for r in (results or [])]
    return {"query_used": query, "results": items}


# ---------- Quick manual test (fast: no market scan) ----------
if __name__ == "__main__":
    print("--- News test ---")
    news = get_stock_news("RELIANCE", "Reliance Industries", "week")
    if "error" in news:
        print("ERROR:", news["error"])
    else:
        for a in news["articles"]:
            print(f"- {a['title']} ({a['source']}, {a['date']})")

    print("\n--- Web search test ---")
    web = search_web("NSE circuit limit rules explained")
    if "error" in web:
        print("ERROR:", web["error"])
    else:
        for r in web["results"]:
            print(f"- {r['title']} | {r['url']}")