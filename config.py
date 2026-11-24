import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"

UNIVERSE_CSV = "EQUITY_L.csv"
DEFAULT_TOP_N = 10
MIN_VOLUME = 100_000      # skips stocks that are rarely traded
MIN_PRICE = 1             # skips stocks below Rs 1
BATCH_SIZE = 200
HISTORY_PERIOD = "7d"
MAX_HISTORY_TURNS = 6     # how many past questions the agent remembers
CACHE_FILE = "market_cache.csv"   # saved scan results
CACHE_MINUTES = 30                # how long saved data stays fresh
YEAR_CACHE_FILE = "year_cache.csv"   # saved 1-year results
YEAR_CACHE_HOURS = 12                # how long they stay fresh
MIN_MONTHS_UP_PCT = 65               # at least this % of months must be positive
MAX_DRAWDOWN_LIMIT = 40              # reject stocks that fell more than this % from a peak
NEWS_MAX_RESULTS = 5       # headlines returned per news search
SEARCH_MAX_RESULTS = 5     # results returned per web search
KB_FILE = "knowledge.txt"        # your notes
KB_DB_FOLDER = "chroma_db"       # where the vector database is saved
KB_RESULTS = 3                   # entries retrieved per question
KB_MAX_DISTANCE = 1.2            # lower = stricter match. Tune it in Step 5
PROFILE_FILE = "user_profile.json"   # saved long-term preferences
USE_LONG_TERM_MEMORY = True          # set to False to turn memory off
LOG_FILE = "agent_log.jsonl"         # one line per question
REVIEW_DATA_CHARS = 7000   # max tool data sent to the reviewer (protects the free token limit)

DISCLAIMER = (
    "\n\n⚠️ Disclaimer: This is not financial advice. "
    "Please do your own research and analysis before investing, "
    "and do not rely solely on this agent."
)