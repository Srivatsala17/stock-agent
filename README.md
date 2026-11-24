# 📈 Indian Stock Market AI Agent

An agentic GenAI app that answers questions about NSE-listed stocks. You can ask for the top performers under a price budget, find stocks that performed consistently over a year, compare companies, get recent news, or learn market concepts.

> ⚠️ **Disclaimer:** This project is for learning. It is not financial advice. Do your own research and analysis before investing, and do not rely solely on this agent.

## Features

- Top gaining stocks within a price range today, with 1-week change
- Stocks that performed consistently over 1 year (return, monthly consistency, biggest fall, trend)
- Quote, price history and side-by-side comparison for named stocks
- Recent news headlines and web search
- Explains market concepts from its own knowledge base (RAG)
- Remembers lasting preferences, such as your usual budget
- A Reviewer agent checks answers against the tool data before you see them

## Screenshots

<!-- Add your screenshots here, for example: -->
<!-- ![Screener](screenshots/screener.png) -->

## GenAI concepts used

| Concept | Where |
|---|---|
| Tool calling and the agent loop | `agent.py` |
| System prompts and guardrails (no buy/sell advice, fixed disclaimer, prompt-injection rule) | `agent.py`, `config.py` |
| Short-term memory (chat history) and long-term memory (saved preferences) | `agent.py`, `memory.py` |
| RAG: chunking, embeddings, vector database (ChromaDB) | `knowledge.py`, `knowledge.txt` |
| Grounding with news and web search | `tools.py` |
| Multi-agent: Researcher, Reviewer, Writer | `multi_agent.py` |
| Evaluation: tool tests, behaviour tests, LLM-as-judge | `test_agent.py` |
| Logging and tracing | `agent.py` (`/log` command) |
| Caching | `tools.py` |

**Design choice:** all numbers (filtering, ranking, returns, volatility) are calculated in plain Python. The LLM only decides which tool to call and explains the results, so it cannot invent figures.

## Tech stack

Python, Groq API (free tier), yfinance, pandas, ChromaDB, ddgs (DuckDuckGo search), Streamlit.

## Setup

1. Clone the repository and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate      # Git Bash on Windows
   pip install -r requirements.txt
   ```
2. Get a free API key at console.groq.com and create a `.env` file in the project folder:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Download the NSE equity list (`EQUITY_L.csv`) from the NSE website and place it in the project folder.

## Run

```bash
streamlit run app.py          # browser chat UI
python agent.py               # terminal chat (commands: /profile, /forget, /log)
python multi_agent.py         # terminal chat with the Reviewer agent
python test_agent.py          # run all tests
```

The first screener question scans about 2,400 stocks and takes a few minutes. Results are cached, so later questions are fast.

## Example questions

- Top 5 stocks under 100 rupees today
- Stocks between 200 and 300 rupees that performed consistently over the last year
- Compare TCS and Infosys over 1 year
- Latest news on Reliance
- What is a circuit limit?

## Project structure

```
config.py          settings and the disclaimer
tools.py           market data tools, screeners, news, caching
knowledge.py       RAG search over knowledge.txt
knowledge.txt      the notes the agent learns from
memory.py          long-term preferences with validation
agent.py           the agent loop, prompts, logging
multi_agent.py     Researcher + Reviewer + Writer
app.py             Streamlit UI
test_agent.py      evaluation suite
```

## Limitations

- Prices come from Yahoo Finance (free) and can be delayed, incomplete or rate-limited.
- Free hosting often cannot reach Yahoo Finance, so this is designed to run locally.
- News comes from DuckDuckGo and may be unrelated or wrong.
- Past performance does not guarantee future results. The agent describes data and never predicts prices or recommends buying or selling.
- The knowledge-base notes are simplified. Check NSE, BSE and SEBI for current rules.