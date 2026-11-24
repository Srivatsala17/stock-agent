import streamlit as st
from agent import trim_history, SYSTEM_PROMPT
from multi_agent import ask_team as ask_agent

st.set_page_config(page_title="Indian Stock Agent", page_icon="📈")
st.title("📈 Indian Stock Market Agent")
st.caption("Ask about NSE stocks, screeners, news or market concepts. Not financial advice.")


def new_memory():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


# Streamlit re-runs this whole file on every click, so state lives in session_state
if "messages" not in st.session_state:
    st.session_state.messages = new_memory()   # what the AI remembers
if "chat" not in st.session_state:
    st.session_state.chat = []                 # what the screen shows

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Try these")
    examples = [
        "Top 5 stocks under 100 rupees today",
        "Stocks between 200 and 300 rupees that performed consistently over the last year",
        "Compare TCS and Infosys over 1 year",
        "Latest news on Reliance",
        "What is a circuit limit?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = new_memory()
        st.session_state.chat = []
        st.rerun()

    st.info("The first screener question scans about 2,400 stocks and can take a few minutes. "
            "After that, answers are fast.")

# ---------- Show the conversation so far ----------
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Get a question (typed or from a sidebar button) ----------
prompt = st.chat_input("Ask about Indian stocks...")
if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Working on it... (market scans can take a few minutes)"):
            st.session_state.messages = trim_history(st.session_state.messages)
            size_before = len(st.session_state.messages)
            try:
                answer = ask_agent(st.session_state.messages, prompt)
            except Exception as e:
                del st.session_state.messages[size_before:]   # keep memory clean after an error
                answer = f"Something went wrong: {e}"
        answer = answer.replace("$", "\\$")    # stops Streamlit treating $ as math
        st.markdown(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})