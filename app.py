from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import GeminiAgent


load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(
    page_title="AI Agent | Milestone 2",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #182b43);
    color: white;
}
.main .block-container {
    max-width: 900px;
    padding-top: 2rem;
}
h1 {
    color: #8be9fd;
}
.stChatMessage {
    border-radius: 14px;
    border: 1px solid #34465e;
}
div[data-testid="stChatInput"] {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Tool-Based AI Agent")
st.caption(
    "Milestone 2 · Gemini AI · Knowledge Search · Calculator"
)

with st.sidebar:
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    try:
        st.session_state.agent = GeminiAgent()
        st.session_state.agent_error = None
    except ValueError as exc:
        st.session_state.agent = None
        st.session_state.agent_error = str(exc)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.agent_error:
    st.warning(st.session_state.agent_error)

question = st.chat_input(
    "Ask about Python or enter a calculation..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.agent is None:
            result = {
                "answer": (
                    "The agent is not configured. Add your Gemini API key "
                    "to the .env file and restart Streamlit."
                ),
                "tool_calls": [],
                "ok": False
            }
        else:
            with st.spinner("Thinking and using tools..."):
                result = st.session_state.agent.answer(question)

        st.markdown(result["answer"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tool_calls": result["tool_calls"]
    })