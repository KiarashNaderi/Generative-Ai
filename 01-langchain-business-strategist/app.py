import streamlit as st
from strategist import BusinessStrategist

st.set_page_config(page_title="AI Business Strategist", layout="centered")

st.title("🧠 AI Business Strategist")
st.write("Talk to me about your business idea. I will help you build a strategy.")

if "strategist" not in st.session_state:
    st.session_state.strategist = BusinessStrategist()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Describe your idea or ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your business..."):
            answer = st.session_state.strategist.ask(user_input, session_id="streamlit-session")
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
