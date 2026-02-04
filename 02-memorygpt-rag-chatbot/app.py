import warnings
try:
    from langchain import LangChainDeprecationWarning
except Exception:
    class LangChainDeprecationWarning(Warning):
        pass
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

import streamlit as st
import os
from memory_store import add_file_to_memory, add_note_to_memory
from rag import ask_question
from config import UPLOAD_DIR

st.set_page_config(page_title="MemoryGPT", layout="wide")
st.title("🧠 MemoryGPT - Your Second Brain")

# Sidebar: Add notes or files
st.sidebar.header("Add to Memory")

# Text note
note_text = st.sidebar.text_area("Write a note:")
if st.sidebar.button("Save Note"):
    if note_text.strip():
        add_note_to_memory(note_text)
        st.sidebar.success("Note added!")
    else:
        st.sidebar.warning("Note is empty.")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload a file (PDF or TXT)", type=["pdf", "txt"])
if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if st.sidebar.button("Add File to Memory"):
        add_file_to_memory(file_path)
        st.sidebar.success("File added!")

# Main chat
st.header("💬 Chat with your Memory")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask something about your memory:")
if st.button("Ask"):
    if question.strip():
        answer, sources = ask_question(question)
        st.session_state.chat_history.append(("You", question))
        st.session_state.chat_history.append(("MemoryGPT", answer))

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 MemoryGPT:** {message}")

# Show sources for last question
if st.session_state.chat_history:
    st.subheader("📚 Sources")
    try:
        _, sources = ask_question(st.session_state.chat_history[-2][1])
        for i, doc in enumerate(sources, 1):
            st.markdown(f"**Source {i}:** {doc.metadata}")
            st.markdown(doc.page_content[:500] + "...")
    except:
        pass
