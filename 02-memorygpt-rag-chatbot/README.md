# 02-memorygpt-rag-chatbot

## Overview
This project is a **Retrieval-Augmented Generation (RAG) chatbot** powered by MemoryGPT. It is designed to provide intelligent, context-aware conversations by combining memory retention with retrieval from a structured knowledge base.  

The chatbot is capable of:
- Maintaining **multi-turn conversation memory**, so it remembers previous interactions.  
- Retrieving **relevant information** from a knowledge base to answer questions accurately.  
- Providing a **user-friendly interface** via Streamlit for interactive chatting.  

This project demonstrates a **modern AI application**, combining state-of-the-art techniques in natural language understanding, memory management, and retrieval-augmented generation.

---

## Project Files

- `02-memorygpt-rag-chatbot/data/` – Stores knowledge base documents used for retrieval.  
- `02-memorygpt-rag-chatbot/src/memorygpt.py` – Core logic for memory management and RAG.  
- `02-memorygpt-rag-chatbot/src/chatbot.py` – Integrates memory and retrieval to generate intelligent responses.  
- `02-memorygpt-rag-chatbot/src/utils.py` – Helper functions for processing data and documents.  
- `02-memorygpt-rag-chatbot/app.py` – Streamlit frontend that allows users to interact with the chatbot.  
- `02-memorygpt-rag-chatbot/requirements.txt` – Lists all Python dependencies required to run the project.

---

## How it works

1. **User input**: The user types a question in the Streamlit app.  
2. **Retrieval**: MemoryGPT searches the knowledge base for relevant information.  
3. **Memory integration**: The system incorporates previous conversation context for more coherent responses.  
4. **Response generation**: The chatbot generates an informed, context-aware reply.  

This combination of **memory and retrieval** ensures the chatbot is more accurate and intelligent than a standard Q&A bot.

---

## Why this project is valuable

- Demonstrates advanced AI concepts in **RAG and memory-driven chatbots**.  
- Modular and easy to extend for new documents or memory strategies.  
- Useful as a **learning project**, showcasing full AI pipeline from knowledge retrieval to user interaction.  

---

## Contact
Owner: **KiarashNaderi**
