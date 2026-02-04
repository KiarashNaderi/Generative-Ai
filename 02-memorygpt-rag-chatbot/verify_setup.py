"""Quick smoke test for memory + QA flow.
Run in your activated env `ai-strategist`: python verify_setup.py
It will add a short note, then query it and print results.
"""

import warnings
try:
    from langchain import LangChainDeprecationWarning
except Exception:
    class LangChainDeprecationWarning(Warning):
        pass
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

if __name__ == "__main__":

    try:
        from memory_store import add_note_to_memory, get_vectorstore
        from rag import ask_question
    except Exception as e:
        print("Import error:", e)
        raise

    print("Testing get_vectorstore()...")
    vs = get_vectorstore()
    print("Vectorstore type:", type(vs))

    print("Adding test note...")
    add_note_to_memory("This is a short test note about OpenAI and python. The note mentions APIs and embeddings.")

    print("Querying...")
    answer, sources = ask_question("What is this note about?")
    print("Answer:\n", answer)
    print("Sources count:", len(sources))
    for i, s in enumerate(sources):
        print(f"--- Source {i+1} metadata: {getattr(s, 'metadata', None)} ---")
        print(getattr(s, 'page_content', '')[:500])
