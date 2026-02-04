from typing import Tuple, List

import warnings
try:
    from langchain import LangChainDeprecationWarning
except Exception:
    class LangChainDeprecationWarning(Warning):
        pass
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning) 

# Try to import optional LangChain pieces from community packages first.
HAS_LANGCHAIN = False
ChatOpenAI = None
RetrievalQA = None
try:
    try:
        from langchain_community.chat_models import ChatOpenAI
        from langchain.chains import RetrievalQA
    except Exception:
        try:
            from langchain_openai import ChatOpenAI
            from langchain.chains import RetrievalQA
        except Exception:
            try:
                from langchain.chat_models import ChatOpenAI
                from langchain.chains import RetrievalQA
            except Exception:
                ChatOpenAI = None
                RetrievalQA = None
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False

from memory_store import get_vectorstore
from config import OPENAI_API_KEY


def get_qa_chain():
    """
    Create a RetrievalQA chain when available; otherwise return a minimal callable that performs retrieval and optional LLM summarization.
    """
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # If LangChain is present, prefer its RetrievalQA
    if HAS_LANGCHAIN:
        try:
            llm = ChatOpenAI(
                temperature=0.2,
                openai_api_key=OPENAI_API_KEY,
                model_name="gpt-3.5-turbo",
            )
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
                )
            except Exception:
                qa_chain = RetrievalQA(llm=llm, retriever=retriever, return_source_documents=True)
            return qa_chain
        except Exception:
            # Fall through to fallback implementation
            pass

    # Fallback: a lightweight callable that performs retrieval and simple summarization
    class FallbackChain:
        def __init__(self, retriever):
            self.retriever = retriever

        def __call__(self, inputs):
            # accept either dict or query string
            query = inputs.get("query") if isinstance(inputs, dict) else inputs
            docs = self.retriever.get_relevant_documents(query)
            # create a simple answer by concatenating top docs
            concat = "\n\n".join([d.page_content for d in docs if getattr(d, "page_content", None)])
            # If an LLM is available, use it; otherwise return the concat as the answer
            try:
                if 'ChatOpenAI' in globals() and OPENAI_API_KEY:
                    llm = ChatOpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
                    prompt = f"Use the following context to answer the question:\n\nContext:\n{concat}\n\nQuestion: {query}\n\nAnswer concisely:"
                    out = llm.generate([{"text": prompt}])  # best-effort; API may vary
                    # normalize output
                    answer = ""
                    if hasattr(out, "generations"):
                        gens = out.generations
                        if gens and gens[0]:
                            answer = gens[0][0].text
                    if not answer and isinstance(out, dict):
                        answer = out.get("text") or out.get("answer") or ""
                    return {"result": answer or concat, "source_documents": docs}
            except Exception:
                return {"result": concat, "source_documents": docs}

    return FallbackChain(retriever)


def ask_question(question: str) -> Tuple[str, List]:
    """
    Ask a question from the memory and return the answer + sources.
    Handles both LangChain RetrievalQA and fallback chain formats.
    """
    qa_chain = get_qa_chain()
    # Some chains are callable, some have run()
    try:
        result = qa_chain.run(question) if hasattr(qa_chain, "run") else qa_chain({"query": question})
    except TypeError:
        # some callables expect dict input
        result = qa_chain({"query": question})

    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or result.get("output_text") or ""
        sources = result.get("source_documents") or result.get("sources") or []
    else:
        answer = result
        sources = []

    return answer, sources
