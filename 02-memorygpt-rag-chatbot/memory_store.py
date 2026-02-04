import os
import json
import math
from typing import List, Union

import warnings
try:
    from langchain import LangChainDeprecationWarning
except Exception:
    class LangChainDeprecationWarning(Warning):
        pass
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning) 

# Prefer community packages for LangChain to avoid deprecation warnings; fall back to older langchain imports when needed.
USE_LANGCHAIN = False
try:
    try:
        # preferred for newer LangChain versions
        from langchain_community.vectorstores import Chroma
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
    except Exception:
        try:
            # alt package that some users might have
            from langchain_chroma import Chroma
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
        except Exception:
            # last-resort fallback to older langchain layout
            from langchain.vectorstores import Chroma
            from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    USE_LANGCHAIN = True
except Exception:
    USE_LANGCHAIN = False
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict = None

    class PyPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            try:
                import PyPDF2
                texts = []
                with open(self.path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for p in reader.pages:
                        texts.append(p.extract_text() or "")
                return [Document(page_content="\n".join(texts), metadata={"source": self.path})]
            except Exception:
                return [Document(page_content="", metadata={"source": self.path})]

    class TextLoader:
        def __init__(self, path, encoding="utf-8"):
            self.path = path
            self.encoding = encoding

        def load(self):
            with open(self.path, "r", encoding=self.encoding, errors="ignore") as f:
                return [Document(page_content=f.read(), metadata={"source": self.path})]

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=800, chunk_overlap=150):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, docs):
            out = []
            for d in docs:
                text = d.page_content or ""
                i = 0
                while i < len(text):
                    chunk = text[i : i + self.chunk_size]
                    out.append(Document(page_content=chunk, metadata=getattr(d, "metadata", {})))
                    i += self.chunk_size - self.chunk_overlap
            return out

from embeddings import get_embedding_model
from config import UPLOAD_DIR, VECTOR_DB_DIR

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


class SimpleVectorStore:
    """A tiny in-memory vector store used when Chroma is not available."""

    def __init__(self, persist_directory=None, embedding_function=None, embedding=None):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.documents = []  # list of dicts: {page_content, metadata, embedding}
        self._load_persisted()

    def _store_path(self):
        if not self.persist_directory:
            return None
        return os.path.join(self.persist_directory, "simple_vectordb.json")

    def _load_persisted(self):
        path = self._store_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception:
                self.documents = []

    def _save(self):
        path = self._store_path()
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.documents, f)
            except Exception:
                pass

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self.embedding_function is None:
            raise RuntimeError("No embedding function available for SimpleVectorStore.")
        # huggingface wrapper exposes embed_documents; handle callable and methods
        if hasattr(self.embedding_function, "embed_documents"):
            return self.embedding_function.embed_documents(texts)
        if callable(self.embedding_function):
            return self.embedding_function(texts)
        if hasattr(self.embedding_function, "embed_query"):
            return [self.embedding_function.embed_query(t) for t in texts]
        raise RuntimeError("Embedding function is not usable")

    def add_documents(self, docs: List[Document]):
        texts = [d.page_content for d in docs]
        metadatas = [getattr(d, "metadata", {}) for d in docs]
        embs = self._embed(texts)
        for t, m, e in zip(texts, metadatas, embs):
            self.documents.append({"page_content": t, "metadata": m or {}, "embedding": e})
        self._save()

    def add_texts(self, texts: List[str], metadatas: List[dict] = None):
        if metadatas is None:
            metadatas = [None] * len(texts)
        embs = self._embed(texts)
        for t, m, e in zip(texts, metadatas, embs):
            self.documents.append({"page_content": t, "metadata": m or {}, "embedding": e})
        self._save()

    def _cosine(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def as_retriever(self, search_kwargs=None):
        k = 4
        if search_kwargs and isinstance(search_kwargs, dict):
            k = int(search_kwargs.get("k", k))

        store = self

        class Retriever:
            def get_relevant_documents(self, query: str):
                q_emb = store._embed([query])[0]
                scored = []
                for doc in store.documents:
                    sim = store._cosine(q_emb, doc["embedding"])
                    scored.append((sim, doc))
                scored.sort(key=lambda x: x[0], reverse=True)
                top = [Document(page_content=d["page_content"], metadata=d["metadata"]) for _, d in scored[:k]]
                return top

        return Retriever()

    # backward-compatible names
    def persist(self):
        self._save()

    def save(self):
        self._save()


def load_documents(file_path: str) -> List[Document]:
    """
    Load documents from a given file path. Supports PDF and TXT files.
    """
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def split_documents(documents: Union[List[Document], Document, List[str], str]) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    Accepts Document instances, lists of Documents, or raw strings.
    """
    if not isinstance(documents, list):
        documents = [documents]

    docs: List[Document] = []
    for d in documents:
        if isinstance(d, str):
            docs.append(Document(page_content=d))
        elif hasattr(d, "page_content"):
            docs.append(d)
        else:
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(docs)


def get_vectorstore():
    """
    Load or create a vector store (Chroma when available, otherwise a SimpleVectorStore).
    """
    embeddings = get_embedding_model()

    # Prefer a simple function that maps List[str] -> List[vector]
    def _embed(texts: List[str]):
        if hasattr(embeddings, "embed_documents"):
            return embeddings.embed_documents(texts)
        if callable(embeddings):
            return embeddings(texts)
        if hasattr(embeddings, "embed_query"):
            return [embeddings.embed_query(t) for t in texts]
        raise RuntimeError("Embedding model does not expose a compatible embed method.")

    if USE_LANGCHAIN:
        try:
            # Try to construct Chroma with common param name
            try:
                return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=_embed)
            except TypeError:
                return Chroma(persist_directory=VECTOR_DB_DIR, embedding=_embed)
        except Exception:
            # Chroma or its runtime dependency (e.g., chromadb) missing — fall back to in-memory store
            return SimpleVectorStore(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    else:
        return SimpleVectorStore(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)


def _add_to_vectordb(vectordb, docs: List[Document]):
    # Support different vectorstore APIs
    if hasattr(vectordb, "add_documents"):
        vectordb.add_documents(docs)
    elif hasattr(vectordb, "add_texts"):
        texts = [d.page_content for d in docs]
        metadatas = [getattr(d, "metadata", {}) for d in docs]
        vectordb.add_texts(texts=texts, metadatas=metadatas)
    else:
        raise RuntimeError("Unsupported vectorstore API for adding documents.")

    # Persist/save if available
    for func in ("persist", "save"):
        if hasattr(vectordb, func):
            getattr(vectordb, func)()
            break


def add_file_to_memory(file_path: str):
    """
    Load a file, split it into chunks, and add it to the vector database.
    """
    documents = load_documents(file_path)
    chunks = split_documents(documents)

    vectordb = get_vectorstore()
    _add_to_vectordb(vectordb, chunks)


def add_note_to_memory(note_text: str):
    """
    Add a plain text note to the vector database.
    """
    doc = Document(page_content=note_text, metadata={"source": "note"})
    chunks = split_documents([doc])

    vectordb = get_vectorstore()
    _add_to_vectordb(vectordb, chunks)
