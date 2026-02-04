"""Return an embedding model. Provide a lightweight fallback when heavy libs are missing."""

import warnings
try:
    from langchain import LangChainDeprecationWarning
except Exception:
    class LangChainDeprecationWarning(Warning):
        pass
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Prefer the dedicated huggingface package or community packages to avoid deprecation warnings
HuggingFaceEmbeddings = None
try:
    from langchain_huggingface import HuggingFaceEmbeddings as HF1
    HuggingFaceEmbeddings = HF1
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as HF2
        HuggingFaceEmbeddings = HF2
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings as HF3
            HuggingFaceEmbeddings = HF3
        except Exception:
            HuggingFaceEmbeddings = None


def get_embedding_model():
    """
    Returns a sentence-transformers embedding model when available, otherwise a simple deterministic fallback.
    The fallback implements `embed_documents` and `embed_query` so it is compatible with the rest of the code.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    if HuggingFaceEmbeddings is not None:
        try:
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception:
            # fall through to the lightweight fallback
            pass

    # Lightweight fallback embedding (deterministic hashing -> small float vector)
    class SimpleFallbackEmbedding:
        def embed_documents(self, texts):
            return [self._embed(t) for t in texts]

        def embed_query(self, text):
            return self._embed(text)

        def _embed(self, text: str):
            import hashlib
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            # Create a small fixed-size vector (32 floats) from hex digest
            vec = []
            for i in range(0, 64, 2):
                chunk = digest[i : i + 2]
                val = int(chunk, 16)
                vec.append((val % 255) / 255.0)
            return vec

    return SimpleFallbackEmbedding()
