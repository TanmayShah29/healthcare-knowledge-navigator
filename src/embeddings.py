"""Provider-agnostic embeddings factory — sibling to llm.py's get_llm(), same
philosophy. Every place that needs a vector embedding (chunk ingestion, question
embedding at query time) goes through get_embeddings() instead of importing a
provider SDK directly, so swapping embedding providers is a .env change
(EMBEDDING_PROVIDER + EMBEDDING_MODEL + EMBEDDING_DIMENSIONS + matching key).

Cached as a module-level singleton since embedding clients are stateless and
cheap to reuse across many calls within a single ingest/query run.
"""
from langchain.embeddings import init_embeddings
from src.config import EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_KEY

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        kwargs = {}
        if EMBEDDING_API_KEY:
            kwargs["api_key"] = EMBEDDING_API_KEY
        _embeddings = init_embeddings(f"{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}", **kwargs)
    return _embeddings


def embed_text(text: str) -> list[float]:
    """Embed a single string. Thin wrapper so callers don't need to know
    whether the underlying client exposes embed_query vs embed_documents."""
    return get_embeddings().embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed multiple strings (used during ingestion)."""
    return get_embeddings().embed_documents(texts)
