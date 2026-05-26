import re
from collections import OrderedDict

from src.retrieval.context import retrieve_semantic_context


def search_paper(vector_store, query: str, k: int = 6) -> str:
    """Shared wrapper used by Q/A and model discovery."""
    return retrieve_semantic_context(
        vector_store=vector_store,
        query=query,
        context_type="text",
        k=k,
    )


def search_paper_docs(vector_store, query: str, k: int = 6):
    """
    Return raw retrieved documents from the vector store.
    Used by model discovery when metadata/page info is needed.
    """
    return vector_store.similarity_search(query, k=k)


def _deduplicate_paper_docs(docs):
    unique = OrderedDict()

    for doc in docs:
        text = (doc.page_content or "").strip()

        if len(text) < 40:
            continue

        key = re.sub(r"\s+", " ", text[:700]).lower()

        if key not in unique:
            unique[key] = doc

    return list(unique.values())


def search_discovery_context(
    vector_store,
    discovery_queries,
    k_per_query: int = 8,
    max_total_chars: int = 40000,
):
    """
    Shared discovery-grade paper retrieval.

    Returns:
    - unique_docs
    - formatted_context_text
    """
    retrieved_docs = []

    for query in discovery_queries:
        retrieved_docs.extend(
            search_paper_docs(vector_store, query=query, k=k_per_query)
        )

    unique_docs = _deduplicate_paper_docs(retrieved_docs)

    retrieved_blocks = []
    total_chars = 0

    for i, doc in enumerate(unique_docs, start=1):
        text = (doc.page_content or "").strip()

        if not text:
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        page = metadata.get("page") or metadata.get("page_number") or "unknown"

        block = f"""
=== RETRIEVED CONTEXT {i} | page {page} ===
{text}
"""

        if total_chars + len(block) > max_total_chars:
            break

        retrieved_blocks.append(block)
        total_chars += len(block)

    formatted_context_text = "".join(retrieved_blocks)

    return unique_docs, formatted_context_text
