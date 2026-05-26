import re
from collections import OrderedDict

from src.retrieval.context import retrieve_semantic_context


def search_mechanisms(
    vector_store,
    query: str,
    k: int = 6,
    entity: str | None = None,
) -> str:
    """Shared wrapper used by Q/A and model discovery."""
    return retrieve_semantic_context(
        vector_store=vector_store,
        query=query,
        context_type="mechanism",
        entity=entity,
        k=k,
    )


def _deduplicate_mechanism_docs(docs):
    unique = OrderedDict()

    for doc in docs:
        text = (doc.page_content or "").strip()

        if len(text) < 40:
            continue

        key = re.sub(r"\s+", " ", text[:700]).lower()

        if key not in unique:
            unique[key] = doc

    return list(unique.values())


def search_mechanism_docs(
    vector_store,
    query: str,
    k: int = 6,
):
    """
    Return raw mechanism-related documents.
    Used by discovery when metadata/page info is needed.
    """
    return vector_store.similarity_search(query, k=k)


def search_mechanism_evidence(
    vector_store,
    mechanism_queries,
    k_per_query: int = 6,
    max_total_chars: int = 15000,
):
    """
    Shared mechanism evidence retrieval used by
    model discovery and future Q/A.
    Returns formatted mechanism evidence text.
    """
    retrieved_docs = []

    for query in mechanism_queries:
        retrieved_docs.extend(
            search_mechanism_docs(vector_store, query=query, k=k_per_query)
        )

    unique_docs = _deduplicate_mechanism_docs(retrieved_docs)

    blocks = []
    total_chars = 0

    mechanism_markers = (
        "mechanism",
        "interaction",
        "feedback",
        "stimulation",
        "inhibition",
        "activation",
        "suppression",
        "transfer",
        "causal",
        "regulation",
    )

    for i, doc in enumerate(unique_docs, start=1):
        text = (doc.page_content or "").strip()

        if not text:
            continue

        text_lower = text.lower()

        if not any(marker in text_lower for marker in mechanism_markers):
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        page = metadata.get("page") or metadata.get("page_number") or "unknown"

        block = f"""
=== MECHANISM EVIDENCE {i} | page {page} ===
{text}
"""

        if total_chars + len(block) > max_total_chars:
            break

        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return "NO MECHANISM EVIDENCE RETRIEVED."

    return "\n".join(blocks)
