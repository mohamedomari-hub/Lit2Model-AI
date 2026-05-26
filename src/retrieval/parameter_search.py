import re
from collections import OrderedDict

from src.retrieval.context import retrieve_semantic_context


def search_parameters(
    vector_store,
    query: str,
    k: int = 6,
    parameter_name: str | None = None,
) -> str:
    """Shared wrapper used by Q/A and model discovery."""
    return retrieve_semantic_context(
        vector_store=vector_store,
        query=query,
        context_type="parameter",
        entity=parameter_name,
        k=k,
    )


def _deduplicate_parameter_docs(docs):
    unique = OrderedDict()

    for doc in docs:
        text = (doc.page_content or "").strip()

        if len(text) < 40:
            continue

        key = re.sub(r"\s+", " ", text[:700]).lower()

        if key not in unique:
            unique[key] = doc

    return list(unique.values())


def search_parameter_evidence(
    vector_store,
    parameter_queries,
    k_per_query: int = 6,
    max_total_chars: int = 15000,
):
    """
    Shared parameter evidence retrieval used by
    model discovery and future Q/A.
    Returns formatted parameter evidence text.
    """
    retrieved_docs = []

    for query in parameter_queries:
        retrieved_docs.extend(
            vector_store.similarity_search(query, k=k_per_query)
        )

    unique_docs = _deduplicate_parameter_docs(retrieved_docs)

    blocks = []
    total_chars = 0

    parameter_markers = (
        "parameter",
        "value",
        "unit",
        "estimated",
        "fitted",
        "fixed",
        "assumed",
        "calibrated",
        "rate constant",
        "coefficient",
    )

    for i, doc in enumerate(unique_docs, start=1):
        text = (doc.page_content or "").strip()

        if not text:
            continue

        text_lower = text.lower()

        if not any(marker in text_lower for marker in parameter_markers):
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        page = metadata.get("page") or metadata.get("page_number") or "unknown"

        block = f"""
=== PARAMETER EVIDENCE {i} | page {page} ===
{text}
"""

        if total_chars + len(block) > max_total_chars:
            break

        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return "NO PARAMETER EVIDENCE RETRIEVED."

    return "\n".join(blocks)
