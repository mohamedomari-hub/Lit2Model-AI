import re
from collections import OrderedDict

from src.modelling.table_extraction import retrieve_table_context_service
from src.retrieval.ranking import sort_docs_by_evidence_score


def search_tables(
    vector_store,
    query: str,
    k: int = 6,
    table_number: str | None = None,
    pdf_path: str | None = None,
) -> str:
    """Shared wrapper used by Q/A and model discovery."""
    return retrieve_table_context_service(
        vector_store=vector_store,
        pdf_path=pdf_path,
        query=query,
        table_number=table_number,
    )


def search_table_docs(vector_store, query: str, k: int = 6):
    """
    Return raw table-related documents from the vector store.
    Used by model discovery when metadata/page info is needed.
    """
    return vector_store.similarity_search(query, k=k)


def _deduplicate_table_docs(docs):
    unique = OrderedDict()

    for doc in docs:
        text = (doc.page_content or "").strip()

        if len(text) < 40:
            continue

        key = re.sub(r"\s+", " ", text[:700]).lower()

        if key not in unique:
            unique[key] = doc

    return list(unique.values())


def search_table_evidence(
    vector_store,
    table_queries,
    k_per_query: int = 6,
    max_total_chars: int = 15000,
):
    """
    Shared table evidence retrieval used by
    model discovery and future Q/A.
    Returns formatted table evidence text.
    """
    retrieved_docs = []

    for query in table_queries:
        retrieved_docs.extend(
            search_table_docs(vector_store, query=query, k=k_per_query)
        )

    unique_docs = _deduplicate_table_docs(retrieved_docs)
    unique_docs = sort_docs_by_evidence_score(
        unique_docs,
        evidence_type="table",
    )

    blocks = []
    total_chars = 0

    table_markers = (
        "table",
        "symbol",
        "| symbol",
        "value",
        "| value",
        "unit",
        "| unit",
        "initial value",
        "threshold",
        "parameter",
        "estimated",
        "fixed",
        "calibrated",
        "maximum effect",
        "half maximal",
        "rate constant",
        "clearance",
        "volume",
        "hill",
        "rate",
    )

    for i, doc in enumerate(unique_docs, start=1):
        text = (doc.page_content or "").strip()

        if not text:
            continue

        text_lower = text.lower()

        if not any(marker in text_lower for marker in table_markers):
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        page = metadata.get("page") or metadata.get("page_number") or "unknown"

        block = f"""
=== TABLE EVIDENCE {i} | page {page} ===
{text}
"""

        if total_chars + len(block) > max_total_chars:
            break

        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return "NO TABLE EVIDENCE RETRIEVED."

    return "\n".join(blocks)
