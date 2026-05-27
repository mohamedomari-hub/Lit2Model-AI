"""
Shared general paper text retrieval helpers.
"""

import re
from collections import OrderedDict

from langchain_core.documents import Document

from src.retrieval.metadata import format_doc_metadata
from src.retrieval.search_engine import multi_query_retrieve, retrieve_semantic_context


BROAD_PAPER_QUESTION_TERMS = [
    "aim",
    "objective",
    "purpose",
    "contribution",
    "abstract",
    "introduction",
    "conclusion",
    "what is this paper about",
    "summarize the paper",
    "summary of the paper",
]


BROAD_PAPER_QUERIES = [
    "abstract aim objective purpose contribution",
    "introduction aim objective purpose",
    "we aim to we propose this paper",
    "conclusion summary contribution",
    "this work we investigate we develop we present",
]


FRONT_MATTER_MARKERS = [
    "## abstract",
    "# abstract",
    "## introduction",
    "# introduction",
    "## conclusion",
    "# conclusion",
    "objective",
    "purpose",
    "we present",
    "we propose",
    "we investigate",
    "this work",
]


def _is_broad_paper_question(query: str) -> bool:
    query_lower = (query or "").lower()

    for term in BROAD_PAPER_QUESTION_TERMS:
        if term in query_lower:
            return True

    return False


def _get_front_matter_context(vector_store, query: str, max_docs: int = 5) -> str:
    if vector_store is None:
        return ""

    try:
        collection = vector_store._collection.get(
            include=["documents", "metadatas"]
        )
        documents = collection.get("documents", []) or []
        metadatas = collection.get("metadatas", []) or []
    except Exception:
        return ""

    blocks = []
    seen = set()

    for index, text in enumerate(documents):
        text = (text or "").strip()

        if not text:
            continue

        text_lower = text.lower()

        if not any(marker in text_lower for marker in FRONT_MATTER_MARKERS):
            continue

        key = re.sub(r"\s+", " ", text[:700]).lower()

        if key in seen:
            continue

        seen.add(key)

        metadata = {}
        if index < len(metadatas):
            metadata = metadatas[index] or {}

        doc = Document(page_content=text, metadata=metadata)
        header = format_doc_metadata(
            doc=doc,
            search_query=query,
            label="Front Matter Text",
        )
        blocks.append(f"{header}\n{text}")

        if len(blocks) >= max_docs:
            break

    return "\n\n---\n\n".join(blocks)


def search_paper(vector_store, query: str, k: int = 6) -> str:
    """Shared wrapper used by Q/A and model discovery."""
    if _is_broad_paper_question(query):
        queries = [query] + BROAD_PAPER_QUERIES

        front_matter_context = _get_front_matter_context(
            vector_store=vector_store,
            query=query,
        )

        semantic_context = multi_query_retrieve(
            vector_store=vector_store,
            queries=queries,
            label="Text Query",
            k=k,
        )

        parts = [
            part for part in [front_matter_context, semantic_context]
            if part and part.strip()
        ]
        context = "\n\n---\n\n".join(parts)

        print(
            "QA: broad_text_context_chars="
            f"{len(context)}"
        )

        return context

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
