def make_chunk_key(doc):
    metadata = doc.metadata or {}

    return (
        metadata.get("source_pdf", "unknown_pdf"),
        metadata.get("page", "unknown"),
        metadata.get("modality", "unknown"),
        metadata.get("content_type", "unknown_type"),
        metadata.get("section_index", "unknown"),
        doc.page_content[:300],
    )


def metadata_order_value(value, default=999):
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sort_docs_for_discovery(docs):
    return sorted(
        docs,
        key=lambda doc: (
            metadata_order_value((doc.metadata or {}).get("page")),
            metadata_order_value((doc.metadata or {}).get("section_index")),
            (doc.metadata or {}).get("chunk_id", ""),
            doc.page_content[:80],
        ),
    )
