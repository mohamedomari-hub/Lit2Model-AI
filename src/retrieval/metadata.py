"""
Metadata formatting helpers for retrieved documents.
"""

def format_doc_metadata(doc, search_query: str, label: str) -> str:
    metadata = doc.metadata or {}

    source_pdf = metadata.get("source_pdf", "unknown_pdf")
    page = metadata.get("page", "unknown")
    section_index = metadata.get("section_index", "unknown")
    modality = metadata.get("modality", "unknown")
    content_type = metadata.get("content_type", "unknown_type")
    figure_number = metadata.get("figure_number", None)
    table_number = metadata.get("table_number", None)
    chunk_id = metadata.get("chunk_id", "unknown_chunk")

    return (
        f"[{label}: {search_query} | "
        f"Source: {source_pdf} | "
        f"Page: {page} | "
        f"Section: {section_index} | "
        f"Modality: {modality} | "
        f"Type: {content_type} | "
        f"Figure: {figure_number} | "
        f"Table: {table_number} | "
        f"Chunk ID: {chunk_id}]"
    )
