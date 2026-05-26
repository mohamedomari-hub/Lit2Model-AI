import re
import fitz


def detect_table_number(query: str, table_number: str | None = None) -> str | None:
    if table_number:
        return str(table_number)

    match = re.search(r"\b(?:table|tab\.?)\s*(\d+)\b", query.lower())
    if match:
        return match.group(1)

    return None


def extract_table_text_from_pdf(pdf_path: str, table_number: str) -> str:
    table_pattern = re.compile(
        rf"\bTable\s+{re.escape(str(table_number))}\b|"
        rf"\bTable\s*{re.escape(str(table_number))}\s*:",
        flags=re.IGNORECASE,
    )

    pdf = fitz.open(pdf_path)

    best_page = None
    best_score = -1

    try:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            if not table_pattern.search(text):
                continue

            score = 0
            lower_text = text.lower()

            if "name" in lower_text:
                score += 2
            if "description" in lower_text:
                score += 2
            if "value" in lower_text:
                score += 2
            if "unit" in lower_text:
                score += 2
            if "source" in lower_text:
                score += 2
            if "rate and effect parameters" in lower_text:
                score += 5

            if score > best_score:
                best_score = score
                best_page = page_index

        if best_page is None:
            return f"Could not find Table {table_number} in the PDF text layer."

        page = pdf[best_page - 1]
        text = page.get_text("text")

        start_match = table_pattern.search(text)
        if not start_match:
            return f"Could not extract Table {table_number} text."

        table_text = text[start_match.start(): start_match.start() + 1800]

        return (
            f"TABLE {table_number} TEXT-LAYER CANDIDATE\n\n"
            f"Source page: {best_page}\n\n"
            f"Extracted table text:\n\n"
            f"{table_text}\n\n"
            "Review note: This table was extracted from the PDF text layer. "
            "Verify values against the original PDF before modelling."
        )

    finally:
        pdf.close()


def retrieve_table_context_service(
    vector_store,
    pdf_path: str | None,
    query: str,
    table_number: str | None = None,
) -> str:
    from src.retrieval import retrieve_semantic_context

    detected_table_number = detect_table_number(query, table_number)

    if pdf_path is None:
        return "Table retrieval skipped: active PDF path is not set."

    if detected_table_number:
        return extract_table_text_from_pdf(
            pdf_path=pdf_path,
            table_number=detected_table_number,
        )

    return retrieve_semantic_context(
        vector_store=vector_store,
        query=query,
        context_type="table",
        entity=None,
        k=2,
    )