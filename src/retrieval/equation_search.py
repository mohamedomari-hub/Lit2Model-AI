import os
import re

from src.ingestion.ocr import extract_visible_equations_with_gpt
from src.retrieval.context import retrieve_equation_context_service


def search_equations(
    vector_store,
    query: str,
    k: int = 8,
    equation_number: str | None = None,
    pdf_path: str | None = None,
) -> str:
    """Shared wrapper used by Q/A and model discovery."""
    return retrieve_equation_context_service(
        vector_store=vector_store,
        pdf_path=pdf_path,
        query=query,
        equation_number=equation_number,
    )


def _looks_like_corrupted_equation_candidate(text: str) -> bool:
    if not text:
        return False

    suspicious_markers = [
        "formula-not-decoded",
        "\x12",
        "\x13",
        "\x14",
        "C10\ne",
        "C7\ne",
        "C10",
        "C7",
    ]

    if any(marker in text for marker in suspicious_markers):
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_symbol_lines = [
        line for line in lines
        if len(line) <= 4 and re.search(r"[A-Za-z0-9]", line)
    ]

    if len(short_symbol_lines) >= 3 and "=" in text:
        return True

    return False


def _extract_image_path_from_candidate_text(text: str) -> str | None:
    match = re.search(r"image_path:\s*(\S+)", text)

    if not match:
        return None

    return match.group(1).strip()


def _repair_equation_candidate_with_ocr(candidate_text: str) -> str:
    if not _looks_like_corrupted_equation_candidate(candidate_text):
        return candidate_text

    image_path = _extract_image_path_from_candidate_text(candidate_text)

    if not image_path or not os.path.exists(image_path):
        return candidate_text + "\n\n[Auto OCR skipped: crop image not found]"

    try:
        ocr_text = extract_visible_equations_with_gpt(
            image_path=image_path,
            model="gpt-4o-mini",
        )

        return f"""
AUTO OCR REPAIRED EQUATION CANDIDATE
Original candidate was detected as corrupted.
Crop image: {image_path}
Use OCR transcription below as higher-priority evidence.

{ocr_text}
"""

    except Exception as error:
        return (
            candidate_text
            + f"\n\n[Auto OCR failed: {type(error).__name__}: {error}]"
        )


def search_equation_candidates(
    vector_store,
    max_equation_candidates: int = 500,
):
    """
    Return equation candidate blocks from retrieved text.
    Shared helper for model discovery and future Q/A.
    """
    equation_candidates = []

    try:
        collection = vector_store._collection.get(
            include=["documents", "metadatas"]
        )

        all_docs = collection.get("documents", []) or []
        all_meta = collection.get("metadatas", []) or []

    except Exception:
        all_docs = []
        all_meta = []

    equation_pattern = re.compile(
        r"("
        r"\bd\s*/\s*dt\b|"
        r"\bd[A-Za-z][A-Za-z0-9_]*\s*/\s*dt\b|"
        r"\bEq\.?\s*\(?\d+\)?|"
        r"\bequation\s*\(?\d+\)?|"
        r"="
        r")",
        flags=re.IGNORECASE,
    )

    window = 5

    for index, text in enumerate(all_docs):
        if not text:
            continue

        page = "unknown"

        if index < len(all_meta):
            page = (
                all_meta[index].get("page")
                or all_meta[index].get("page_number")
                or "unknown"
            )

        lines = text.splitlines()

        for line_index, line in enumerate(lines):
            line = line.strip()

            if not line:
                continue

            if not equation_pattern.search(line):
                continue

            start = max(0, line_index - window)
            end = min(len(lines), line_index + window + 1)
            neighborhood = "\n".join(lines[start:end]).strip()

            if len(neighborhood) < 20:
                continue

            candidate_block = f"""
                [page {page}]
                {neighborhood}
                """

            candidate_block = _repair_equation_candidate_with_ocr(
                candidate_block
            )

            equation_candidates.append(candidate_block)

    equation_candidates = list(dict.fromkeys(equation_candidates))

    return equation_candidates[:max_equation_candidates]
