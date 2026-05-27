"""
Shared equation retrieval helpers for Q/A and discovery.
"""

import os
import re
from collections import OrderedDict

from src.paper_processing.ocr import extract_visible_equations_with_gpt
from src.retrieval.search_engine import retrieve_equation_context_service


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


def _clean_candidate_key(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text[:1200]


def _get_page_label(metadata: dict) -> str:
    return (
        metadata.get("page")
        or metadata.get("page_number")
        or "unknown"
    )


def _get_equation_group_key(metadata: dict, text: str):
    image_path = metadata.get("image_path")
    equation_number = metadata.get("equation_number")
    page = _get_page_label(metadata)

    if image_path:
        return ("image", image_path)

    if equation_number:
        return ("equation", page, equation_number)

    return ("text", page, _clean_candidate_key(text))


def _format_equation_candidate(metadata: dict, text: str) -> str:
    page = _get_page_label(metadata)
    equation_number = metadata.get("equation_number") or "unknown"
    image_path = metadata.get("image_path") or "not reported"

    return f"""
[page {page} | equation {equation_number}]
image_path: {image_path}
{text.strip()}
"""


def _has_strong_equation_signal(text: str) -> bool:
    text = text or ""
    text_lower = text.lower()

    strong_markers = [
        "formula-not-decoded",
        "d/dt",
        "\\frac",
        "frac{",
        "∂",
    ]

    for marker in strong_markers:
        if marker in text_lower:
            return True

    if re.search(r"\bd\s*[A-Za-z][A-Za-z0-9_]*\s*/\s*dt\b", text):
        return True

    assignment_match = re.search(
        r"(^|\n)\s*[A-Za-z][A-Za-z0-9_{}\\\-\u2212]{0,40}\s*=",
        text,
    )

    if assignment_match:
        return True

    return False


def search_equation_candidates(
    vector_store,
    max_equation_candidates: int = 500,
    enable_ocr_repair: bool = False,
):
    """
    Return equation candidate blocks from retrieved text.
    Shared helper for model discovery and future Q/A.
    """
    equation_candidates = []
    raw_candidate_count = 0
    skipped_noise_count = 0

    try:
        collection = vector_store._collection.get(
            include=["documents", "metadatas"]
        )

        all_docs = collection.get("documents", []) or []
        all_meta = collection.get("metadatas", []) or []

    except Exception:
        all_docs = []
        all_meta = []

    grouped_candidates = OrderedDict()

    for index, text in enumerate(all_docs):
        metadata = {}

        if index < len(all_meta):
            metadata = all_meta[index] or {}

        if metadata.get("content_type") != "equation_candidate":
            continue

        if not text or len(text.strip()) < 20:
            continue

        raw_candidate_count += 1

        if not metadata.get("equation_number") and not _has_strong_equation_signal(text):
            skipped_noise_count += 1
            continue

        group_key = _get_equation_group_key(metadata, text)

        if group_key not in grouped_candidates:
            grouped_candidates[group_key] = {
                "metadata": metadata,
                "texts": [],
            }

        if text.strip() not in grouped_candidates[group_key]["texts"]:
            grouped_candidates[group_key]["texts"].append(text.strip())

    if grouped_candidates:
        for candidate in grouped_candidates.values():
            metadata = candidate["metadata"]
            text = "\n\n".join(candidate["texts"])
            candidate_block = _format_equation_candidate(metadata, text)

            if enable_ocr_repair:
                candidate_block = _repair_equation_candidate_with_ocr(
                    candidate_block
                )

            equation_candidates.append(candidate_block)

        deduped_candidates = list(
            OrderedDict(
                (
                    _clean_candidate_key(candidate),
                    candidate,
                )
                for candidate in equation_candidates
            ).values()
        )

        included_candidates = deduped_candidates[:max_equation_candidates]

        print(
            "DISCOVERY: equation candidates "
            f"raw={raw_candidate_count} "
            f"deduped={len(deduped_candidates)} "
            f"included={len(included_candidates)} "
            f"skipped_noise={skipped_noise_count}"
        )

        for index, candidate in enumerate(included_candidates[:10], start=1):
            first_line = next(
                (
                    line.strip()
                    for line in candidate.splitlines()
                    if line.strip()
                ),
                "unknown",
            )
            print(f"DISCOVERY: equation candidate {index}: {first_line}")

        return included_candidates

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

            raw_candidate_count += 1

            candidate_block = f"""
                [page {page}]
                {neighborhood}
                """

            if enable_ocr_repair:
                candidate_block = _repair_equation_candidate_with_ocr(
                    candidate_block
                )

            equation_candidates.append(candidate_block)

    equation_candidates = list(dict.fromkeys(equation_candidates))
    included_candidates = equation_candidates[:max_equation_candidates]

    print(
        "DISCOVERY: equation candidates "
        f"raw={raw_candidate_count} "
        f"deduped={len(equation_candidates)} "
        f"included={len(included_candidates)}"
    )

    for index, candidate in enumerate(included_candidates[:10], start=1):
        first_line = next(
            (
                line.strip()
                for line in candidate.splitlines()
                if line.strip()
            ),
            "unknown",
        )
        print(f"DISCOVERY: equation candidate {index}: {first_line}")

    return included_candidates
