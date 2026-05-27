from __future__ import annotations

import json
import os
from typing import Any

import fitz


def extract_pdf_pages(pdf_path: str) -> list[dict[str, Any]]:
    """
    Extract PDF text page by page.

    General-purpose:
    - works for QSP, PK/PD, PBPK, SIR, ODE, systems biology papers
    - does not interpret
    - does not call LLM
    """

    pages = []

    pdf = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""

            pages.append(
                {
                    "page": page_index,
                    "text": text.strip(),
                    "char_count": len(text),
                }
            )

    finally:
        pdf.close()

    return pages


def save_raw_pages(
    pages: list[dict[str, Any]],
    output_path: str = "outputs/raw_pages.json",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(pages, file, indent=2, ensure_ascii=False)