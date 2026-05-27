"""
Builds page-aware scientific text chunks for extraction.
"""

from __future__ import annotations

import json
import os
from typing import Any


def build_scientific_chunks(
    pages: list[dict[str, Any]],
    max_chars: int = 4500,
    overlap_chars: int = 500,
) -> list[dict[str, Any]]:
    """
    Build page-aware chunks for LLM extraction.

    General-purpose:
    - not paper-specific
    - keeps page provenance
    - avoids whole-paper prompts
    """

    chunks = []
    buffer = ""
    page_start = None
    page_end = None
    chunk_id = 1

    for page in pages:
        page_number = page["page"]
        text = page.get("text", "")

        if not text.strip():
            continue

        if page_start is None:
            page_start = page_number

        candidate = buffer + "\n\n" + f"[PAGE {page_number}]\n" + text

        if len(candidate) <= max_chars:
            buffer = candidate
            page_end = page_number
            continue

        if buffer.strip():
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id}",
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": buffer.strip(),
                }
            )
            chunk_id += 1

        overlap = buffer[-overlap_chars:] if buffer else ""

        buffer = overlap + "\n\n" + f"[PAGE {page_number}]\n" + text
        page_start = page_number
        page_end = page_number

    if buffer.strip():
        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_id}",
                "page_start": page_start,
                "page_end": page_end,
                "text": buffer.strip(),
            }
        )

    return chunks


def save_scientific_chunks(
    chunks: list[dict[str, Any]],
    output_path: str = "outputs/scientific_chunks.json",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)