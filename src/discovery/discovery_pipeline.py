"""
LLM-first discovery pipeline for extracting structured evidence from chunks.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.paper_processing.pdf_text import extract_pdf_pages, save_raw_pages
from src.paper_processing.chunk_builder import build_scientific_chunks, save_scientific_chunks
from src.discovery.structured_extractor import extract_chunk_evidence
from src.schemas.evidence_schema import ChunkEvidence
from src.discovery.discovery_formatters import format_extracted_evidence_for_review


def _model_dump(obj) -> dict:
    return obj.model_dump()


def save_json(
    data: Any,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_text(
    text: str,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)


def _deduplicate_items(
    items: list[dict[str, Any]],
    key_fields: list[str],
) -> list[dict[str, Any]]:
    seen = set()
    unique = []

    for item in items:
        key = tuple(
            str(item.get(field, "")).strip().lower()
            for field in key_fields
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def merge_chunk_evidence(
    chunks: list[ChunkEvidence],
) -> dict[str, Any]:
    merged = {
        "equations": [],
        "parameters": [],
        "state_variables": [],
        "mechanisms": [],
        "tables": [],
        "figures": [],
        "observations": [],
        "missing_or_uncertain": [],
    }

    for chunk in chunks:
        data = _model_dump(chunk)

        merged["equations"].extend(data.get("equations", []))
        merged["parameters"].extend(data.get("parameters", []))
        merged["state_variables"].extend(data.get("state_variables", []))
        merged["mechanisms"].extend(data.get("mechanisms", []))
        merged["tables"].extend(data.get("tables", []))
        merged["figures"].extend(data.get("figures", []))
        merged["observations"].extend(data.get("observations", []))
        merged["missing_or_uncertain"].extend(data.get("missing_or_uncertain", []))

    merged["equations"] = _deduplicate_items(
        merged["equations"],
        ["equation_id", "raw_text", "page"],
    )

    merged["parameters"] = _deduplicate_items(
        merged["parameters"],
        ["symbol", "name", "value", "unit", "page"],
    )

    merged["state_variables"] = _deduplicate_items(
        merged["state_variables"],
        ["symbol", "name", "page"],
    )

    merged["mechanisms"] = _deduplicate_items(
        merged["mechanisms"],
        ["source_entity", "relation", "target_entity", "page"],
    )

    merged["tables"] = _deduplicate_items(
        merged["tables"],
        ["table_id", "page"],
    )

    merged["figures"] = _deduplicate_items(
        merged["figures"],
        ["figure_id", "page"],
    )

    return merged


def run_llm_first_discovery(
    pdf_path: str,
    max_chunks: int | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    pages = extract_pdf_pages(pdf_path)
    save_raw_pages(pages, "outputs/raw_pages.json")

    chunks = build_scientific_chunks(pages)
    save_scientific_chunks(chunks, "outputs/scientific_chunks.json")

    chunks_to_process = chunks[:max_chunks] if max_chunks is not None else chunks

    extracted_chunks = []

    for chunk in chunks_to_process:
        print(
            f"Extracting {chunk['chunk_id']} "
            f"(pages {chunk['page_start']}-{chunk['page_end']})"
        )

        evidence = extract_chunk_evidence(
            chunk=chunk,
            model=model,
        )

        extracted_chunks.append(evidence)

    chunk_outputs = [_model_dump(item) for item in extracted_chunks]
    save_json(chunk_outputs, "outputs/chunk_evidence.json")

    merged = merge_chunk_evidence(extracted_chunks)
    save_json(merged, "outputs/extracted_evidence.json")

    review_markdown = format_extracted_evidence_for_review(merged)
    save_text(review_markdown, "outputs/reviewed_model.md")

    return merged


def run_llm_first_discovery_markdown(
    pdf_path: str,
    max_chunks: int | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    evidence = run_llm_first_discovery(
        pdf_path=pdf_path,
        max_chunks=max_chunks,
        model=model,
    )

    return format_extracted_evidence_for_review(evidence)