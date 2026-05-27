from __future__ import annotations

from typing import Any


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def _short(text: str | None, max_chars: int = 350) -> str:
    text = _clean(text)
    if not text:
        return "not reported"
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


def _looks_like_equation(text: str) -> bool:
    text = text or ""
    return any(token in text for token in ["=", "d/dt", "∂", "'"])


def _dedupe(items: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    seen = set()
    out = []

    for item in items:
        key = tuple(_clean(item.get(k)).lower() for k in keys)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


def format_extracted_evidence_for_review(evidence: dict[str, Any]) -> str:
    lines = [
        "# Lit2Model-AI Extracted Evidence Review",
        "",
        "status: draft / requires human review",
        "",
        "---",
        "",
    ]

    equations = [
        eq for eq in evidence.get("equations", [])
        if _looks_like_equation(eq.get("raw_text", ""))
    ]

    equations = _dedupe(equations, ["equation_id", "equation_type", "raw_text"])

    equation_groups = {
        "state_equation": "State equations",
        "coupling_equation": "Coupling equations",
        "regulatory_function": "Regulatory functions",
        "derived_definition": "Derived definitions",
        "unknown": "Uncertain equation candidates",
    }

    for eq_type, title in equation_groups.items():
        items = [eq for eq in equations if eq.get("equation_type") == eq_type]

        if not items:
            continue

        lines.append(f"## {title}")
        lines.append("")

        for eq in items:
            label = eq.get("equation_id") or "candidate"
            page = eq.get("page") or "not reported"
            confidence = eq.get("confidence", "low")

            flags = []
            if confidence != "high":
                flags.append(f"confidence: {confidence}")
            if eq.get("needs_ocr"):
                flags.append("needs OCR")

            flag_text = f" ({'; '.join(flags)})" if flags else ""

            lines.append(f"### {label} — page {page}{flag_text}")
            lines.append("")
            lines.append("```text")
            lines.append(_short(eq.get("raw_text"), 1000))
            lines.append("```")

            variables = eq.get("variables") or []
            if variables:
                lines.append("")
                lines.append("variables: " + ", ".join(variables))

            if eq.get("review_note"):
                lines.append("")
                lines.append(f"review_note: {_short(eq.get('review_note'))}")

            lines.append("")

        lines.append("---")
        lines.append("")

    parameters = _dedupe(
        evidence.get("parameters", []),
        ["symbol", "value", "unit", "source"],
    )

    if parameters:
        lines.append("## Parameters")
        lines.append("")

        for p in parameters:
            symbol = p.get("symbol") or p.get("name") or "parameter"
            lines.append(f"### {symbol}")
            lines.append(f"- value: {p.get('value') or 'not reported'}")
            lines.append(f"- unit: {p.get('unit') or 'not reported'}")
            lines.append(f"- meaning: {_short(p.get('meaning'))}")
            lines.append(f"- source: {p.get('source') or 'not reported'}")
            lines.append(f"- page: {p.get('page') or 'not reported'}")
            lines.append(f"- confidence: {p.get('confidence', 'low')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    states = _dedupe(
        evidence.get("state_variables", []),
        ["symbol", "name"],
    )

    if states:
        lines.append("## State variables / model quantities")
        lines.append("")

        for s in states:
            symbol = s.get("symbol") or s.get("name") or "state"
            lines.append(f"### {symbol}")
            lines.append(f"- name: {s.get('name') or 'not reported'}")
            lines.append(f"- meaning: {_short(s.get('meaning'))}")
            lines.append(f"- unit: {s.get('unit') or 'not reported'}")
            lines.append(f"- page: {s.get('page') or 'not reported'}")
            lines.append(f"- confidence: {s.get('confidence', 'low')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    mechanisms = [
        m for m in evidence.get("mechanisms", [])
        if m.get("confidence") == "high"
    ]

    mechanisms = _dedupe(
        mechanisms,
        ["source_entity", "relation", "target_entity"],
    )

    if mechanisms:
        lines.append("## High-confidence mechanisms")
        lines.append("")

        for m in mechanisms[:20]:
            source = m.get("source_entity") or "unknown source"
            relation = m.get("relation") or "related_to"
            target = m.get("target_entity") or "unknown target"

            lines.append(f"- **{source}** — {relation} → **{target}**")
            lines.append(f"  - page: {m.get('page') or 'not reported'}")
            lines.append(f"  - evidence: {_short(m.get('evidence_text'))}")
            lines.append("")

        lines.append("---")
        lines.append("")

    tables = _dedupe(
        evidence.get("tables", []),
        ["table_id", "caption_or_title"],
    )

    if tables:
        lines.append("## Tables needing review")
        lines.append("")

        for t in tables:
            if not t.get("likely_contains_parameters"):
                continue

            table_id = t.get("table_id") or "table candidate"
            lines.append(f"### {table_id}")
            lines.append(f"- title/caption: {_short(t.get('caption_or_title'))}")
            lines.append(f"- purpose: {_short(t.get('purpose'))}")
            lines.append(f"- page: {t.get('page') or 'not reported'}")
            lines.append(f"- needs_table_extraction: {t.get('needs_table_extraction')}")
            lines.append(f"- confidence: {t.get('confidence', 'low')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    figures = _dedupe(
        evidence.get("figures", []),
        ["figure_id", "caption_or_title"],
    )

    if figures:
        lines.append("## Figures needing review")
        lines.append("")

        for f in figures:
            if not f.get("likely_contains_mechanism_graph"):
                continue

            figure_id = f.get("figure_id") or "figure candidate"
            lines.append(f"### {figure_id}")
            lines.append(f"- title/caption: {_short(f.get('caption_or_title'))}")
            lines.append(f"- purpose: {_short(f.get('purpose'))}")
            lines.append(f"- page: {f.get('page') or 'not reported'}")
            lines.append(f"- needs_vision: {f.get('needs_vision')}")
            lines.append(f"- confidence: {f.get('confidence', 'low')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    observations = evidence.get("observations", [])

    if observations:
        lines.append("## Key observations / measured outputs")
        lines.append("")

        for o in observations[:15]:
            lines.append(f"- quantity: {o.get('observed_quantity') or 'not reported'}")
            lines.append(f"  - description: {_short(o.get('description'))}")
            lines.append(f"  - page: {o.get('page') or 'not reported'}")
            lines.append("")

        lines.append("---")
        lines.append("")

    missing = [
        item for item in evidence.get("missing_or_uncertain", [])
        if "chunk text does not contain" not in item.lower()
    ]

    if missing:
        lines.append("## Missing / uncertain")
        lines.append("")

        for item in missing[:15]:
            lines.append(f"- {_short(item, 500)}")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("IMPORTANT: This is extracted evidence, not a validated model.")

    return "\n".join(lines)