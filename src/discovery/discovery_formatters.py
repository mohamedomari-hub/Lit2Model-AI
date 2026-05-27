"""
Discovery formatting helpers for evidence reviews and compact model reports.
"""

from __future__ import annotations

from typing import Any

def _evidence_clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def _evidence_short(text: str | None, max_chars: int = 350) -> str:
    text = _evidence_clean(text)
    if not text:
        return "not reported"
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


def _evidence_looks_like_equation(text: str) -> bool:
    text = text or ""
    return any(token in text for token in ["=", "d/dt", "∂", "'"])


def _evidence_dedupe(items: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    seen = set()
    out = []

    for item in items:
        key = tuple(_evidence_clean(item.get(k)).lower() for k in keys)

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
        if _evidence_looks_like_equation(eq.get("raw_text", ""))
    ]

    equations = _evidence_dedupe(equations, ["equation_id", "equation_type", "raw_text"])

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
            lines.append(_evidence_short(eq.get("raw_text"), 1000))
            lines.append("```")

            variables = eq.get("variables") or []
            if variables:
                lines.append("")
                lines.append("variables: " + ", ".join(variables))

            if eq.get("review_note"):
                lines.append("")
                lines.append(f"review_note: {_evidence_short(eq.get('review_note'))}")

            lines.append("")

        lines.append("---")
        lines.append("")

    parameters = _evidence_dedupe(
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
            lines.append(f"- meaning: {_evidence_short(p.get('meaning'))}")
            lines.append(f"- source: {p.get('source') or 'not reported'}")
            lines.append(f"- page: {p.get('page') or 'not reported'}")
            lines.append(f"- confidence: {p.get('confidence', 'low')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    states = _evidence_dedupe(
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
            lines.append(f"- meaning: {_evidence_short(s.get('meaning'))}")
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

    mechanisms = _evidence_dedupe(
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
            lines.append(f"  - evidence: {_evidence_short(m.get('evidence_text'))}")
            lines.append("")

        lines.append("---")
        lines.append("")

    tables = _evidence_dedupe(
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
            lines.append(f"- title/caption: {_evidence_short(t.get('caption_or_title'))}")
            lines.append(f"- purpose: {_evidence_short(t.get('purpose'))}")
            lines.append(f"- page: {t.get('page') or 'not reported'}")
            lines.append(f"- needs_table_extraction: {t.get('needs_table_extraction')}")
            lines.append(f"- confidence: {t.get('confidence', 'low')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    figures = _evidence_dedupe(
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
            lines.append(f"- title/caption: {_evidence_short(f.get('caption_or_title'))}")
            lines.append(f"- purpose: {_evidence_short(f.get('purpose'))}")
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
            lines.append(f"  - description: {_evidence_short(o.get('description'))}")
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
            lines.append(f"- {_evidence_short(item, 500)}")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("IMPORTANT: This is extracted evidence, not a validated model.")

    return "\n".join(lines)

# Compact controlled-discovery review formatter.

"""
Formats controlled discovery JSON into a compact scientific review.
"""




def _clean(value: Any) -> str:
    if value is None:
        return "not reported"

    text = " ".join(str(value).split())
    return text if text else "not reported"


def _is_empty(value: Any) -> bool:
    if value is None:
        return True

    text = str(value).strip().lower()

    return text in {
        "",
        "none",
        "null",
        "na",
        "n/a",
        "not applicable",
        "not reported",
    }


def _status_text(item: dict[str, Any]) -> str:
    status = _clean(item.get("status"))

    formula = item.get("formula")
    if _is_empty(formula):
        return status

    return f"{status}: {_clean(formula)}"


def _source_text(source: Any) -> str:
    if isinstance(source, list):
        return "; ".join(_clean(item) for item in source)

    return _clean(source)


def _bullet_list(items: list[Any]) -> list[str]:
    if not items:
        return ["- not reported"]

    return [f"- {_clean(item)}" for item in items]

def _equation_block(equation: Any) -> list[str]:
    """
    Simple readable equation display.

    Avoid raw HTML because it looks messy in the review editor.
    """

    eq = _clean(equation)

    return [
        "",
        "```text",
        eq,
        "```",
        "",
    ]

def _parameter_table(parameters: list[dict[str, Any]]) -> list[str]:
    if not parameters:
        return ["not reported"]

    lines = [
        "| Symbol | Value | Unit | Status |",
        "|---|---:|---|---|",
    ]

    for item in parameters:
        symbol = _clean(item.get("symbol"))
        value = _clean(item.get("value"))
        unit = _clean(item.get("unit"))
        status = _status_text(item)

        lines.append(f"| {symbol} | {value} | {unit} | {status} |")

    return lines


def _input_table(inputs: list[dict[str, Any]]) -> list[str]:
    if not inputs:
        return ["not reported"]

    lines = [
        "| Symbol | Value | Unit | Meaning |",
        "|---|---:|---|---|",
    ]

    seen = set()

    for item in inputs:

        symbol = _clean(item.get("symbol"))

        if symbol in seen:
            continue

        seen.add(symbol)

        value = _clean(item.get("value"))
        unit = _clean(item.get("unit"))
        meaning = _clean(item.get("meaning"))

        lines.append(
            f"| {symbol} | {value} | {unit} | {meaning} |"
        )

    return lines


def _render_ode(ode: dict[str, Any]) -> list[str]:
    state = _clean(ode.get("state"))
    meaning = _clean(ode.get("meaning"))
    unit = _clean(ode.get("unit"))

    title = f"### ODE: `{state}`"
    if meaning != "not reported":
        title += f" — {meaning}"

    lines = [
        title,
        "",
        "**Equation**",
    ]

    lines.extend(_equation_block(ode.get("equation")))


    initial_condition = ode.get("initial_condition") or {}
    ic_value = _clean(initial_condition.get("value"))
    ic_unit = _clean(initial_condition.get("unit"))

    if ic_unit == "not reported":
        ic_unit = unit

    lines.extend(
        [
            "**State / Initial condition**",
            "",
            "| State | Meaning | Unit | Initial value |",
            "|---|---|---|---:|",
            f"| {state} | {meaning} | {unit} | {ic_value} {ic_unit} |",
            "",
        ]
    )


    lines.extend(
        [
            "",
            "**Parameters**",
            "",
        ]
    )
    lines.extend(_parameter_table(ode.get("parameters", [])))

    lines.extend(
        [
            "",
            "**Inputs**",
            "",
        ]
    )
    lines.extend(_input_table(ode.get("inputs", [])))

    observed_data = ode.get("observed_data", [])

    if observed_data:
        lines.extend(
            [
                "",
                "**Observed data**",
                "",
            ]
        )
        lines.extend(_bullet_list(observed_data))

    lines.extend(
        [
            "",
            f"*{_source_text(ode.get('source'))}*",
            "",
            "---",
            "",
        ]
    )

    return lines


def _render_process_module(module: dict[str, Any]) -> list[str]:
    name = _clean(module.get("name"))

    lines = [
        f"### Process module: {name}",
        "",
        "**Equations**",
    ]

    equations = module.get("equations", [])

    if not equations:
        lines.extend(["", "not reported", ""])
    else:
        for equation in equations:
            lines.extend(_equation_block(equation))

    variables = module.get("variables", [])

    if variables:

        lines.extend(["**Key variables**", ""])

        for var in variables[:6]:

            symbol = _clean(var.get("symbol"))
            meaning = _clean(var.get("meaning"))

            if meaning == "not reported":
                lines.append(f"- `{symbol}`")
            else:
                lines.append(
                    f"- `{symbol}`: {meaning}"
                )

    parameters = module.get("parameters", [])

    if parameters:
        lines.extend(["", "**Parameters**", ""])
        lines.extend(_parameter_table(parameters))

    source = _source_text(
        module.get("source")
    )

    if source != "not reported":
        lines.extend(
            [
                "",
                f"*{source}*",
            ]
        )

    lines.extend(["", "---", ""])

    return lines


def _render_mechanisms(mechanisms: list[dict[str, Any]]) -> list[str]:
    if not mechanisms:
        return []

    lines = ["## Mechanisms", ""]

    for item in mechanisms:
        source = _clean(item.get("source"))
        relation = _clean(item.get("relation"))
        target = _clean(item.get("target"))
        evidence = _clean(item.get("evidence"))

        lines.append(
            f"- **{source}** "
            f"→ {relation} → "
            f"**{target}**"
        )

        if evidence != "not reported":
            lines.append(
                f"  - {evidence}"
            )

        lines.append("")

    lines.extend(["---", ""])
    return lines


def _render_flowchart(flowchart: str) -> list[str]:
    if not flowchart or not flowchart.strip():
        return []

    return [
        "## Mechanism flowchart",
        "",
        "```mermaid",
        flowchart.strip(),
        "```",
        "",
        "---",
        "",
    ]


def _render_missing(items: list[Any]) -> list[str]:
    if not items:
        return []

    lines = ["## Missing for simulation / review notes", ""]
    lines.extend(_bullet_list(items))
    lines.append("")
    return lines


def format_compact_review(evidence: dict[str, Any]) -> str:
    lines = [
        "# Model Discovery Review",
        "",
        "status: draft / requires human review",
        "",
        "## Model introduction",
        "",
        _clean(evidence.get("model_introduction")),
        "",
        "---",
        "",
        "## Equations / Parameters / Inputs / Observed data",
        "",
    ]

    for ode in evidence.get("odes", []):
        lines.extend(_render_ode(ode))

    for module in evidence.get("process_modules", []):
        lines.extend(_render_process_module(module))

    lines.extend(_render_mechanisms(evidence.get("mechanisms", [])))
    lines.extend(_render_flowchart(evidence.get("mechanism_flowchart", "")))
    lines.extend(_render_missing(evidence.get("missing_for_simulation", [])))

    lines.extend(
        [
            "---",
            "",
            "IMPORTANT: This is extracted evidence, not a validated model.",
        ]
    )

    return "\n".join(lines)