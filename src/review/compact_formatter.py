from __future__ import annotations

from typing import Any


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