"""Build and format indexes of scientific equations, tables, and figures."""

import json
import os
import re
from typing import Any

import fitz

ENABLE_LLM_CANDIDATE_CLASSIFIER = True
MAX_LLM_CANDIDATES = 8

from src.document.candidate_classifier import (
    classify_equation_candidate,
)

def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _looks_like_equation_line(line: str) -> bool:
    """
    Conservative equation-line detector.

    Goal:
    - keep real mathematical candidates
    - reject paper titles, prose, and random scientific sentences
    """

    line = _clean_line(line)

    if not line:
        return False

    if len(line) > 220:
        return False

    lower = line.lower()

    prose_starters = [
        "abstract",
        "introduction",
        "discussion",
        "conclusion",
        "figure",
        "table",
        "a pharmacokinetic",
        "the model",
        "the aim",
        "in this study",
        "results",
    ]

    if any(lower.startswith(starter) for starter in prose_starters):
        return False

    strong_math_patterns = [
        r"\bd[A-Za-z0-9_{}\-]+\/dt\b",
        r"\bd\s*[A-Za-z0-9_{}\-]+\s*\/\s*d[tT]\b",
        r"\\frac",
        r"\bexp\s*\(",
        r"\be\^",
        r"\bEmax\b",
        r"\bEC50\b",
        r"\bIC50\b",
    ]

    if any(re.search(pattern, line) for pattern in strong_math_patterns):
        return True

    if "=" not in line:
        return False

    # Require equation-like density, not just prose with equals.
    symbol_tokens = re.findall(
        r"[A-Za-zΑ-ω][A-Za-z0-9_{}\-]*|\d+|\+|\-|\*|\/|\^|\(|\)",
        line,
    )

    if len(symbol_tokens) < 5:
        return False

    # Reject obvious prose despite equals.
    words = re.findall(r"[A-Za-z]{3,}", line)
    if len(words) > 18:
        return False

    return True

def _has_unspaced_hyphenated_identifier(text: str) -> bool:
    """
    Detect model names like glucasec-dxm or effectdxm-gluca.

    These should be treated as one variable name, not subtraction.
    """
    return bool(
        re.search(
            r"\b[A-Za-z][A-Za-z0-9]*-[A-Za-z][A-Za-z0-9]*\b",
            text,
        )
    )


def _looks_like_inline_parameter_value(text: str) -> bool:
    """
    Detect prose-like inline parameter values such as:
    'F = 72%' or 'c0 = 20%'.

    These are useful, but they are parameter evidence, not model equations.
    """
    cleaned = _clean_line(text)

    if not re.search(r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*[-+]?\d", cleaned):
        return False

    prose_clues = [
        "estimated",
        "parameter",
        "value",
        "fraction",
        "dose",
        "around",
        "consider",
        "assume",
        "reported",
    ]

    return any(clue in cleaned.lower() for clue in prose_clues)

def _classify_equation_candidate(text: str) -> str:
    """
    Deterministic scientific equation classifier.

    Generalizes across:
    - QSP
    - PK/PD
    - PBPK
    - systems biology
    - SIR/SEIR
    - compartmental ODE models

    Conservative by design.
    """

    cleaned = _clean_line(text)
    lower = cleaned.lower()

    # --------------------------------------------------
    # 1. Inline parameter values
    # Example:
    # F = 72%
    # ka = 13.4
    # CL = 0.5 L/h
    # --------------------------------------------------

    if _looks_like_inline_parameter_value(cleaned):
        return "parameter_value_inline"

    # --------------------------------------------------
    # 2. State equations (ODEs)
    # Covers:
    # dC/dt
    # dCe/dt
    # d/dt Ce
    # d/dtCe
    # x' =
    # --------------------------------------------------

    state_patterns = [
        # Classical ODEs
        r"\bd[A-Za-z0-9_{}\-]+\s*/\s*d[tT]\b",

        # d/dt C
        r"\bd\s*/\s*d[tT]\s*[A-Za-z0-9_{}\-]+",

        # PDF broken forms:
        # dt C =
        r"\bd[tT]\s+[A-Za-z0-9_{}\-]+\s*=",

        # dtCe =
        r"\bd[tT][A-Za-z0-9_{}\-]+\s*=",

        # multiline PDF break:
        # d \n dt C =
        r"d\s*\n?\s*d[tT]\s*[A-Za-z0-9_{}\-]+\s*=",

        # x' =
        r"[A-Za-z0-9_{}]+\s*'\s*=",
    ]

    if any(re.search(pattern, cleaned) for pattern in state_patterns):
        return "state_equation"

# --------------------------------------------------
# 3. Regulatory functions
# General mechanistic modifiers
# --------------------------------------------------

    regulatory_terms = [
        # pharmacology
        "emax",
        "ec50",
        "ic50",

        # kinetics
        "vmax",
        "km",
        "michaelis",
        "mass action",

        # regulatory biology
        "hill",
        "stim",
        "stimulation",
        "inhib",
        "inhibition",
        "activation",
        "suppression",
        "induction",
        "repression",
        "feedback",

        # general mechanistic modifiers
        "effect",
        "response",
        "modifier",
        "uptake",
        "clearance",
        "binding",
        "transport",
        "degradation",
        "synthesis",
    ]

    contains_regulatory_term = any(
        term in lower for term in regulatory_terms
    )

    looks_equation_like = any([
        "=" in cleaned,
        "/" in cleaned,
        "*" in cleaned,
        "·" in cleaned,
        "^" in cleaned,
        "(" in cleaned and ")" in cleaned,
    ])

    short_text = len(cleaned.split()) <= 25

    if (
        contains_regulatory_term
        and looks_equation_like
        and short_text
    ):
        return "regulatory_function"
    # --------------------------------------------------
    # 4. Derived parameter definitions
    # Example:
    # CL = ke * Vd
    # k = ln(2)/t_half
    # --------------------------------------------------

    lhs_rhs_match = re.search(
        r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$",
        cleaned
    )

    if lhs_rhs_match:
        lhs = lhs_rhs_match.group(1)
        rhs = lhs_rhs_match.group(2)

        if re.search(r"[\*\u00b7/\+\-]", rhs):
            short_symbol = len(lhs) <= 6

            if short_symbol:
                return "derived_definition"

    # --------------------------------------------------
    # 5. Coupling equations
    # Mechanistic dependency without derivative
    # Example:
    # glucose_uptake = uptake * effect_dxm
    # R = beta * S * I
    # --------------------------------------------------

# --------------------------------------------------
# Reject descriptive scientific sentences
# --------------------------------------------------

    descriptive_terms = [
        "significant",
        "increase",
        "decrease",
        "median",
        "observed",
        "simulation",
        "results",
        "study",
        "represented",
        "illustrates",
        "figure",
        "table",
    ]

# --------------------------------------------------
# Reject narrative scientific text
# Generalizable across scientific papers
# --------------------------------------------------

    narrative_patterns = [
        r"\bwith\s+[A-Za-z]+\s*\(.*\)\s*=",
        r"\bmodel\b",
        r"\bstudy\b",
        r"\bsimulation\b",
        r"\bresults?\b",
        r"\bshow(s|ed)?\b",
        r"\bobserved\b",
        r"\brepresented\b",
        r"\billustrat(es|ed)?\b",
        r"\bfigure\b",
        r"\btable\b",
    ]

    if any(re.search(pattern, lower) for pattern in narrative_patterns):
        return "noise"

    if any(term in lower for term in descriptive_terms):
        return "noise"

    if "=" in cleaned:
        return "coupling_equation"

    # --------------------------------------------------
    # 6. Fallback
    # --------------------------------------------------

    return "noise"

def _equation_classification_metadata(
    model_category: str,
    candidate_text: str,
) -> dict[str, Any]:
    """
    Add confidence and review metadata to deterministic equation classification.

    This prepares the system for selective LLM use later:
    - high confidence -> no LLM needed
    - medium/low confidence -> candidate for small LLM classifier
    """

    cleaned = _clean_line(candidate_text)

    if model_category == "state_equation":
        return {
            "confidence": "high",
            "requires_review": True,
            "llm_review_recommended": False,
            "reason": "contains derivative-like notation indicating a dynamic state equation",
        }

    if model_category == "regulatory_function":
        return {
            "confidence": "high",
            "requires_review": True,
            "llm_review_recommended": False,
            "reason": "contains regulatory/function terms such as effect, Emax, Hill, EC50, or IC50",
        }

    if model_category == "parameter_value_inline":
        return {
            "confidence": "medium",
            "requires_review": True,
            "llm_review_recommended": False,
            "reason": "contains an inline parameter value; useful evidence but not a state equation",
        }

    if model_category == "derived_definition":
        return {
            "confidence": "medium",
            "requires_review": True,
            "llm_review_recommended": True,
            "reason": "short left-hand-side symbol defined by a mathematical expression",
        }

    if model_category == "coupling_equation":
        return {
            "confidence": "medium",
            "requires_review": True,
            "llm_review_recommended": True,
            "reason": "contains an algebraic relationship but no derivative; may be coupling or model modifier",
        }

    if model_category == "algebraic_equation":
        return {
            "confidence": "low",
            "requires_review": True,
            "llm_review_recommended": True,
            "reason": "contains equality but category is ambiguous",
        }

    return {
        "confidence": "low",
        "requires_review": True,
        "llm_review_recommended": False,
        "reason": "low-confidence or noisy candidate; excluded from LLM review by default",
    }

def discover_equations(pdf_path: str) -> list[dict[str, Any]]:
    """
    Discover equation candidates from the PDF text layer.

    No OCR.
    No LLM.
    Conservative by design.
    """

    candidates = []
    seen = set()

    equation_number_pattern = re.compile(r"\(\s*(\d{1,3})\s*\)\s*$")

    pdf = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(pdf, start=1):
            lines = [
                _clean_line(line)
                for line in page.get_text("text").splitlines()
                if _clean_line(line)
            ]

            for line_index, line in enumerate(lines):
                number_match = equation_number_pattern.search(line)

                window_lines = lines[
                    max(0, line_index - 3): min(len(lines), line_index + 4)
                ]
                window_text = "\n".join(window_lines)

                formula_lines = [
                    candidate_line
                    for candidate_line in window_lines
                    if _looks_like_equation_line(candidate_line)
                ]

                if number_match and formula_lines:
                    equation_number = number_match.group(1)
                    candidate_text = window_text
                elif _looks_like_equation_line(line):
                    equation_number = None
                    candidate_text = line
                else:
                    continue

                normalized = re.sub(r"\s+", " ", candidate_text)
                key = (page_index, normalized)

                if key in seen:
                    continue

                seen.add(key)

                model_category = _classify_equation_candidate(candidate_text)

                metadata = _equation_classification_metadata(
                    model_category=model_category,
                    candidate_text=candidate_text,
                )

                candidates.append(
                    {
                        "type": "equation",
                        "equation_number": equation_number,
                        "page": page_index,
                        "candidate_text": candidate_text,
                        "model_category": model_category,
                        "source_method": "pdf_text_layer",
                        "needs_ocr": equation_number is not None,
                        **metadata,
                    }
                )

    finally:
        pdf.close()

    deduplicated_candidates = _deduplicate_equation_candidates(candidates)

    return _run_optional_llm_candidate_classifier(
        deduplicated_candidates
    )


def discover_tables(pdf_path: str) -> list[dict[str, Any]]:
    """
    Discover table candidates from PDF text layer.

    No OCR.
    No LLM.
    """

    table_pattern = re.compile(
        r"\bTable\s+(\d{1,3})\s*[:.]?\s*(.*)",
        flags=re.IGNORECASE,
    )

    candidates = []
    seen = set()

    pdf = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(pdf, start=1):
            lines = [
                _clean_line(line)
                for line in page.get_text("text").splitlines()
                if _clean_line(line)
            ]

            for line_index, line in enumerate(lines):
                match = table_pattern.search(line)

                if not match:
                    continue

                table_number = match.group(1)
                nearby_lines = lines[line_index: line_index + 25]
                candidate_text = "\n".join(nearby_lines)

                key = (table_number, page_index, line)

                if key in seen:
                    continue

                seen.add(key)

                candidates.append(
                    {
                        "type": "table",
                        "table_number": table_number,
                        "page": page_index,
                        "caption_or_anchor": line,
                        "candidate_text": candidate_text,
                        "source_method": "pdf_text_layer",
                        "confidence": "candidate",
                        "requires_review": True,
                        "needs_ocr": False,
                    }
                )

    finally:
        pdf.close()

    return candidates


def discover_figures(pdf_path: str) -> list[dict[str, Any]]:
    """
    Discover figure candidates from PDF text layer.

    No vision.
    No OCR.
    No LLM.
    """

    figure_pattern = re.compile(
        r"\b(?:Figure|Fig\.?)\s+(\d{1,3})\s*[:.]?\s*(.*)",
        flags=re.IGNORECASE,
    )

    candidates = []
    seen = set()

    pdf = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(pdf, start=1):
            lines = [
                _clean_line(line)
                for line in page.get_text("text").splitlines()
                if _clean_line(line)
            ]

            for line_index, line in enumerate(lines):
                match = figure_pattern.search(line)

                if not match:
                    continue

                figure_number = match.group(1)
                nearby_lines = lines[line_index: line_index + 10]
                caption_or_nearby = " ".join(nearby_lines)

                key = (figure_number, page_index, line)

                if key in seen:
                    continue

                seen.add(key)

                candidates.append(
                    {
                        "type": "figure",
                        "figure_number": figure_number,
                        "page": page_index,
                        "caption_or_nearby_text": caption_or_nearby,
                        "source_method": "pdf_text_layer",
                        "confidence": "candidate",
                        "requires_review": True,
                        "needs_vision": False,
                    }
                )

    finally:
        pdf.close()

    return candidates

def _run_optional_llm_candidate_classifier(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Optionally refine ambiguous equation candidates using a tiny LLM classifier.

    Default is disabled for low-cost deterministic discovery.
    Controlled by:
    - ENABLE_LLM_CANDIDATE_CLASSIFIER
    - MAX_LLM_CANDIDATES
    """

    if not ENABLE_LLM_CANDIDATE_CLASSIFIER:
        return candidates

    updated_candidates = []
    llm_calls_used = 0

    for candidate in candidates:
        should_classify = (
            candidate.get("llm_review_recommended") is True
            and llm_calls_used < MAX_LLM_CANDIDATES
        )

        if not should_classify:
            updated_candidates.append(candidate)
            continue

        llm_result = classify_equation_candidate(
            candidate_text=candidate.get("candidate_text", "")
        )

        candidate = {
            **candidate,
            "llm_model_category": llm_result.get("category"),
            "llm_confidence": llm_result.get("confidence"),
            "llm_reason": llm_result.get("reason"),
            "llm_classifier_used": True,
        }

        llm_calls_used += 1
        updated_candidates.append(candidate)

    return updated_candidates

def build_artifact_index(pdf_path: str) -> dict[str, Any]:
    """
    Build a cheap candidate map of equations, tables, and figures.

    This is the shared foundation for:
    - Q/A
    - discovery
    - review/validate
    """

    return {
        "pdf_path": pdf_path,
        "source_method": "pdf_text_layer",
        "ocr_used": False,
        "llm_used": False,
        "equations": discover_equations(pdf_path),
        "tables": discover_tables(pdf_path),
        "figures": discover_figures(pdf_path),
    }


def save_artifact_index(
    pdf_path: str,
    output_path: str = "outputs/artifact_index.json",
) -> dict[str, Any]:
    """
    Build and save artifact index.
    """

    artifact_index = build_artifact_index(pdf_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(artifact_index, file, indent=2, ensure_ascii=False)

    return artifact_index


def _short_text(text: str, max_chars: int = 180) -> str:
    """
    Shorten long candidate text for readable UI display.
    Full raw text remains saved in outputs/artifact_index.json.
    """

    text = _clean_line(str(text))

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def _format_equation_group(
    title: str,
    items: list[dict[str, Any]],
) -> list[str]:
    lines = [f"### {title}"]

    if not items:
        lines.append("- none detected")
        return lines

    for item in items:
        equation_number = item.get("equation_number")
        page = item.get("page")
        needs_ocr = item.get("needs_ocr")
        text = _short_text(item.get("candidate_text", ""))

        label = f"Eq({equation_number})" if equation_number else "candidate"

        review_flag = "requires OCR/review" if needs_ocr else "text-layer candidate"

        lines.append(
            f"- **{label}**, page {page} — {review_flag}: `{text}`"
        )

    return lines


def format_artifact_index_for_review(artifact_index: dict[str, Any]) -> str:
    """
    Human-readable compact candidate map for Streamlit / review.

    Full machine-readable candidates are saved separately in:
    outputs/artifact_index.json
    """

    equations = artifact_index.get("equations", [])
    tables = artifact_index.get("tables", [])
    figures = artifact_index.get("figures", [])

    equation_groups = {
        "State equations": [],
        "Coupling equations": [],
        "Regulatory functions": [],
        "Derived definitions": [],
        "Inline parameter values": [],
        "Other algebraic candidates": [],
        "Noise / low-confidence candidates": [],
    }

    for item in equations:
        category = item.get("llm_model_category") or item.get("model_category")

        if category == "state_equation":
            equation_groups["State equations"].append(item)
        elif category == "coupling_equation":
            equation_groups["Coupling equations"].append(item)
        elif category == "regulatory_function":
            equation_groups["Regulatory functions"].append(item)
        elif category == "derived_definition":
            equation_groups["Derived definitions"].append(item)
        elif category == "parameter_value_inline":
            equation_groups["Inline parameter values"].append(item)
        elif category == "algebraic_equation":
            equation_groups["Other algebraic candidates"].append(item)
        else:
            equation_groups["Noise / low-confidence candidates"].append(item)

    lines = [
        "# Lit2Model-AI Discovery Candidate Map",
        "",
        "This is a cheap deterministic discovery step.",
        "",
        "**Status:** candidate map only / needs review",
        "**Source method:** PDF text layer",
        "**OCR used:** false",
        "**LLM used:** false",
        "",
        "The full raw candidate map is saved in `outputs/artifact_index.json`.",
        "",
        "---",
        "",
        "## Equation Candidates",
        "",
        f"- Total equation-like candidates: **{len(equations)}**",
        f"- State equations: **{len(equation_groups['State equations'])}**",
        f"- Coupling equations: **{len(equation_groups['Coupling equations'])}**",
        f"- Regulatory functions: **{len(equation_groups['Regulatory functions'])}**",
        f"- Derived definitions: **{len(equation_groups['Derived definitions'])}**",
        f"- Inline parameter values: **{len(equation_groups['Inline parameter values'])}**",
        "",
    ]

    for group_title in [
        "State equations",
        "Coupling equations",
        "Regulatory functions",
        "Derived definitions",
        "Inline parameter values",
    ]:
        lines.extend(_format_equation_group(group_title, equation_groups[group_title]))
        lines.append("")

    low_confidence_count = (
        len(equation_groups["Other algebraic candidates"])
        + len(equation_groups["Noise / low-confidence candidates"])
    )

    lines.extend(
        [
            "### Low-confidence / hidden candidates",
            f"- {low_confidence_count} additional candidates were detected but hidden from this summary.",
            "- Inspect `outputs/artifact_index.json` if needed.",
            "",
            "---",
            "",
            "## Table Candidates",
            "",
        ]
    )

    if not tables:
        lines.append("- none detected")
    else:
        for item in tables:
            lines.append(
                "- "
                f"**Table {item.get('table_number')}**, page {item.get('page')}: "
                f"{_short_text(item.get('caption_or_anchor', ''))}"
            )

    lines.extend(["", "---", "", "## Figure Candidates", ""])

    if not figures:
        lines.append("- none detected")
    else:
        seen_figures = set()

        for item in figures:
            figure_number = item.get("figure_number")
            page = item.get("page")
            key = (figure_number, page)

            if key in seen_figures:
                continue

            seen_figures.add(key)

            lines.append(
                "- "
                f"**Figure {figure_number}**, page {page}: "
                f"{_short_text(item.get('caption_or_nearby_text', ''))}"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Next Step",
            "",
            "- Use **Review & Validate Model** to inspect selected candidates.",
            "- Run OCR/vision only for weak or visually omitted equations, tables, or figures.",
            "- Save accepted items into the reviewed model draft.",
            "- Generate the Python model only from reviewed evidence.",
        ]
    )

    return "\n".join(lines)

def _deduplicate_equation_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Prefer compact, cleaner candidates over large overlapping noisy windows.
    """

    def score(item: dict[str, Any]) -> tuple[int, int]:
        text = _clean_line(item.get("candidate_text", ""))
        equals_count = text.count("=")

        penalty = 0

        if len(text) > 300:
            penalty += 5
        if "where" in text.lower():
            penalty += 3
        if equals_count > 2:
            penalty += 2
        if item.get("model_category") == "noise":
            penalty += 10

        return (penalty, len(text))

    grouped = {}

    for item in candidates:
        page = item.get("page")
        category = item.get("model_category")
        text = _clean_line(item.get("candidate_text", ""))

        lhs_match = re.search(r"([A-Za-z][A-Za-z0-9_\-−]*)\s*=", text)
        lhs = lhs_match.group(1) if lhs_match else text[:40]

        key = (page, category, lhs)

        if key not in grouped or score(item) < score(grouped[key]):
            grouped[key] = item

    return list(grouped.values())
