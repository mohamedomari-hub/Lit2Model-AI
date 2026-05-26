import json
import os
import re

import fitz

from src.document.crops import render_pdf_page_for_equation
from src.document.ocr import (
    extract_equation_with_gpt,
    extract_equation_with_pix2tex,
)
from src.retrieval import format_doc_metadata, make_chunk_key

VECTOR_STORE = None
ACTIVE_PDF_PATH = None
MISSING_EQUATIONS_PATH = "outputs/missing_equations.json"


def configure_equation_recovery_runtime(vector_store=None, pdf_path: str | None = None, missing_equations_path: str | None = None):
    global VECTOR_STORE, ACTIVE_PDF_PATH, MISSING_EQUATIONS_PATH
    VECTOR_STORE = vector_store
    ACTIVE_PDF_PATH = pdf_path
    if missing_equations_path is not None:
        MISSING_EQUATIONS_PATH = missing_equations_path


def _equation_anchor_regex(equation_number: str) -> str:
    return (
        rf"(?<!\d)"
        rf"(?:"
        rf"\({re.escape(equation_number)}\)"
        rf"|equation\s*\(?{re.escape(equation_number)}\)?"
        rf"|eq\.?\s*\(?{re.escape(equation_number)}\)?"
        rf")"
        rf"(?!\d)"
    )

def _is_weak_equation_reference(text: str, equation_number: str) -> bool:
    text_lower = text.lower()
    weak_patterns = [
        rf"described\s+by\s+eq\.?\s*\(?{equation_number}\)?",
        rf"expressed\s+by\s+eq\.?\s*\(?{equation_number}\)?",
        rf"shown\s+in\s+eq\.?\s*\(?{equation_number}\)?",
        rf"given\s+by\s+eq\.?\s*\(?{equation_number}\)?",
        rf"according\s+to\s+eq\.?\s*\(?{equation_number}\)?",
        rf"using\s+eq\.?\s*\(?{equation_number}\)?",
        rf"from\s+eq\.?\s*\(?{equation_number}\)?",
        rf"see\s+eq\.?\s*\(?{equation_number}\)?",
        rf"as\s+in\s+eq\.?\s*\(?{equation_number}\)?",
        rf"eq\.?\s*\(?\d+\)?\s+(?:and|or|to|-)\s+eq\.?\s*\(?{equation_number}\)?",
    ]
    return any(re.search(pattern, text_lower) for pattern in weak_patterns)

def _looks_like_formula_line(line: str) -> bool:
    line = line.strip()
    line_lower = line.lower()

    if not line:
        return False

    if _is_weak_equation_reference(line_lower, r"\d+"):
        return False

    strong_markers = [
        r"\\frac",
        r"\frac",
        "d/dt",
        "\\dot",
        "\\partial",
        "\\sum",
        "\\prod",
        "\\int",
        "\\left",
        "\\right",
    ]

    if any(marker in line_lower for marker in strong_markers):
        return True

    if "=" in line:
        math_symbol_count = len(
            re.findall(r"[A-Za-zΑ-ω][A-Za-z0-9_{}\\-]*|\^|/|\+|-|\*|\(|\)", line)
        )
        return math_symbol_count >= 4

    if re.search(r"\b[dD][A-Za-z0-9_{}]+\s*/\s*d[tT]\b", line):
        return True

    return False

def is_exact_equation_match(
    content: str,
    equation_number: str
) -> bool:
    """
    Detect whether the chunk likely contains the actual numbered equation,
    not only a prose reference like 'described by eq.(2)'.
    """

    if not content:
        return False

    anchor_pattern = _equation_anchor_regex(equation_number)
    lines = content.splitlines()

    anchor_line_indices = [
        index for index, line in enumerate(lines)
        if re.search(anchor_pattern, line, flags=re.IGNORECASE)
    ]

    if not anchor_line_indices:
        return False

    for index in anchor_line_indices:
        window_start = max(0, index - 2)
        window_end = min(len(lines), index + 3)
        window_lines = lines[window_start:window_end]
        window_text = "\n".join(window_lines)

        if _is_weak_equation_reference(window_text, equation_number):
            if not any(_looks_like_formula_line(line) for line in window_lines):
                continue

        if any(_looks_like_formula_line(line) for line in window_lines):
            return True

    return False

def extract_best_equation_page_from_context(
    context: str,
    equation_number: str
):
    """
    Pick the best candidate page for equation OCR.

    Prefer pages where the requested equation number appears near formula-like
    text or parser omission markers. Do not reward prose references alone too much.
    """

    chunks = context.split("---")
    page_scores = {}

    anchor_pattern = _equation_anchor_regex(equation_number)

    for chunk in chunks:
        page_match = re.search(r"Page:\s*(\d+)", chunk)

        if not page_match:
            continue

        page = int(page_match.group(1))
        chunk_lower = chunk.lower()
        score = page_scores.get(page, 0)

        has_anchor = bool(re.search(anchor_pattern, chunk, flags=re.IGNORECASE))
        has_formula = has_equation_formula(chunk)
        weak_reference = _is_weak_equation_reference(chunk, equation_number)

        if has_anchor and has_formula:
            score += 40
        elif has_anchor and not weak_reference:
            score += 15
        elif has_anchor and weak_reference:
            score += 4

        if "picture intentionally omitted" in chunk_lower:
            score += 20

        if "start of picture text" in chunk_lower:
            score += 10

        if any(term in chunk_lower for term in ["mathematical model", "differential equation", "ordinary differential"]):
            score += 5

        if any(term in chunk_lower for term in ["where", "parameter", "variable", "defined as"]):
            score += 2

        if has_formula:
            score += 8

        page_scores[page] = score

    if not page_scores:
        return None

    best_page, best_score = max(page_scores.items(), key=lambda item: item[1])

    print(
        f"Best OCR page for Equation {equation_number}: "
        f"{best_page}, score={best_score}"
    )

    return best_page

def _format_ocr_candidate(label: str, candidate: str) -> str:
    """
    Format OCR output for the UI without exposing noisy exception text as if it
    were an equation candidate.
    """

    candidate = (candidate or "").strip()

    if not candidate:
        return f"**{label}:** not available."

    if "failed:" in candidate.lower() or "error" in candidate.lower():
        return f"**{label}:** not available for this crop."

    return f"""**{label} candidate transcription:**\n\n```text\n{candidate}\n```"""

def _format_numbered_equation_ocr_response(
    equation_number: str,
    image_path: str,
    pix2tex_context: str,
    gpt_context: str,
) -> str:
    """
    User-facing response for numbered-equation OCR fallback.

    OCR candidates remain explicitly unvalidated and should not be used for
    simulation/model generation before human review.
    """

    return f"""
No validated Equation {equation_number} formula was found in the retrieved text.

I rendered a candidate crop for OCR:

```text
{image_path}
```

{_format_ocr_candidate("pix2tex / LaTeX-OCR", pix2tex_context)}

{_format_ocr_candidate("GPT Vision OCR", gpt_context)}

**Review note:** These are candidate transcriptions only. They are not validated equations and should not be used for simulation or model generation until the crop is checked by a human.
"""

def extract_numbered_equation_text_from_pdf(
    pdf_path: str,
    equation_number: str
) -> dict | None:
    """
    Try to extract the actual numbered equation from selectable PDF text.

    This is used before OCR. It is intentionally generic: find a compact
    equation-number anchor inside PDF text blocks, then accept it only when it
    is physically close to formula-like text. This avoids mistaking page
    footers such as "3" or "4" for equation numbers.
    """

    pdf = fitz.open(pdf_path)
    anchor_line_pattern = re.compile(
        rf"^(?:"
        rf"\(\s*{re.escape(equation_number)}\s*\)"
        rf"|eq\.?\s*\(?\s*{re.escape(equation_number)}\s*\)?"
        rf"|equation\s*\(?\s*{re.escape(equation_number)}\s*\)?"
        rf")[.,]?$",
        flags=re.IGNORECASE,
    )

    try:
        for page_index, page in enumerate(pdf):
            raw_blocks = page.get_text("blocks")
            blocks = [
                {
                    "rect": fitz.Rect(block[:4]),
                    "text": str(block[4]).strip(),
                }
                for block in raw_blocks
                if str(block[4]).strip()
            ]
            blocks.sort(key=lambda item: (item["rect"].y0, item["rect"].x0))

            for block_index, block in enumerate(blocks):
                block_lines = [
                    line.strip()
                    for line in block["text"].splitlines()
                    if line.strip()
                ]

                if not any(anchor_line_pattern.match(line) for line in block_lines):
                    continue

                candidate_blocks = [block]

                # Equations are often split into neighboring PDF blocks: one
                # block for the formula and one for the closing parenthesis plus
                # equation number. Include close blocks immediately above.
                for previous_index in range(block_index - 1, -1, -1):
                    previous = blocks[previous_index]
                    vertical_gap = block["rect"].y0 - previous["rect"].y1
                    overlaps_vertically = previous["rect"].y1 >= block["rect"].y0 - 5

                    if vertical_gap > 45 and not overlaps_vertically:
                        break

                    candidate_blocks.insert(0, previous)

                    if "=" in previous["text"]:
                        break

                # Include close explanatory/continuation text immediately below,
                # but only for explanation, not as part of the candidate formula.
                supporting_blocks = []
                for next_index in range(block_index + 1, min(len(blocks), block_index + 4)):
                    next_block = blocks[next_index]
                    vertical_gap = next_block["rect"].y0 - block["rect"].y1
                    if vertical_gap > 70:
                        break
                    supporting_blocks.append(next_block)

                candidate_text = "\n".join(item["text"] for item in candidate_blocks)

                if _is_weak_equation_reference(candidate_text, equation_number):
                    continue

                if "=" not in candidate_text:
                    continue

                # Reject likely page footer/section-number matches. These often
                # have large vertical gaps from the nearest formula block.
                nearest_formula_gap = min(
                    abs(block["rect"].y0 - item["rect"].y1)
                    for item in candidate_blocks
                    if "=" in item["text"]
                )
                if nearest_formula_gap > 80:
                    continue

                candidate_lines = [
                    line.strip()
                    for line in candidate_text.splitlines()
                    if line.strip()
                ]
                target_indices = [
                    line_index
                    for line_index, candidate_line in enumerate(candidate_lines)
                    if anchor_line_pattern.match(candidate_line)
                ]

                if not target_indices:
                    continue

                target_index = target_indices[-1]
                equals_indices = [
                    line_index
                    for line_index, candidate_line in enumerate(candidate_lines[:target_index])
                    if "=" in candidate_line
                ]

                if not equals_indices:
                    continue

                start_index = equals_indices[-1]

                # Initial-condition lines often contain "=", but belong with the
                # preceding differential equation.
                if (
                    candidate_lines[start_index].lower().startswith("with ")
                    and len(equals_indices) >= 2
                ):
                    start_index = equals_indices[-2]

                equation_text = "\n".join(candidate_lines[start_index:target_index + 1])
                supporting_prefix = "\n".join(candidate_lines[:start_index])
                supporting_suffix = "\n".join(item["text"] for item in supporting_blocks)
                supporting_text = "\n".join(
                    part for part in [supporting_prefix, supporting_suffix] if part.strip()
                )

                equation_text = (
                    equation_text
                    .replace("\x12", "(")
                    .replace("\x13", ")")
                    .replace("eﬀect", "effect")
                )
                supporting_text = (
                    supporting_text
                    .replace("\x12", "(")
                    .replace("\x13", ")")
                    .replace("eﬀect", "effect")
                )

                return {
                    "page": page_index + 1,
                    "equation_text": equation_text,
                    "supporting_text": supporting_text,
                }

    finally:
        pdf.close()

    return None

def _format_numbered_equation_text_response(
    equation_number: str,
    extracted: dict,
) -> str:
    return f"""
Equation {equation_number} was found in the PDF text layer.

PDF page: {extracted["page"]}

PDF text-layer candidate:

```text
{extracted["equation_text"]}
```

Nearby explanatory text:

```text
{extracted.get("supporting_text", "")}
```

Review note: PDF text-layer extraction can break fractions, superscripts, and line layout. Treat this as a source-backed candidate and check the original PDF before simulation or model generation.
"""

def retrieve_numbered_equation_context(
    vector_store=None,
    pdf_path: str | None = None,
    query: str = "",
    missing_equations_path: str = "outputs/missing_equations.json",
) -> str:
    """
    Retrieve a specific numbered equation from the PDF.

    Examples:
    - Explain equation 7
    - What does Eq. (4) mean?
    - Explain eq 10
    """
        
    configure_equation_recovery_runtime(vector_store, pdf_path, missing_equations_path)


    if VECTOR_STORE is None:
        return "Vector store is not initialized."

    query_lower = query.lower()

    equation_match = re.search(
        r"\b(?:equation|equa|eqn|eq\.?)\s*\(?\s*(\d+)\s*\)?",
        query_lower
    )

    if not equation_match:
        return (
            "No equation number detected. "
            "Please specify something like: "
            "'Explain equation 7'."
        )

    equation_number = equation_match.group(1)

    search_queries = [
        query,

        # Exact equation references
        f"Equation {equation_number}",
        f"equation {equation_number}",
        f"Eq {equation_number}",
        f"Eq. {equation_number}",
        f"eq {equation_number}",
        f"eq. {equation_number}",
        f"({equation_number})",
        f"equation ({equation_number})",

        # Context around equations
        f"equation {equation_number} model",
        f"equation {equation_number} differential equation",
        f"equation {equation_number} parameter",
        f"equation {equation_number} explanation",
        f"mathematical model equation {equation_number}",
    ]

    exact_results = []
    fallback_results = []
    seen = set()

    for search_query in search_queries:

        docs = VECTOR_STORE.similarity_search(
            search_query,
            k=6
        )

        print(
            f"\nRetrieved for query: {search_query}"
        )

        for doc in docs:

            chunk_key = make_chunk_key(doc)

            if chunk_key in seen:
                continue

            seen.add(chunk_key)

            content = doc.page_content

            is_exact_match = is_exact_equation_match(
                content=content,
                equation_number=equation_number
            )

            if "picture intentionally omitted" in content.lower():
                content = (
                    "[EQUATION OCR WARNING: "
                    "Displayed equation may have been omitted "
                    "by the parser and could require OCR "
                    "or human review.]\n"
                    + content
                )

            header = format_doc_metadata(
                doc=doc,
                search_query=search_query,
                label="Equation Search Query"
            )

            entry = (
                f"{header} | Exact Match: {is_exact_match}\n"
                f"{content}"
            )

            if is_exact_match:
                exact_results.append({
                    "entry": entry,
                    "content": content
                })
            else:
                fallback_results.append(entry)
            
            #print(doc.page_content[:300])

    exact_formula_results = [
        item["entry"]
        for item in exact_results
        if has_equation_formula(
            item["content"]
        )
    ]

    if exact_formula_results:
        print(
            f"[Equation Recovery] "
            f"Found text-layer/vector candidates for Equation {equation_number}, "
            f"but using OCR-first recovery for equations."
        )


    if ACTIVE_PDF_PATH is not None:

        retrieved_context_for_page_detection = "\n\n---\n\n".join(
            [item["entry"] for item in exact_results]
            + fallback_results
        )

        fallback_text = retrieved_context_for_page_detection

        print(
            f"[Equation Recovery] "
            f"Skipping PDF text-layer equation extraction for Equation {equation_number}. "
            f"Using OCR-first recovery."
        )

        page_number = extract_best_equation_page_from_context(
        fallback_text, equation_number)

        if page_number is None:
            return f"""
    No exact Equation {equation_number} match was found in text retrieval.

    Could not identify a candidate page for OCR.

    FALLBACK RETRIEVED CONTEXT:
    {fallback_text}
    """

        image_path = render_pdf_page_for_equation(
            pdf_path=ACTIVE_PDF_PATH,
            page_number=page_number,
            equation_number=equation_number
        )

        pix2tex_context = (
            "Skipped: GPT-4o-mini Vision selected "
            "as primary equation OCR."
        )

        print(
            f"[Equation Recovery] "
            f"Using GPT Vision OCR for Equation "
            f"{equation_number}"
        )
        
        try:
            gpt_context = extract_equation_with_gpt(
                image_path=image_path,
                equation_number=equation_number
            )
        except Exception as error:
            gpt_context = (
                f"GPT equation OCR failed: "
                f"{type(error).__name__}: {error}"
            )

        return _format_numbered_equation_ocr_response(
            equation_number=equation_number,
            image_path=image_path,
            pix2tex_context=pix2tex_context,
            gpt_context=gpt_context,
        )
    if fallback_results:
        return (
            f"No exact Equation {equation_number} match was found.\n\n"
            f"Fallback retrieved context:\n\n"
            + "\n\n---\n\n".join(fallback_results)
        )

    return f"No context found for Equation {equation_number}."

def extract_equation_candidates_from_text_layer(pdf_path: str | None = None) -> str:
    if pdf_path is not None:
        configure_equation_recovery_runtime(pdf_path=pdf_path)
    """
    Local, non-OCR scan for equation-like text blocks in the PDF text layer.

    This catches selectable equations before relying on semantic retrieval or
    vision OCR. Candidates remain unvalidated because PDF text extraction can
    damage fractions, superscripts, and layout.
    """

    if ACTIVE_PDF_PATH is None or not os.path.exists(ACTIVE_PDF_PATH):
        return "No active PDF available for text-layer equation candidate scan."

    pdf = fitz.open(ACTIVE_PDF_PATH)
    candidates = []
    seen = set()

    equation_number_pattern = re.compile(r"\(\s*(\d+)\s*\)\s*$")

    try:
        for page_index, page in enumerate(pdf, start=1):
            blocks = [
                {
                    "rect": fitz.Rect(block[:4]),
                    "text": str(block[4]).strip(),
                }
                for block in page.get_text("blocks")
                if str(block[4]).strip()
            ]
            blocks.sort(key=lambda item: (item["rect"].y0, item["rect"].x0))

            for block_index, block in enumerate(blocks):
                lines = [
                    line.strip()
                    for line in block["text"].splitlines()
                    if line.strip()
                ]

                if not lines:
                    continue

                formula_lines = [
                    line for line in lines
                    if _looks_like_formula_line(line)
                ]
                number_matches = [
                    equation_number_pattern.search(line)
                    for line in lines
                ]
                equation_numbers = [
                    match.group(1)
                    for match in number_matches
                    if match is not None
                ]

                if not formula_lines and not equation_numbers:
                    continue

                context_lines = lines

                if equation_numbers and not formula_lines and block_index > 0:
                    previous_lines = [
                        line.strip()
                        for line in blocks[block_index - 1]["text"].splitlines()
                        if line.strip()
                    ]
                    if any(_looks_like_formula_line(line) for line in previous_lines):
                        context_lines = previous_lines + lines

                candidate_text = "\n".join(context_lines)

                if not has_equation_formula(candidate_text):
                    continue

                normalized_key = re.sub(r"\s+", " ", candidate_text)

                if normalized_key in seen:
                    continue

                seen.add(normalized_key)
                equation_number = equation_numbers[-1] if equation_numbers else ""

                candidates.append(
                    "Equation text-layer candidate:\n"
                    f"- Equation number: {equation_number or 'not detected'}\n"
                    f"- Source page: {page_index}\n"
                    "- Source: PDF text layer\n"
                    "- Method: local text-layer scan\n"
                    "- Confidence: candidate / requires human review\n"
                    "- Requires review: true\n"
                    "- Candidate text:\n"
                    "```text\n"
                    f"{candidate_text}\n"
                    "```"
                )
    finally:
        pdf.close()

    if not candidates:
        return "No equation-like candidates found in the PDF text layer."

    return "\n\n---\n\n".join(candidates)

def find_referenced_equation_numbers_in_pdf() -> list[str]:
    if ACTIVE_PDF_PATH is None or not os.path.exists(ACTIVE_PDF_PATH):
        return []

    pdf = fitz.open(ACTIVE_PDF_PATH)
    found = set()

    try:
        for page in pdf:
            page_text = page.get_text("text")
            lines = page_text.splitlines()
            page_rect = page.rect

            for number in range(1, 51):
                for rect in page.search_for(f"({number})"):
                    is_right_margin_label = rect.x0 > page_rect.width * 0.60
                    is_body_area = (
                        page_rect.height * 0.08
                        < rect.y0
                        < page_rect.height * 0.92
                    )

                    if is_right_margin_label and is_body_area:
                        found.add(str(number))

            for match in re.finditer(
                r"\b(?:equation|eq\.?|eqn)\s*\(?\s*(\d{1,2})\s*\)?",
                page_text,
                flags=re.IGNORECASE,
            ):
                found.add(match.group(1))

            for index, line in enumerate(lines):
                label_match = re.search(r"^\s*\(?\s*(\d{1,2})\s*\)?\s*$", line)

                if not label_match:
                    label_match = re.search(r"\(\s*(\d{1,2})\s*\)\s*$", line)

                if not label_match:
                    continue

                number = label_match.group(1)
                window = "\n".join(
                    lines[max(0, index - 3):min(len(lines), index + 4)]
                )

                if any(_looks_like_formula_line(candidate) for candidate in window.splitlines()):
                    found.add(number)
    finally:
        pdf.close()

    return sorted(found, key=lambda value: int(value))

def find_equation_numbers_in_text(text: str) -> set[str]:
    return set(
        group
        for match in re.finditer(
            r"Equation\s+(?:number:\s*)?(\d+)|Equation\s+(\d+)|###\s*Equation\s+(\d+)",
            text,
            flags=re.IGNORECASE,
        )
        for group in match.groups()
        if group
    )

def recover_missing_numbered_equations(
    extracted_equations: str,
    equation_context: str,
    pdf_path: str | None = None,
    missing_equations_path: str = "outputs/missing_equations.json",
) -> str:
    configure_equation_recovery_runtime(
        pdf_path=pdf_path,
        missing_equations_path=missing_equations_path,
    )
    referenced_numbers = find_referenced_equation_numbers_in_pdf()

    if not referenced_numbers:
        return "No numbered equation references found for missing-equation recovery."

    found_numbers = find_equation_numbers_in_text(extracted_equations)
    missing_numbers = [
        number for number in referenced_numbers
        if number not in found_numbers
    ]

    if not missing_numbers:
        os.makedirs("outputs", exist_ok=True)
        with open(MISSING_EQUATIONS_PATH, "w", encoding="utf-8") as file:
            json.dump([], file, indent=2)
        return "No missing numbered equations detected."

    records = []
    manifest_records = []

    for equation_number in missing_numbers:
        page_number = extract_best_equation_page_from_context(
            equation_context,
            equation_number,
        )

        if page_number is None:
            page_number = 1

        try:
            image_path = render_pdf_page_for_equation(
                pdf_path=ACTIVE_PDF_PATH,
                page_number=page_number,
                equation_number=equation_number,
            )
        except Exception as error:
            manifest_records.append(
                {
                    "equation_number": equation_number,
                    "status": "crop render failed",
                    "source_page": None,
                    "candidate_crop": None,
                    "requires_review": True,
                    "review_notes": f"{type(error).__name__}: {error}",
                }
            )
            records.append(
                "Missing equation recovery candidate:\n"
                f"- Equation number: {equation_number}\n"
                "- Status: missing / crop render failed\n"
                "- Source page: not determined\n"
                "- Method/source: local missing-equation recovery\n"
                "- Requires review: true\n"
                f"- Review notes: {type(error).__name__}: {error}"
            )
            continue

        rendered_page_match = re.search(r"_page_(\d+)\.png$", image_path)
        rendered_page_number = (
            int(rendered_page_match.group(1))
            if rendered_page_match
            else page_number
        )

        manifest_records.append(
            {
                "equation_number": equation_number,
                "status": "missing / crop available",
                "source_page": rendered_page_number,
                "candidate_crop": image_path,
                "requires_review": True,
                "review_notes": (
                    "Run local pix2tex or gpt-4o OCR from Review & Validate "
                    "Model, then accept a candidate into the draft."
                ),
            }
        )

        records.append(
            "Missing equation recovery candidate:\n"
            f"- Equation number: {equation_number}\n"
            "- Status: missing / crop available\n"
            f"- Source page: {rendered_page_number}\n"
            "- Method/source: targeted local crop recovery\n"
            f"- Candidate crop: {image_path}\n"
            "- Confidence: candidate / requires human review\n"
            "- Requires review: true\n"
            "- Review notes: Run OCR from Review & Validate Model and accept a candidate into the draft."
        )

    os.makedirs("outputs", exist_ok=True)
    with open(MISSING_EQUATIONS_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest_records, file, indent=2)

    return "\n\n---\n\n".join(records)

def has_equation_formula(content: str) -> bool:
    """
    Check whether raw retrieved text likely contains an actual mathematical
    formula, not only a prose reference such as 'described by eq.(2)'.
    """

    if not content:
        return False

    lines = [line.strip() for line in content.splitlines() if line.strip()]

    for line in lines:
        if _looks_like_formula_line(line):
            return True

    content_lower = content.lower()

    if "picture intentionally omitted" in content_lower:
        return False

    if "start of picture text" in content_lower and "=" in content:
        return True

    return False
