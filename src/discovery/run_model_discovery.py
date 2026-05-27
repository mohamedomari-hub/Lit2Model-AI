from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.ingestion.pdf_parser import extract_equation_candidates_with_pymupdf
from src.ingestion.ocr import extract_visible_equations_with_gpt
from src.discovery.discovery_prompts import SYSTEM_PROMPT
from src.retrieval.equation_search import (
    search_equation_candidates,
)
from src.retrieval.paper_search import search_discovery_context
from src.retrieval.table_search import search_table_evidence
from src.discovery.review_formatter import format_compact_review


load_dotenv()

if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


DISCOVERY_QUERIES = [
    # Dynamic equations / state evolution
    (
        "numbered equation model equation "
        "ODE differential equation dynamic equation "
        "state variable compartment equation "
        "state transition initial condition"
    ),

    # Algebraic / coupling / process equations
    (
        "algebraic equation auxiliary equation "
        "coupling equation process equation "
        "modifier equation regulatory equation "
        "interaction equation transfer function"
    ),

    # Regulatory / nonlinear / mechanistic functions
    (
        "regulatory function nonlinear function "
        "feedback stimulation inhibition activation "
        "suppression saturation threshold modifier "
        "switching function coupling term"
    ),

    # Parameters and parameterization
    (
        "parameter value parameter table "
        "estimated fitted calibrated fixed assumed "
        "derived parameter parameter unit "
        "rate constant coefficient"
    ),

    # Inputs / interventions / external forcing
    (
        "external input intervention forcing function "
        "dose treatment administration perturbation "
        "boundary condition control signal"
    ),

    # Outputs / observations / calibration
    (
        "observed data measured output "
        "validation calibration experiment "
        "time course trajectory concentration response"
    ),

    # Mechanistic relationships / causal structure
    (
        "mechanism interaction transfer "
        "causal relationship feedback "
        "compartment flow activation inhibition "
        "dependency network"
    ),

    # Symbol definitions and explanatory text
    (
        "where parameter representing defined as "
        "symbol definition explanation "
        "half maximal threshold constant "
        "meaning variable interpretation"
    ),
]

# TODO: Parameter evidence is currently mixed into DISCOVERY_QUERIES.
# Move it to search_parameter_evidence() only when output formatting is preserved.
# TODO: Mechanism evidence is currently mixed into DISCOVERY_QUERIES.
# Move it to search_mechanism_evidence() only when output formatting is preserved.

TABLE_QUERIES = [
    "Table parameter values units symbol explanation",
    "Table initial values component state unit",
    "Table threshold parameters value unit explanation",
    "Table rate effect parameters value unit explanation",
    "parameter table Symbol Value Unit Explanation",
]


def _deduplicate_docs(docs: list[Any]) -> list[Any]:
    unique = OrderedDict()

    for doc in docs:
        text = (doc.page_content or "").strip()

        if len(text) < 40:
            continue

        key = re.sub(r"\s+", " ", text[:700]).lower()

        if key not in unique:
            unique[key] = doc

    return list(unique.values())

# TODO: Keep these legacy equation helpers until shared equation retrieval is fully verified.
def looks_like_corrupted_equation_candidate(text: str) -> bool:
    """
    Detect equation candidates where PDF text extraction likely damaged math layout.
    """

    if not text:
        return False

    suspicious_markers = [
        "formula-not-decoded",
        "\x13",
        "\x14",
        "",
        "C10\ne",
        "C7\ne",
        "C10",
        "C7",
    ]

    if any(marker in text for marker in suspicious_markers):
        return True

    # Broken math often has many short isolated symbol lines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_symbol_lines = [
        line for line in lines
        if len(line) <= 4 and re.search(r"[A-Za-z0-9]", line)
    ]

    if len(short_symbol_lines) >= 3 and "=" in text:
        return True

    return False


def extract_image_path_from_candidate_text(text: str) -> str | None:
    """
    Extract image_path from an equation candidate record.
    """

    match = re.search(r"image_path:\s*(\S+)", text)

    if not match:
        return None

    return match.group(1).strip()


def repair_equation_candidate_with_ocr(candidate_text: str) -> str:
    """
    If equation candidate is corrupted and has an image crop path,
    run GPT vision OCR and return repaired candidate text.
    """

    if not looks_like_corrupted_equation_candidate(candidate_text):
        return candidate_text

    image_path = extract_image_path_from_candidate_text(candidate_text)

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
    
def retrieve_discovery_context(
    vector_store,
    k_per_query: int = 8,
    max_total_chars: int = 40000,
    max_equation_candidates: int = 500,
) -> str:
    # Discovery currently keeps custom document-level retrieval because it needs raw docs,
    # metadata, deduplication, and equation candidate scanning.
    # Later this should be moved into src/retrieval/paper_search.py.
    # Kept custom discovery logic for now.
    unique_docs, retrieved_context = search_discovery_context(
        vector_store=vector_store,
        discovery_queries=DISCOVERY_QUERIES,
        k_per_query=k_per_query,
        max_total_chars=max_total_chars,
    )

    equation_candidates = search_equation_candidates(
        vector_store=vector_store,
        max_equation_candidates=max_equation_candidates,
    )

    equation_text = "\n".join(
        equation_candidates[:max_equation_candidates]
    )

    return f"""

equation_text = (
    "IMPORTANT:\n"
    "Equations below are high-priority evidence.\n"
    "If a model term is used in one equation "
    "(example: effectdxm_gluca), "
    "and another equation defines that term, "
    "both equations must be extracted together.\n\n"
    + equation_text
)
GLOBAL EQUATION CANDIDATES
Use this section to avoid missing equations.
If equations here define related model terms, group them into process_modules.

{equation_text}

GENERAL RETRIEVED CONTEXT

{retrieved_context}
"""

def retrieve_table_evidence(
    vector_store,
    k_per_query: int = 6,
    max_total_chars: int = 15000,
) -> str:
    """
    Retrieve table-heavy chunks separately so parameters are not missed.
    This is cheap text-layer retrieval. OCR fallback can be added later.
    """

    # Table evidence currently keeps custom document-level retrieval because it filters
    # raw docs by table markers and preserves the discovery output format.
    # Later this should be moved into src/retrieval/table_search.py.
    # Kept custom discovery logic for now.
    return search_table_evidence(
        vector_store=vector_store,
        table_queries=TABLE_QUERIES,
        k_per_query=k_per_query,
        max_total_chars=max_total_chars,
    )

def _extract_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output.")

    return text[start:end + 1]


def _safe_json_parse(text: str) -> dict[str, Any]:
    candidate = _extract_json_text(text)

    try:
        return json.loads(candidate)

    except json.JSONDecodeError:
        repair_llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature=0,
        )

        repaired = repair_llm.invoke(
            "Repair this malformed JSON. "
            "Return only valid JSON. "
            "Do not add scientific information.\n\n"
            + candidate
        ).content

        return json.loads(_extract_json_text(repaired))


def _norm_symbol(value: Any) -> str:
    text = str(value or "")
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("ﬀ", "ff")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Single generalized cleanup.

    No paper-specific symbols.
    """

    input_markers = (
        "dose",
        "dosage",
        "input",
        "external_input",
        "externalforcing",
        "external_forcing",
        "forcing",
        "forcingfunction",
        "forcing_function",
        "intervention",
        "treatment",
        "administration",
        "administered",
        "infusion",
        "injection",
        "bolus",
        "exposure",
        "perturbation",
        "stimulus",
    )

    for ode in result.get("odes", []):
        equation_norm = _norm_symbol(ode.get("equation"))
        parameters = ode.get("parameters", []) or []
        inputs = ode.get("inputs", []) or []

        cleaned_parameters = []

        for param in parameters:
            symbol_norm = _norm_symbol(param.get("symbol"))
            meaning_norm = _norm_symbol(param.get("meaning"))

            is_external_input = (
                symbol_norm in equation_norm
                and any(marker in symbol_norm or marker in meaning_norm for marker in input_markers)
            )

            if is_external_input:
                inputs.append(
                    {
                        "symbol": param.get("symbol"),
                        "value": param.get("value", "not reported"),
                        "unit": param.get("unit", "not reported"),
                        "meaning": param.get("meaning", "external input/intervention"),
                    }
                )
            else:
                cleaned_parameters.append(param)

        ode["parameters"] = cleaned_parameters
        ode["inputs"] = inputs

        # Keep only observations that mention the state symbol/name.
        state_norm = _norm_symbol(ode.get("state"))
        meaning_norm = _norm_symbol(ode.get("meaning"))

        cleaned_observed = []

        for obs in ode.get("observed_data", []) or []:
            obs_norm = _norm_symbol(obs)

            if state_norm and state_norm in obs_norm:
                cleaned_observed.append(obs)
            elif meaning_norm and meaning_norm in obs_norm:
                cleaned_observed.append(obs)

        ode["observed_data"] = cleaned_observed

        for param in ode.get("parameters", []):
            status = _norm_symbol(param.get("status"))
            value = _norm_symbol(param.get("value"))
            formula = _norm_symbol(param.get("formula"))

            if status == "missing" and formula:
                param["status"] = "derived"

            if status == "missing" and value not in {"", "missing", "notreported"}:
                param["status"] = "reported"

    modules = result.get("process_modules", []) or []
    merged = []
    used = set()

    for i, module in enumerate(modules):
        if i in used:
            continue

        equations = module.get("equations", []) or []
        module_text_norm = _norm_symbol(" ".join(equations))

        for j, other in enumerate(modules):
            if i == j or j in used:
                continue

            other_equations = other.get("equations", []) or []
            should_merge = False

            for eq in other_equations:
                if "=" not in eq:
                    continue

                lhs = _norm_symbol(eq.split("=", 1)[0])

                if lhs and lhs in module_text_norm:
                    should_merge = True
                    break

            if should_merge:
                equations.extend(other_equations)
                module["variables"] = (module.get("variables", []) or []) + (
                    other.get("variables", []) or []
                )
                module["parameters"] = (module.get("parameters", []) or []) + (
                    other.get("parameters", []) or []
                )
                module["source"] = (module.get("source", []) or []) + (
                    other.get("source", []) or []
                )
                used.add(j)

        module["equations"] = list(dict.fromkeys(equations))
        merged.append(module)

    result["process_modules"] = merged

    # Remove missing notes that refer to quantities already present as reported/derived/fixed/estimated.
    available = set()

    for ode in result.get("odes", []):
        for param in ode.get("parameters", []) or []:
            status = _norm_symbol(param.get("status"))

            if status in {"reported", "derived", "fixed", "estimated"}:
                available.add(_norm_symbol(param.get("symbol")))

    filtered_missing = []

    for item in result.get("missing_for_simulation", []) or []:
        item_norm = _norm_symbol(item)

        if any(symbol and symbol in item_norm for symbol in available):
            continue

        filtered_missing.append(item)

    result["missing_for_simulation"] = filtered_missing

    return result


def _save_outputs(result: dict[str, Any], markdown: str) -> None:
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/extracted_evidence.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    with open("outputs/reviewed_model.md", "w", encoding="utf-8") as file:
        file.write(markdown)

    with open("outputs/extraction_review.md", "w", encoding="utf-8") as file:
        file.write(markdown)



def get_discovery_llm(model: str):

    if model.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
        )

    elif model.startswith("deepseek"):
        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("Openrouter_API_KEY"),
        )

    return ChatOpenAI(
        model=model,
        temperature=0,
    )

def run_controlled_discovery(
    vector_store,
    pdf_path: str | None = None,
    model: str = "gpt-4o-mini",
    equation_candidates_dir: str | None = None,
) -> str:
    if pdf_path and equation_candidates_dir:
        try:
            extract_equation_candidates_with_pymupdf(
                pdf_path=pdf_path,
                output_dir=equation_candidates_dir,
            )

            print(f"Equation crops generated in: {equation_candidates_dir}")

        except Exception as error:
            print(
                "Equation crop generation failed:",
                type(error).__name__,
                error,
            )

    print("DISCOVERY STEP A: retrieving discovery context...")

    context = retrieve_discovery_context(vector_store=vector_store)
    table_evidence = retrieve_table_evidence(vector_store=vector_store)

    print(f"DISCOVERY STEP B: context built. chars={len(context)}")

    llm = get_discovery_llm(model)

    print(f"DISCOVERY STEP C: calling LLM model={model}...")


    response = llm.invoke(
        f"""
{SYSTEM_PROMPT}

TABLE PARAMETER EVIDENCE:

Use this section before marking any parameter as missing.

If a symbol appears in a table with value/unit, record it as reported/estimated/fixed according to table text.

Do not mix table rows with equations.

{table_evidence}

RETRIEVED MODEL/EQUATION CONTEXT:

{context}

"""
    )

    print("DISCOVERY STEP D: LLM response received.")

    result = _safe_json_parse(response.content)
    result = _normalize_result(result)

    markdown = format_compact_review(result)
    _save_outputs(result, markdown)

    return markdown
