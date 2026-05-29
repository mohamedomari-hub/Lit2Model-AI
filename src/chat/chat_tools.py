"""
LangChain tool functions used by the Q/A agent.
"""

import re

from langchain_core.tools import tool

from src.modeling.generate_model import (
    compile_reviewed_model_scaffold,
    run_model_discovery_pipeline,
)
from src.retrieval import retrieve_semantic_context
from src.retrieval.equation_search import search_equations
from src.retrieval.figure_search import search_figures
from src.retrieval.mechanism_search import search_mechanisms
from src.retrieval.text_search import search_paper
from src.retrieval.parameter_search import search_parameters
from src.retrieval.simulation_search import search_simulations
from src.retrieval.table_search import search_tables

VECTOR_STORE = None
ACTIVE_PDF_PATH = None
MISSING_EQUATIONS_PATH = "outputs/missing_equations.json"


def _is_modeling_or_mechanism_query(query: str) -> bool:
    query_lower = (query or "").lower()

    markers = [
        "mechanism",
        "modeling",
        "modelling",
        "how is",
        "how are",
        "modeled",
        "modelled",
        "mathematical formulation",
        "equation",
        "dynamics",
        "pk/pd",
        "pkpd",
        "model structure",
        "represented in the model",
        "implemented in the model",
        "effect of",
        "mechanism of",
        "mechanisms reported",
        "how does",
        "affect",
    ]

    return any(marker in query_lower for marker in markers)


def _query_likely_needs_parameters(query: str) -> bool:
    query_lower = (query or "").lower()

    markers = [
        "parameter",
        "constant",
        "value",
        "unit",
        "rate",
        "hill",
        "threshold",
        "maximum effect",
        "half maximal",
    ]

    return any(marker in query_lower for marker in markers)


def _roman_to_int(value: str) -> int | None:
    roman_values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }

    text = (value or "").upper()

    if not text or any(character not in roman_values for character in text):
        return None

    total = 0
    previous = 0

    for character in reversed(text):
        current = roman_values[character]

        if current < previous:
            total -= current
        else:
            total += current
            previous = current

    return total


def _normalize_equation_number(value: str) -> int | None:
    text = (value or "").strip()

    if text.isdigit():
        return int(text)

    return _roman_to_int(text)


def _extract_equation_numbers_with_ranges(text: str) -> tuple[list[str], list[str]]:
    numbers = set()
    expanded_ranges = []
    source_text = text or ""

    number_token = r"(\d+|[IVXLCDM]+)"

    range_pattern = re.compile(
        r"\b(?:eqs?\.?|equations?)\s*"
        + r"\(?"
        + number_token
        + r"\)?\s*(?:[-–]|to)\s*\(?"
        + number_token
        + r"\)?",
        flags=re.IGNORECASE,
    )

    for match in range_pattern.finditer(source_text):
        start = _normalize_equation_number(match.group(1))
        end = _normalize_equation_number(match.group(2))

        if start is not None and end is not None and start <= end and end - start <= 12:
            expanded_ranges.append(f"{start}-{end}")
            for number in range(start, end + 1):
                numbers.add(str(number))

    list_pattern = re.compile(
        r"\b(?:eqs?\.?|equations?)\s+"
        r"((?:\(?\d+\)?\s*(?:,|and)?\s*){2,})",
        flags=re.IGNORECASE,
    )

    for match in list_pattern.finditer(source_text):
        for number in re.findall(r"\d+", match.group(1)):
            numbers.add(number)

    repeated_pattern = re.compile(
        r"\b(?:eqs?\.?|equations?)\s*"
        + r"\(?"
        + number_token
        + r"\)?"
        r"(?:\s*(?:,|and)\s*(?:eqs?\.?|equations?)?\s*\(?"
        + number_token
        + r"\)?)+",
        flags=re.IGNORECASE,
    )

    for match in repeated_pattern.finditer(source_text):
        for value in re.findall(r"\b\d+\b", match.group(0)):
            number = _normalize_equation_number(value)

            if number is not None:
                numbers.add(str(number))

    single_pattern = re.compile(
        r"\b(?:eq\.?|eqs\.?|equation|equations)\s*\(?"
        + number_token
        + r"\)?",
        flags=re.IGNORECASE,
    )

    for match in single_pattern.finditer(source_text):
        number = _normalize_equation_number(match.group(1))

        if number is not None:
            numbers.add(str(number))

    return sorted(numbers, key=lambda value: int(value))[:12], expanded_ranges


def _extract_equation_numbers(text: str) -> list[str]:
    numbers, _ = _extract_equation_numbers_with_ranges(text)
    return numbers


def _deduplicate_text_blocks(text_blocks: list[str]) -> list[str]:
    unique_blocks = []
    seen = set()

    for block in text_blocks:
        clean_block = "\n".join(
            line.strip()
            for line in (block or "").splitlines()
            if line.strip()
        )
        key = re.sub(r"\s+", " ", clean_block).lower()

        if not key or key in seen:
            continue

        seen.add(key)
        unique_blocks.append(block)

    return unique_blocks



def set_active_pdf_path(pdf_path: str):
    global ACTIVE_PDF_PATH
    ACTIVE_PDF_PATH = pdf_path


def set_vector_store(vector_store):
    global VECTOR_STORE
    VECTOR_STORE = vector_store


@tool
def retrieve_text_context(query: str, k: int = 6) -> str:
    """Retrieve general text evidence from the paper."""
    context = search_paper(VECTOR_STORE, query, k=k)
    print(f"QA: text_context_chars={len(context)}")
    return context


@tool
def retrieve_state_context(query: str, state_name: str | None = None) -> str:
    """Retrieve evidence about states, compartments, and dynamic variables."""
    return retrieve_semantic_context(VECTOR_STORE, query, "state", entity=state_name)


@tool
def retrieve_equation_context(
    query: str,
    equation_number: str | None = None,
    ocr_mode: str = "auto",
) -> str:
    """Retrieve equations, ODEs, algebraic equations, and symbols."""
    context = search_equations(
        vector_store=VECTOR_STORE,
        query=query,
        equation_number=equation_number,
        pdf_path=ACTIVE_PDF_PATH,
    )
    print(f"QA: equation_context_chars={len(context)}")
    return context


@tool
def retrieve_parameter_context(query: str, parameter_name: str | None = None) -> str:
    """Retrieve parameter names, values, units, and status evidence."""
    return search_parameters(
        vector_store=VECTOR_STORE,
        query=query,
        parameter_name=parameter_name,
    )


@tool
def retrieve_input_context(query: str, input_name: str | None = None) -> str:
    """Retrieve external inputs, interventions, forcing functions, and doses."""
    return retrieve_semantic_context(VECTOR_STORE, query, "input", entity=input_name)


@tool
def retrieve_observation_context(
    query: str,
    observation_name: str | None = None,
) -> str:
    """Retrieve measured outputs, biomarkers, mappings, and data sources."""
    return retrieve_semantic_context(
        VECTOR_STORE,
        query,
        "observation",
        entity=observation_name,
    )


@tool
def retrieve_mechanism_context(query: str, entity: str | None = None) -> str:
    """Retrieve biological mechanism evidence."""
    mechanism_context = search_mechanisms(
        vector_store=VECTOR_STORE,
        query=query,
        entity=entity,
    )

    is_modeling_route = _is_modeling_or_mechanism_query(query)
    equation_numbers, expanded_ranges = _extract_equation_numbers_with_ranges(
        query + "\n" + mechanism_context
    )
    should_get_parameters = (
        is_modeling_route
        and _query_likely_needs_parameters(query)
    )

    print(
        "QA_MODELING_ROUTE: "
        f"mechanism=True equation={is_modeling_route} "
        f"parameter={should_get_parameters}"
    )
    print(
        "QA_MODELING_ROUTE: "
        f"equation_numbers_detected={equation_numbers}"
    )
    print(
        "QA_EQUATION_FOLLOWUP: "
        f"detected_equation_numbers={equation_numbers}"
    )
    print(
        "QA_EQUATION_FOLLOWUP: "
        f"expanded_ranges={expanded_ranges}"
    )

    if not is_modeling_route:
        return mechanism_context

    extra_sections = [mechanism_context]
    retrieved_equation_blocks = []

    for equation_number in equation_numbers:
        try:
            equation_context = search_equations(
                vector_store=VECTOR_STORE,
                query=f"equation {equation_number} {query}",
                equation_number=equation_number,
                pdf_path=ACTIVE_PDF_PATH,
            )
        except ValueError as error:
            print(
                "QA_MODELING_ROUTE: "
                f"equation {equation_number} skipped: {error}"
            )
            continue

        if equation_context.strip():
            retrieved_equation_blocks.append(
                "\n\nRelevant equation context "
                f"(equation {equation_number}):\n"
                + equation_context
            )

    deduplicated_equation_blocks = _deduplicate_text_blocks(
        retrieved_equation_blocks
    )

    print(
        "QA_EQUATION_FOLLOWUP: "
        f"retrieved_equation_count={len(retrieved_equation_blocks)}"
    )
    print(
        "QA_EQUATION_FOLLOWUP: "
        f"deduplicated_count={len(deduplicated_equation_blocks)}"
    )

    extra_sections.extend(deduplicated_equation_blocks)

    if should_get_parameters:
        parameter_context = search_parameters(
            vector_store=VECTOR_STORE,
            query=query,
        )

        if parameter_context.strip():
            extra_sections.append(
                "\n\nRelevant parameter context:\n"
                + parameter_context
            )

    combined_context = "\n".join(extra_sections)
    print(f"QA_MODELING_ROUTE: combined_context_chars={len(combined_context)}")

    return combined_context


@tool
def retrieve_table_context(query: str, table_number: str | None = None) -> str:
    """Retrieve table context plus GPT Vision table OCR candidate."""
    return search_tables(
        vector_store=VECTOR_STORE,
        query=query,
        table_number=table_number,
        pdf_path=ACTIVE_PDF_PATH,
    )

@tool
def retrieve_figure_context(query: str, figure_number: str | None = None) -> str:
    """Retrieve figure captions plus visual OCR/vision interpretation."""
    return search_figures(
        vector_store=VECTOR_STORE,
        query=query,
        figure_number=figure_number,
        pdf_path=ACTIVE_PDF_PATH,
    )


@tool
def retrieve_simulation_context(query: str) -> str:
    """Retrieve solver, time grid, initial conditions, and scenarios."""
    return search_simulations(VECTOR_STORE, query)


@tool
def retrieve_assumption_context(query: str) -> str:
    """Retrieve assumptions, simplifications, limitations, and uncertainties."""
    return retrieve_semantic_context(VECTOR_STORE, query, "assumption")


@tool
def propose_candidate_ode_model(extracted_summary: str) -> str:
    """Compile extracted evidence into a candidate model scaffold."""
    return compile_reviewed_model_scaffold(extracted_summary)


@tool
def run_model_discovery_workflow(pdf_path: str | None = None) -> str:
    """Run the full scientific model discovery workflow."""
    return run_model_discovery_pipeline(
        VECTOR_STORE,
        pdf_path=pdf_path or ACTIVE_PDF_PATH,
        missing_equations_path=MISSING_EQUATIONS_PATH,
    )
