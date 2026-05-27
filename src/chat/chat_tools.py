"""
LangChain tool functions used by the Q/A agent.
"""

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
    return search_mechanisms(
        vector_store=VECTOR_STORE,
        query=query,
        entity=entity,
    )


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
