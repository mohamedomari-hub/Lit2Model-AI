from typing import List
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from src.gemini_ocr import extract_equations_with_gemini, describe_figure_with_gemini


VECTOR_STORE = None
ACTIVE_PDF_PATH = None


def set_active_pdf_path(pdf_path: str):
    global ACTIVE_PDF_PATH
    ACTIVE_PDF_PATH = pdf_path

def format_doc_metadata(doc, search_query: str, label: str) -> str:
    metadata = doc.metadata or {}

    source_pdf = metadata.get("source_pdf", "unknown_pdf")
    page = metadata.get("page", "unknown")
    section_index = metadata.get("section_index", "unknown")
    modality = metadata.get("modality", "unknown")
    content_type = metadata.get("content_type", "unknown_type")
    figure_number = metadata.get("figure_number", None)
    table_number = metadata.get("table_number", None)
    chunk_id = metadata.get("chunk_id", "unknown_chunk")

    return (
        f"[{label}: {search_query} | "
        f"Source: {source_pdf} | "
        f"Page: {page} | "
        f"Section: {section_index} | "
        f"Modality: {modality} | "
        f"Type: {content_type} | "
        f"Figure: {figure_number} | "
        f"Table: {table_number} | "
        f"Chunk ID: {chunk_id}]"
    )

def _make_chunk_key(doc):
    """
    Stable deduplication key for retrieved chunks.
    """

    metadata = doc.metadata or {}

    return (
        metadata.get("source_pdf", "unknown_pdf"),
        metadata.get("page", "unknown"),
        metadata.get("modality", "unknown"),
        metadata.get("content_type", "unknown_type"),
        metadata.get("section_index", "unknown"),
        doc.page_content[:300],
    )

def set_vector_store(vector_store):
    """
    Store the vector database so the tools can access it.
    """
    global VECTOR_STORE
    VECTOR_STORE = vector_store

def _make_chunk_key(doc):
    metadata = doc.metadata or {}

    return (
        metadata.get("source_pdf", "unknown_pdf"),
        metadata.get("page", "unknown"),
        metadata.get("modality", "unknown"),
        metadata.get("content_type", "unknown_type"),
        metadata.get("section_index", "unknown"),
        doc.page_content[:300],
    )

def _multi_query_retrieve(queries, label: str, k: int = 4):
    """
    Run multiple retrieval queries and deduplicate chunks across all results.
    """

    if VECTOR_STORE is None:
        return "Vector store is not initialized."

    all_results = []
    seen = set()

    modality_priority = {
        "table": 0,
        "marker_markdown": 1,
        "figure_ocr": 2,
        "figure_image": 3,
        "text": 4,
        "figure": 5,
        "unknown": 99,
    }

    for query in queries:
        docs = VECTOR_STORE.similarity_search(query, k=k)

        docs = sorted(
            docs,
            key=lambda doc: modality_priority.get(
                (doc.metadata or {}).get("modality", "unknown"),
                99
            )
        )

        for doc in docs:
            chunk_key = _make_chunk_key(doc)

            if chunk_key in seen:
                continue

            seen.add(chunk_key)

            header = format_doc_metadata(
                doc=doc,
                search_query=query,
                label=label
            )

            all_results.append(
                f"{header}\n{doc.page_content}"
            )

    if not all_results:
        return "No relevant context was retrieved from the PDF."

    return "\n\n---\n\n".join(all_results)

@tool
def retrieve_pdf_context(query: str) -> str:
    """
    Retrieve relevant chunks from the PDF for a specific user question.

    This is used for general Q&A:
    - tables
    - figures
    - parameters
    - equations
    - mechanisms
    - assumptions
    - model reconstruction questions
    """

    if VECTOR_STORE is None:
        return "Vector store is not initialized."

    query_lower = query.lower()

    search_queries = [query]

    # --------------------------------------------------
    # General table retrieval
    # --------------------------------------------------
    if "table" in query_lower:
        search_queries.extend([
            "table caption",
            "table content rows columns",
            "table values units source",
            "supplementary table values units",
        ])

    # --------------------------------------------------
    # General figure / diagram retrieval
    # --------------------------------------------------
    if (
        "figure" in query_lower
        or "fig" in query_lower
        or "diagram" in query_lower
        or "graph" in query_lower
    ):
        search_queries.extend([
            "figure caption",
            "diagram caption",
            "model figure mechanism",
            "system diagram compartments interactions",
            "mechanism diagram arrows stimulation inhibition",
        ])

    # --------------------------------------------------
    # Parameter-specific retrieval
    # --------------------------------------------------
    if (
        "parameter" in query_lower
        or "rate" in query_lower
        or "constant" in query_lower
        or "value" in query_lower
        or "unit" in query_lower
        or "estimated" in query_lower
    ):
        search_queries.extend([
            "model parameters values units",
            "parameter descriptions values units",
            "rate constants effect parameters",
            "estimated parameters fixed parameters source",
            "bioavailability clearance volume distribution absorption elimination",
        ])

    # --------------------------------------------------
    # Equation-specific retrieval
    # --------------------------------------------------
    if (
        "equation" in query_lower
        or "ode" in query_lower
        or "function" in query_lower
        or "formula" in query_lower
        or "differential" in query_lower
        or "derive" in query_lower
    ):
        search_queries.extend([
            "ordinary differential equation model",
            "differential equations dynamic model",
            "algebraic equation coupling function",
            "effect function stimulation inhibition",
            "Hill function Emax EC50 IC50 equation",
            "effect compartment equation delay equation",
            "model modifying terms treatment effect equation",
        ])

    # --------------------------------------------------
    # Mechanism-specific retrieval
    # --------------------------------------------------
    if (
        "mechanism" in query_lower
        or "affect" in query_lower
        or "effect" in query_lower
        or "stimulate" in query_lower
        or "inhibit" in query_lower
        or "increase" in query_lower
        or "decrease" in query_lower
        or "regulate" in query_lower
    ):
        search_queries.extend([
            "biological mechanism stimulation inhibition regulation",
            "pharmacodynamic mechanism drug effect",
            "coupling hypotheses biological mechanisms",
            "drug affects glucose metabolism insulin glucagon",
            "model mechanisms used in equations",
        ])

    # --------------------------------------------------
    # Assumption / limitation retrieval
    # --------------------------------------------------
    if (
        "assumption" in query_lower
        or "assumptions" in query_lower
        or "limitation" in query_lower
        or "limitations" in query_lower
        or "uncertain" in query_lower
        or "uncertainty" in query_lower
        or "missing" in query_lower
    ):
        search_queries.extend([
            "model assumptions simplifications",
            "we assume model assumption",
            "assumed equal steady state",
            "limitations uncertainty unclear",
            "missing information model validation assumptions",
        ])

    # --------------------------------------------------
    # Model rebuilding / implementation retrieval
    # --------------------------------------------------
    if (
        "rebuild" in query_lower
        or "implement" in query_lower
        or "python" in query_lower
        or "code" in query_lower
        or "simulate" in query_lower
        or "simulation" in query_lower
        or "reconstruct" in query_lower
        or "build this model" in query_lower
    ):
        search_queries.extend([
            "model equations parameters initial conditions",
            "state variables differential equations model implementation",
            "parameter table values units",
            "coupling hypotheses mathematical model",
            "simulation inputs initial conditions assumptions",
            "model structure compartments equations",
            "pharmacokinetic pharmacodynamic model implementation",
            "data used for parameter estimation validation simulation",
        ])

    # --------------------------------------------------
    # Deduplicate search queries
    # --------------------------------------------------
    search_queries = list(dict.fromkeys(search_queries))

    results = []
    seen = set()

    for search_query in search_queries:

        docs = VECTOR_STORE.similarity_search(
            search_query,
            k=8
        )

        for doc in docs:
            chunk_key = _make_chunk_key(doc)

            if chunk_key in seen:
                continue

            seen.add(chunk_key)

            content = doc.page_content

            if "Start of picture text" in content:
                content = (
                    "[FIGURE OCR TEXT: visual figure was omitted by parser, "
                    "but OCR extracted visible labels, axes, legends, or units.]\n"
                    + content
                )

            header = format_doc_metadata(
                doc=doc,
                search_query=search_query,
                label="Search Query"
            )

            results.append(
                f"{header}\n{content}"
            )

            if not results:
                return "No relevant context was retrieved from the PDF."

            return "\n\n---\n\n".join(results)

@tool
def retrieve_model_building_context(_: str = "") -> str:
    """
    Retrieve broad mechanistic model-building context.
    """

    queries = [
        "mechanistic model structure compartments interactions",
        "pharmacokinetic pharmacodynamic model structure",
        "biological mechanisms stimulation inhibition feedback",
        "model assumptions coupling hypotheses",
        "effect compartment delayed drug effect",
        "system diagram model figure mechanism network",
        "metabolic reproductive feedback model hormones glucose insulin",
    ]

    return _multi_query_retrieve(
        queries=queries,
        label="Model-building Query",
        k=4
    )

def retrieve_parameter_context() -> str:
    """
    Helper function: retrieve parameter-specific context.
    Not exposed directly to the agent.
    """

    queries = [
        "model parameters table values units",
        "parameter values rate constants coefficients units",
        "half maximal inhibition stimulation concentration parameter table value unit",
        "equations parameters variables constants model",
        "initial conditions parameter estimates fitted estimated fixed",
        "clearance volume dose bioavailability absorption elimination",
        "Hill coefficient EC50 IC50 Emax Km Vmax threshold half maximal",
        "ordinary differential equations parameters rates compartments",
        "supplementary table model parameters values",
    ]

    return _multi_query_retrieve(
        queries=queries,
        label="Parameter Query",
        k=4
    )

def extract_parameters(parameter_context: str) -> str:
    """
    Helper function: extract reported, derived, and missing parameters.
    Not exposed directly to the agent.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a parameter extraction assistant for scientific mechanistic modelling papers.

Use only the provided context.

Extract parameters into exactly these sections:

1. reported_parameters
- Parameters with explicit numerical values.
- Copy value and unit exactly as written.
- Include name/symbol, meaning, value, unit, evidence, and page.

2. derived_parameters
- Parameters defined through equations, symbolic relationships, assumptions, or formulas.
- Include name/symbol, formula or relationship, evidence, and page.
- A derived parameter does NOT need a numerical value.

3. missing_parameters
- Parameters mentioned but without value, unit, formula, relationship, or derivation.
- Do not include a parameter here if it appears in a table or equation.
- Do not include a parameter here if it is symbolically defined.

4. model_quantities_for_review
- State variables, concentrations, outputs, biomarkers, observed quantities, or simulation results that may look like parameters but should not be classified as missing parameters.
- Include name/symbol, meaning, evidence, and page if available.

Table priority rule:
- Scientific parameter tables are the highest priority source.
- If a parameter appears in a table with value and unit, it MUST be classified as reported.
- Narrative text cannot override table information.
- Before classifying any parameter as missing, scan tables again.

Derived-parameter detection:
- Search the context for symbolic relationships, equalities, formulas, or definitions.
- If a symbol/name appears on the left-hand side of an equation, classify it as derived unless it also has a reported numeric value.
- If a parameter is described as being calculated, expressed, defined, written, assumed equal, or derived from other quantities, classify it as derived.
- Do not require a numeric value for derived parameters.

Generic patterns to detect:
- symbol = expression
- symbol is expressed as ...
- symbol is calculated from ...
- symbol is defined as ...
- symbol is assumed equal to ...
- symbol can be written as ...
- symbol is proportional to ...

State-variable / model-quantity rule:
- Do not classify concentrations, compartments, biomarkers, observed outputs, simulation outputs, or dynamic quantities as missing parameters.
- Symbols such as concentrations, amounts, states, compartments, maximum observed concentrations, and measured variables should be listed under model_quantities_for_review, not missing_parameters.
- Missing parameters must be parameters, constants, coefficients, rates, thresholds, exponents, doses, volumes, clearances, initial conditions, or fixed numerical values.
- If uncertain whether an item is a parameter or a state/model quantity, place it under model_quantities_for_review.

Critical rules:
- Do not invent parameters.
- Do not convert units.
- Preserve units exactly.
- Prefer parameter tables over explanatory text.
- If a value appears in a table, it is reported, not missing.
- If a parameter is defined by equality, relationship, or formula, it is derived, not missing.
- Before listing any missing parameter, check whether it appears in a table, equation, symbolic relationship, or definition.

Context:
{parameter_context}
"""

    result = llm.invoke(prompt)
    return result.content

@tool
def extract_parameters_from_pdf(_: str = "") -> str:
    """
    Retrieve parameter-specific context and extract parameters in one step.
    """

    parameter_context = retrieve_parameter_context()

    extracted_parameters = extract_parameters(
        parameter_context
    )

    return extracted_parameters

def retrieve_equation_context() -> str:
    """
    Helper function: retrieve equation-specific context.
    Not exposed directly to the agent.
    """

    queries = [
        "ordinary differential equations mathematical model equations",
        "differential equations dynamic model state variables rates",
        "algebraic equations model coupling functions",
        "pharmacodynamic effect function stimulation inhibition equation",
        "Hill function Emax EC50 IC50 half maximal effect equation",
        "compartment model transfer elimination clearance equation",
        "delay effect compartment indirect response equation",
        "model equations uptake secretion production degradation terms",
        "feedback regulation equation biological mechanism model",
        "parameter definitions symbolic relationships derived equations",
        "model modifying terms treatment drug intervention effect equation",
        "simulation model equations initial conditions assumptions",
    ]

    return _multi_query_retrieve(
        queries=queries,
        label="Equation Query",
        k=6
    )

def extract_equations(equation_context: str) -> str:
    """
    Helper function: extract reported ODEs, algebraic equations,
    Hill functions, coupling equations, and pharmacodynamic equations.
    Not exposed directly to the agent.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
    You are a scientific mechanistic model equation extractor.

    Your task is to extract mathematical equations from retrieved context.
    Use only the provided context.

    Return exactly these sections:

    1. reported_odes
    - Explicitly reported differential equations or dynamic equations.
    - Copy the full equation exactly when possible.
    - Include page number and evidence.
    - Do not summarize ODEs in words.
    - Do not include parameter definitions here.

    2. reported_algebraic_or_coupling_equations
    - Explicitly reported algebraic equations, coupling equations, effect functions,
    stimulation/inhibition functions, feedback functions, compartment transfer equations,
    Hill functions, Michaelis-Menten functions, Emax/EC50/IC50 relationships,
    balance equations, or model-modifying terms.
    - Copy the full equation exactly when possible.
    - Include page number and evidence.
    - Do not include prose parameter descriptions here.

    3. parameter_definitions_or_model_quantities
    - Symbol definitions, parameter meanings, state-variable descriptions,
    units, table-derived descriptions, and model quantity descriptions.
    - These are not reported equations unless a mathematical relationship is explicitly shown.

    4. inferred_or_missing_equations
    - Equations referenced but not fully visible.
    - Equations partially omitted by PDF parsing.
    - Equations that require OCR or human review.
    - Do not invent missing mathematics.

    5. human_review_flags
    - Flag damaged OCR.
    - Flag missing displayed equations.
    - Flag ambiguous subscripts/superscripts.
    - Flag any equation reconstructed from partial context.
    - If Gemini OCR failed or parser omitted equation images, explicitly say equations require human review.

    Important distinction rules:
    - Distinguish between mathematical equations, parameter definitions,
    variable descriptions, mechanistic explanations, and assumptions.
    - Only classify content as an equation if it expresses a mathematical relationship
    between quantities.
    - Do not classify textual descriptions of symbols, units, mechanisms,
    or parameter-table entries as equations.
    - If a mathematical equation is referenced but visually omitted,
    partially missing, or OCR is incomplete, place it under inferred_or_missing_equations.
    - Distinguish explicitly reported equations from partially observed equations,
    inferred model structure, and missing information.

    Equation extraction rules:
    - Extract equations exactly as written.
    - Preserve mathematical symbols.
    - If an ODE is present, copy the full equation.
    - Do not summarize equations in words.
    - Prefer displayed equations over narrative descriptions.
    - Include compartment equations, Hill functions, Emax/EC50/IC50 relationships,
    algebraic couplings, balance equations, and delay/effect-compartment equations
    only when they are explicitly present.

    Variable-name formatting rule:
    - If a biological/model term is written with hyphens, subscripts, or special characters,
    treat it as one variable name unless the context clearly shows mathematical subtraction.
    - Convert complex variable names to readable underscore notation only when needed.
    - Do not change the biological meaning.

    Equation cleanup rule:
    - When PDF extraction splits superscripts or subscripts, reconstruct them carefully using nearby text and symbols.
    - Preserve the intended equation, but do not invent new terms.
    - Do not add artificial divisions, missing powers, or missing denominators.

    Fallback rule:
    - If the context says Gemini OCR failed, parser omitted images, or displayed equations were not retrieved:
    do not invent equations.
    Do not classify parameter descriptions as equations.
    Place the affected equations under inferred_or_missing_equations.
    Add a human_review_flags note.

    Context:
    {equation_context}
    """

    result = llm.invoke(prompt)
    return result.content

@tool
def extract_equations_from_pdf():
    """
    Retrieve and extract equations from the paper.

    Uses Gemini OCR only when parser output suggests
    omitted equation images or missing math.
    """

    equation_context = retrieve_equation_context()

    # ----------------------------------------
    # Trigger Gemini fallback only if needed
    # ----------------------------------------
    needs_gemini = any(
        trigger in equation_context.lower()
        for trigger in [
            "picture",
            "omitted",
            "image",
            "==>",
        ]
    )

    gemini_equations = ""

    if (
        needs_gemini
        and ACTIVE_PDF_PATH is not None
    ):
        print(
            "Gemini equation OCR fallback triggered."
        )

        gemini_equations = (
            extract_equations_with_gemini(
                ACTIVE_PDF_PATH
            )
        )

    # ----------------------------------------
    # Combine retrieval + Gemini OCR
    # ----------------------------------------
    combined_equation_context = f"""
TEXT RETRIEVAL EQUATION CONTEXT:
{equation_context}

GEMINI OCR EQUATION CONTEXT:
{gemini_equations}
"""

    extracted_equations = extract_equations(
        combined_equation_context
    )

    # ----------------------------------------
    # Return debug output
    # ----------------------------------------
    return f"""
RAW EQUATION RETRIEVAL CONTEXT:
{equation_context}

GEMINI OCR EQUATION CONTEXT:
{gemini_equations}

==================================================

EXTRACTED EQUATIONS:
{extracted_equations}
"""

def is_exact_figure_match(content: str, metadata: dict, figure_number: str) -> bool:
    """
    Check whether retrieved content explicitly corresponds to the requested figure.
    """

    metadata = metadata or {}
    content_lower = content.lower()

    if str(metadata.get("figure_number")) == str(figure_number):
        return True

    patterns = [
        rf"\bfigure\s+{figure_number}\b",
        rf"\bfig\.\s*{figure_number}\b",
        rf"\bfig\s+{figure_number}\b",
    ]

    return any(
        re.search(pattern, content_lower)
        for pattern in patterns
    )

def retrieve_figure_context(query: str) -> str:
    """
    Retrieve figure-specific evidence:
    captions, OCR picture text, nearby discussion, and figure references.

    Uses exact figure-number matches first.
    Falls back to semantic figure retrieval only if no exact match is found.
    """

    if VECTOR_STORE is None:
        return "Vector store is not initialized."

    query_lower = query.lower()

    figure_match = re.search(
        r"(figure|fig\.?|fig)\s*(\d+)",
        query_lower
    )

    search_queries = [query]

    figure_number = None

    if figure_match:
        figure_number = figure_match.group(2)

        search_queries.extend([
            f"Figure {figure_number}",
            f"Fig {figure_number}",
            f"Fig. {figure_number}",
            f"Figure {figure_number} caption",
            f"Fig. {figure_number} caption",
            f"description of Figure {figure_number}",
            f"results shown in Figure {figure_number}",
            f"text discussing Figure {figure_number}",
        ])

    else:
        search_queries.extend([
            "figure caption",
            "figure description",
            "figure results",
            "diagram caption",
            "model diagram",
        ])

    search_queries = list(dict.fromkeys(search_queries))

    exact_results = []
    fallback_results = []
    seen = set()

    for search_query in search_queries:

        docs = VECTOR_STORE.similarity_search(
            search_query,
            k=6
        )

        for doc in docs:
            chunk_key = _make_chunk_key(doc)

            if chunk_key in seen:
                continue

            seen.add(chunk_key)

            content = doc.page_content

            if "Start of picture text" in content:
                content = (
                    "[FIGURE OCR TEXT: visual figure was omitted by parser, "
                    "but OCR extracted visible labels, axes, legends, or units.]\n"
                    + content
                )

            header = format_doc_metadata(
                doc=doc,
                search_query=search_query,
                label="Figure Search Query"
            )

            entry = f"{header}\n{content}"

            if figure_number is not None:
                if is_exact_figure_match(
                    content=content,
                    metadata=doc.metadata or {},
                    figure_number=figure_number
                ):
                    exact_results.append(entry)
                else:
                    fallback_results.append(entry)
            else:
                fallback_results.append(entry)

    if exact_results:
        return "\n\n---\n\n".join(exact_results)

    if figure_number is not None and fallback_results:
        warning = (
            f"No exact Figure {figure_number} match was found. "
            "The following chunks were retrieved semantically and may not correspond "
            "to the requested figure."
        )

        return warning + "\n\n---\n\n" + "\n\n---\n\n".join(fallback_results)

    if fallback_results:
        return "\n\n---\n\n".join(fallback_results)

    return "No figure-specific context was retrieved from the PDF."



def should_use_gemini_for_figure(figure_context: str) -> bool:
    """
    Decide whether Gemini vision should be used for figure interpretation.
    """

    context_lower = figure_context.lower()

    triggers = [
        "picture intentionally omitted",
        "figure image extracted",
        "vision description skipped",
        "no exact figure",
        "may not correspond to the requested figure",
        "ocr",
        "visual figure was omitted",
    ]

    return any(trigger in context_lower for trigger in triggers)

def extract_first_page_number_from_context(figure_context: str):
    match = re.search(r"Page:\s*(\d+)", figure_context)

    if match:
        return int(match.group(1))

    return None

def render_pdf_page_to_image(pdf_path: str, page_number: int) -> str:
    """
    Render a PDF page to an image for Gemini vision.

    page_number is 1-based.
    """

    import os
    import fitz

    os.makedirs("outputs/gemini_pages", exist_ok=True)

    pdf = fitz.open(pdf_path)

    page_index = page_number - 1

    if page_index < 0 or page_index >= len(pdf):
        raise ValueError(f"Invalid page number: {page_number}")

    page = pdf[page_index]

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

    image_path = f"outputs/gemini_pages/page_{page_number}.png"

    pix.save(image_path)

    return image_path

def get_gemini_figure_fallback(figure_context: str) -> str:
    """
    Use Gemini vision on the retrieved figure page if figure context is incomplete.
    """

    if ACTIVE_PDF_PATH is None:
        return "Gemini figure fallback skipped: active PDF path is not set."

    page_number = extract_first_page_number_from_context(figure_context)

    if page_number is None:
        return "Gemini figure fallback skipped: no page number found in retrieved context."

    try:
        image_path = render_pdf_page_to_image(
            pdf_path=ACTIVE_PDF_PATH,
            page_number=page_number
        )

        gemini_description = describe_figure_with_gemini(image_path)

        return f"""
GEMINI FIGURE VISION FALLBACK
Rendered PDF page: {page_number}
Image path: {image_path}

{gemini_description}
"""

    except Exception as error:
        return f"""
Gemini figure fallback failed.

Reason:
{type(error).__name__}: {error}
"""

def extract_figure_explanation(
    figure_context: str,
    user_question: str
) -> str:
    """
    Convert retrieved figure context into a cautious scientific explanation.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a scientific figure interpretation assistant.

Use only the retrieved figure context to answer the user's question.

User question:
{user_question}

Retrieved figure context:
{figure_context}

Rules:
- If the context contains "FIGURE OCR TEXT", treat it as OCR-derived visual evidence.
- Extract visible axis labels, units, variables, legends, and panel labels.
- Separate:
  1. Directly retrieved evidence
  2. Interpretation
  3. Limitations
- Do not invent curve shapes or trends if they are not visible in the retrieved context.
- Do not say no figure context was found if OCR figure text is available.
- If the original visual image is unavailable, clearly say that detailed visual interpretation is limited.
- Be scientifically cautious.

Critical limitation wording rule:
- If GEMINI VISUAL FIGURE CONTEXT contains a non-empty Gemini response, NEVER write:
  "the visual representation was absent"
  "the actual figure image was absent"
  "without the visual representation"
- Instead write:
  "The text parser omitted the original figure image, but Gemini analyzed a rendered PDF page image."
- Mention that fine visual details still require human review.

Limitations section rule:
- If Gemini visual context is available, the limitation is NOT absence of the image.
- The limitation is that the figure was interpreted from a rendered PDF page and should be human-reviewed for fine details.

Return a clear explanation.
"""

    result = llm.invoke(prompt)

    return result.content

@tool
def explain_figure_from_pdf(query: str) -> str:
    """
    Retrieve and explain figure-specific context from the active PDF.
    Use this when the user asks about a figure, plot, diagram, or visual result.
    """

    figure_context = retrieve_figure_context(query)

    gemini_context = ""

    if should_use_gemini_for_figure(figure_context):
        gemini_context = get_gemini_figure_fallback(
            figure_context
        )

    combined_context = f"""
RETRIEVED FIGURE CONTEXT:
{figure_context}

GEMINI VISUAL FIGURE CONTEXT:
{gemini_context}
"""

    answer = extract_figure_explanation(
        figure_context=combined_context,
        user_question=query
    )

    return f"""
FIGURE RETRIEVAL CONTEXT:
{figure_context}

GEMINI VISUAL FIGURE CONTEXT:
{gemini_context}

==================================================

FIGURE EXPLANATION:
{answer}
"""



@tool
def extract_mechanisms(model_context: str) -> str:
    """
    Extract model-relevant biological/pharmacological mechanisms.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a mechanism extraction assistant for mechanistic modelling papers.

Use only the provided context.

Extract mechanisms into exactly these sections:

1. model_mechanisms
- Mechanisms explicitly used in the mathematical model,
  coupling hypotheses, simulations, or model diagrams.

2. background_mechanisms
- Biological background mechanisms mentioned in the paper
  but not clearly included in the mathematical model.

3. assumptions
- Modelling assumptions, simplifications, excluded mechanisms,
  or unresolved biological uncertainties.

Mechanism classification rule:
- Separate mechanisms explicitly used in the mathematical model from general biological background.
- A mechanism belongs to model_mechanisms only if it appears in equations, coupling hypotheses, model diagrams, simulation assumptions, or model terms.
- Otherwise put it under background_mechanisms.

Critical rules:
- Do not mix background biology with model structure.
- Prioritize mechanisms used in equations, model diagrams, coupling hypotheses, or simulations.
- Do not invent mechanisms.

Context:
{model_context}
"""

    result = llm.invoke(prompt)
    return result.content

class MechanismEdge(BaseModel):
    source: str
    relation: str
    target: str
    evidence: str
    page: str | None = None

class MechanismGraph(BaseModel):
    edges: List[MechanismEdge]

@tool
def extract_mechanism_edges(context: str) -> str:
    """
    Extract graph-ready source-relation-target edges from model mechanisms.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    structured_llm = llm.with_structured_output(MechanismGraph)

    prompt = f"""
    You are extracting graph-ready mechanistic model edges from a scientific modelling paper.

    Your goal is to reconstruct the mechanistic structure of the model as a graph.

    Extract only relationships explicitly used in:
    - mathematical equations
    - mechanistic model structure
    - coupling hypotheses
    - model diagrams
    - compartment diagrams
    - simulation assumptions
    - explicit model descriptions

    Do NOT extract:
    - general background biology
    - speculative mechanisms
    - literature discussion not directly implemented in the model
    - observational statements unless explicitly modeled

    Causal and directional extraction rules:
    1. Preserve directionality exactly as reported in the paper.
    Do not reverse causal, transport, regulatory,
    or compartmental relationships.

    2. Only extract relationships explicitly supported by:
    - equations
    - figure captions
    - model diagrams
    - table descriptions
    - mechanistic text
    - model assumptions

    3. Do not infer biological effects unless explicitly stated.

    4. Distinguish carefully between:
    - stimulation
    - inhibition
    - activation
    - repression
    - transport
    - transfer
    - conversion
    - degradation
    - elimination
    - production
    - consumption
    - regulation
    - feedback
    - delay
    - association/correlation

    5. If directionality, mechanism, or causal meaning is ambiguous:
    - mark as uncertain
    - include supporting evidence
    - prefer human review

    6. Prefer relationships directly supported by equations
    over qualitative narrative descriptions.

    Allowed relation types:
    - stimulates
    - inhibits
    - activates
    - represses
    - increases
    - decreases
    - transfers_to
    - transports_to
    - converts_to
    - produces
    - consumes
    - degrades
    - eliminates
    - regulates
    - delays_effect_on
    - feedback_positive
    - feedback_negative
    - associated_with
    - uncertain_relation

    For each edge return:

    - source
    - relation
    - target
    - confidence
        * explicit_equation
        * explicit_model_text
        * figure_supported
        * inferred_uncertain
    - evidence
    - page if available
    - requires_human_review (true/false)

    Graph extraction rules:
    - Return only graph-ready mechanistic edges.
    - Use consistent biological/entity names.
    - Preserve compartment names exactly when possible.
    - If an edge comes only from narrative text and is not clearly implemented in the model,
    mark requires_human_review = true.
    - If OCR/parser omitted diagrams or equations,
    avoid inventing missing edges.

    If the source text does not explicitly contain a source-relation-target relationship, do not create an edge.
    Do not convert vague biological statements into graph edges.
    Do not use "produces", "degrades", "transfers_to", or "regulates" unless the exact relationship is explicitly stated.
    If unsure, omit the edge rather than guessing.
    Prefer fewer high-confidence edges over many speculative edges.

    Context:
    {context}
    """

    result = structured_llm.invoke(prompt)
    return result.model_dump_json(indent=2)

@tool
def generate_mechanism_graph(edges_json: str) -> str:
    """
    Generate an interactive HTML mechanism graph from mechanism edges JSON.
    """

    import os
    import json
    from pyvis.network import Network

    os.makedirs("outputs", exist_ok=True)

    data = json.loads(edges_json)

    if isinstance(data, dict) and "edges" in data:
        edges = data["edges"]
    else:
        edges = data

    output_path = "outputs/mechanism_graph.html"

    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False
    )

    net.force_atlas_2based()

    for edge in edges:
        source = edge.get("source", "Unknown source")
        target = edge.get("target", "Unknown target")
        relation = edge.get("relation", "related_to")
        evidence = edge.get("evidence", "")
        page = edge.get("page", "")

        net.add_node(source, label=source, title=source)
        net.add_node(target, label=target, title=target)

        net.add_edge(
            source,
            target,
            label=relation,
            title=(
                f"Relation: {relation}<br>"
                f"Page: {page}<br>"
                f"Evidence: {evidence}"
            )
        )

    net.write_html(output_path)

    return f"Mechanism graph generated and saved to {output_path}"

@tool
def propose_candidate_ode_model(extracted_summary: str) -> str:
    """
    Compile extracted parameters, equations, and mechanisms
    into a candidate mechanistic model scaffold.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a scientific report compiler.

Your task is to compile extracted scientific evidence into a clean mechanistic model scaffold.

You are NOT an extractor in this step.
You are NOT allowed to add new scientific information.

Use only the extracted information provided below.

SOURCE-OF-TRUTH RULES
- Use PARAMETER EXTRACTION OUTPUT as the only source for parameters.
- Use EQUATION EXTRACTION OUTPUT as the only source for equations.
- Use MECHANISM EXTRACTION OUTPUT as the only source for mechanisms.
- Use GRAPH EDGES OUTPUT as the only source for graph edges.
- Do not use prior pharmacology or biology knowledge.
- Do not invent, infer, simplify, derive, or correct equations.
- Do not independently decide what is missing.
- Do not classify state variables or observed quantities as missing parameters.
- Preserve all reported values, units, symbols, and source labels exactly.

EQUATION PRESERVATION RULES
- If EQUATION EXTRACTION OUTPUT contains reported_odes, copy them exactly into section 6.
- If EQUATION EXTRACTION OUTPUT contains reported_algebraic_or_coupling_equations, copy them exactly into section 7.
- Do not omit equations.
- Do not rewrite equations.
- Do not summarize equations in prose.
- Do not replace equations with parameter descriptions.
- Preserve labels such as "(GEMINI OCR CONTEXT)".
- If an equation is marked as GEMINI OCR CONTEXT, OCR-derived, uncertain, or partially reconstructed, include it but append:
  "(requires human review)".
- If no equations were extracted, write "not extracted".

PARAMETER RULES
- Copy reported parameters exactly.
- Copy derived parameters exactly.
- Copy missing parameters exactly.
- If no missing parameters were extracted, write "None reported by extraction."
- Do not move model quantities, observations, inputs, or states into missing parameters.
- Never place state variables, concentrations, observed data, model outputs, doses, or compartments under Missing parameters.
- If such quantities appear under missing_parameters in extracted text, move them to "State variables / model quantities" or "Missing information / human-review notes".

GRAPH RULES
- Include only graph edges supported by GRAPH EDGES OUTPUT.
- Remove duplicate edges with the same scientific meaning.
- Do not include reverse directions unless explicitly supported.
- Prefer fewer high-confidence edges over speculative edges.
- If an edge conflicts with mechanisms or equations, move it to section 11 as requiring review.
- If an edge is uncertain, include it only if clearly labelled "(requires human review)".

COMPARTMENT FLOWCHART RULES
- Create a simple text flowchart using arrows.
- Include only compartmental or process-flow relationships explicitly supported by extracted evidence.
- Do not include speculative graph edges.

- Use the transfer/linking rate explicitly supported by
  equations, mechanisms, or graph extraction.

- Do not substitute:
  clearance,
  elimination,
  degradation,
  or dissipation parameters
  for compartment-transfer parameters.

- If the transfer parameter is ambiguous, write:
  "↓ transfer rate (requires human review)"

- Use this style when supported:

  IM injection
      ↓ ka
  Central compartment / C
      ↓ ke1
  Effect compartment / Ce
      ↓ delayed pharmacodynamic effect
  Response / model output

FORMATTING RULES
- Copy equations exactly as they appear in EQUATION EXTRACTION OUTPUT.
- Do not reformat LaTeX equations.
- Do not convert LaTeX into plain Unicode math.
- Do not split equations across many lines.
- If an equation is already wrapped in $$ ... $$, preserve it exactly.
- If an equation is not wrapped, wrap the entire equation in one $$ ... $$ block.

REQUIRED OUTPUT SECTIONS
1. Model scope
2. State variables / model quantities
3. Reported parameters
4. Derived parameters
5. Missing parameters
6. Reported ODEs
7. Reported algebraic/coupling equations
8. Model mechanisms
9. Mechanism graph summary
10. Compartment flowchart
11. Missing information / human-review notes

IMPORTANT
- This is a draft scaffold based only on extracted evidence.
- Do not claim the model is validated.

EXTRACTED INFORMATION:
{extracted_summary}
"""

    result = llm.invoke(prompt)
    return result.content

@tool
def run_model_discovery_workflow(_: str = "") -> str:
    """
    Run the full scientific model discovery workflow in a fixed order.

    The agent calls this one tool, while the internal scientific workflow
    runs deterministically.
    """

    model_context = retrieve_model_building_context.invoke("")

    parameters = extract_parameters_from_pdf.invoke("")

    equations = extract_equations_from_pdf.invoke("")

    mechanisms = extract_mechanisms.invoke({
        "model_context": model_context
    })

    edges = extract_mechanism_edges.invoke({
        "context": mechanisms
    })

    graph_result = generate_mechanism_graph.invoke({
        "edges_json": edges
    })

    combined_summary = f"""
PARAMETER EXTRACTION OUTPUT:
{parameters}

EQUATION EXTRACTION OUTPUT:
{equations}

MECHANISM EXTRACTION OUTPUT:
{mechanisms}

GRAPH EDGES OUTPUT:
{edges}

GRAPH GENERATION OUTPUT:
{graph_result}
"""
    import os

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/extraction_review.md", "w", encoding="utf-8") as file:
        file.write(combined_summary)

    final_answer = propose_candidate_ode_model.invoke({
        "extracted_summary": combined_summary
    })

    return final_answer