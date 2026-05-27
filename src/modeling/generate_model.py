import os
import re
import json
from langchain_openai import ChatOpenAI

from src.ingestion.scientific_assets import (
    save_artifact_index,
    format_artifact_index_for_review,
)

from src.ingestion.ocr import extract_equations_with_gemini
from src.modeling.equation_recovery import (
    extract_equation_candidates_from_text_layer,
    recover_missing_numbered_equations,
)

from src.ingestion.scientific_assets import (
    save_artifact_index,
    format_artifact_index_for_review,
)

from src.modeling.graph_generation import (
    extract_mechanism_edges_service,
    generate_mechanism_graph_service,
)
from src.retrieval import multi_query_retrieve
from src.retrieval.context import retrieve_semantic_context
from src.retrieval import retrieve_equation_context_service

VECTOR_STORE = None
ACTIVE_PDF_PATH = None



def save_text(path: str, content: str):
    """
    Save text content to a file.
    """

    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def save_json(path: str, data):
    """
    Save dictionary/list data to JSON.
    """

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def generate_simple_ode_template():
    """
    Generate a simple Python ODE scaffold for the Dexa PK/effect-compartment part.
    This is not the full validated MetRep model.
    """

    code = '''
import numpy as np


def dexa_pk_effect_model(t, y, p):
    """
    Candidate scaffold for dexamethasone PK/effect-compartment model.

    State variables:
    y[0] = C  : dexamethasone concentration in central compartment
    y[1] = Ce : dexamethasone concentration in effect compartment

    Parameters:
    p["dose"] = dose per kg body weight
    p["F"]    = bioavailability
    p["ka"]   = absorption rate constant
    p["ke"]   = elimination rate constant
    p["Vd"]   = volume of distribution
    p["keo"]  = effect compartment equilibration rate
    """

    C, Ce = y

    dose = p["dose"]
    F = p["F"]
    ka = p["ka"]
    ke = p["ke"]
    Vd = p["Vd"]
    keo = p["keo"]

    CL = ke * Vd

    dCdt = (dose * F * ka * np.exp(-ka * t) - CL * C) / Vd
    dCedt = keo * (C - Ce)

    return [dCdt, dCedt]


def effect_dxm_gluca(Ce, p):
    """
    Candidate pharmacodynamic stimulation function for glucagon secretion.
    """

    Emax = p["Emax"]
    Ca = p["Ca"]

    return 1 + Emax * (Ce**10 / (Ce**10 + Ca**10))


def effect_dxm_bt(Ce, p):
    """
    Candidate pharmacodynamic inhibition function for glucose uptake.
    """

    Cb = p["Cb"]

    return 1 - (Ce**7 / (Ce**7 + Cb**7))
'''

    return code


def save_candidate_model_code(path: str):
    """
    Save candidate Python model code to file.
    """

    code = generate_simple_ode_template()

    with open(path, "w", encoding="utf-8") as file:
        file.write(code)



def generate_python_model_from_extraction(
    reviewed_extraction: str,
    simulation_setup: dict
) -> str:
    """
    Generate executable Python model code from reviewed extraction
    and user-provided simulation setup.

    This is model-agnostic and should work for different mechanistic papers,
    as long as equations and required inputs are available.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are an expert scientific modeller and Python developer.

Generate executable Python code to simulate the reviewed mechanistic model.

Use:
- numpy
- scipy.integrate.solve_ivp
- matplotlib

Source of truth:
- Use only the reviewed extraction and simulation setup.
- Do not invent equations.
- Do not invent mechanisms.
- Do not invent parameter values.
- If something required for simulation is missing, insert a clear TODO comment.
- Treat reviewed JSON as the primary source of truth.
- If a parameter has a reviewed formula, implement that formula exactly.
- Do not invent parameter values, formulas, transformations, or unit conversions.
- If a required value or formula is missing, leave a clear TODO instead of guessing.
- Prefer simulation_setup values only when they are explicitly provided by the user.

Code requirements:
1. Import required packages.
2. Define a parameter dictionary.
3. Define initial conditions.
4. Define the ODE right-hand-side function.
5. Define algebraic/effect/helper functions if present.
6. Run solve_ivp.
7. Plot all simulated state variables.
8. Save plots to outputs/simulation_plot.png.
9. Save simulation results to outputs/simulation_results.csv.
10. Make the script executable from terminal.

Solver configuration:
- Read solver method from environment variable SOLVER_METHOD.
- Read relative tolerance from RTOL.
- Read absolute tolerance from ATOL.
- Use defaults if environment variables are missing:
  method="LSODA", rtol=1e-6, atol=1e-9.

The generated script must save:
- outputs/simulation_plot.png
- outputs/simulation_results.csv

Return ONLY raw Python code.
Do not include explanations.
Do not include markdown fences.
Do not write "Here is the code".
The first line must be a valid Python import statement or comment.

Scientific rules:
- Clearly mark OCR-derived equations as requiring human review in comments.
- Clearly mark assumptions.
- If only part of the model is simulatable, generate code only for the simulatable subsystem.
- Do not pretend the generated model is validated.

Use reviewed human-entered values exactly. Do not treat them as guesses.
If a reviewed parameter value exists in reviewed JSON or simulation_setup, include it in params.
If a reviewed formula references another parameter, include that referenced parameter too.

Reviewed extraction:
{reviewed_extraction}

Simulation setup:
{simulation_setup}
"""

    result = llm.invoke(prompt)

    code = result.content.strip()

    # Extract code if model returned markdown fenced code
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0].strip()

    # Remove common preambles if no code fence was used
    code_start_markers = [
        "import ",
        "from ",
    ]

    for marker in code_start_markers:
        idx = code.find(marker)
        if idx != -1:
            code = code[idx:].strip()
            break

    return code

def save_generated_python_model(
    path: str,
    reviewed_extraction: str,
    simulation_setup: dict
):
    """
    Generate and save model-agnostic Python simulation code.
    """

    code = generate_python_model_from_extraction(
        reviewed_extraction=reviewed_extraction,
        simulation_setup=simulation_setup
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(code)

    return code


# Model discovery and review scaffold services.
def retrieve_parameter_discovery_context() -> str:
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

    return multi_query_retrieve(
        vector_store=VECTOR_STORE,
        queries=queries,
        label="Parameter Query",
        k=4
    )

def extract_parameters_from_context(parameter_context: str) -> str:
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
Copy explicit evidence. Do not infer, complete, normalize, or summarize beyond
what the context states. Prefer fewer high-confidence entries over broad lists.

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
- Do not infer parameter values, units, or roles from outside the quoted context.
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

def retrieve_equation_discovery_context() -> str:
    """
    Helper function: retrieve equation-specific context.
    Not exposed directly to the agent.
    """

    queries = [
        "Equation candidate record equation_number candidate text requires_review image_path",
        "displayed equation candidate rendered crop PDF text layer",
        "equation candidate record source PDF text layer method local parser candidate extraction",
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

    return multi_query_retrieve(
        vector_store=VECTOR_STORE,
        queries=queries,
        label="Equation Query",
        k=6
    )

def extract_equations_from_context(equation_context: str) -> str:
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
    Copy explicit evidence. Do not infer or reconstruct equations unless the
    source text already contains the required symbols clearly.

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
    - Distinguish between mathematical equations, parameter definitions, variable descriptions, mechanistic explanations, and assumptions.
    - Only classify content as an equation if it expresses a mathematical relationship between quantities.
    - Do not classify textual descriptions of symbols, units, mechanisms, or parameter-table entries as equations.
    - If a mathematical equation is referenced but visually omitted, partially missing, or OCR is incomplete, place it under inferred_or_missing_equations.
    - Distinguish explicitly reported equations from partially observed equations, inferred model structure, and missing information.

    Equation extraction rules:
    - Treat "Equation candidate record" and "Equation text-layer candidate"
      blocks as first-class candidate evidence.
    - Preserve their equation_number, source page, method, image_path when present,
      confidence, and requires_review status.
    - Extract equations exactly as written.
    - Preserve mathematical symbols.
    - Do not correct, simplify, normalize, or reinterpret equations.
    - Prefer fewer explicitly visible equations over broad reconstruction.
    - If an ODE is present, copy the full equation.
    - Do not summarize equations in words.
    - Prefer displayed equations over narrative descriptions.
    - Include compartment equations, Hill functions, Emax/EC50/IC50 relationships, algebraic couplings, balance equations, and delay/effect-compartment equations only when they are explicitly present.
    - If an OCR-derived equation contains unusual exponents such as Cb^Cb, Ca^Ca, Ce^Ce, or a parameter raised to itself, mark it as suspicious and requiring human review. Do not treat it as validated. OCR validation rule:
    - Treat Gemini OCR equations as candidate transcriptions, not validated equations.
    - Treat local parser/text-layer candidates as candidate transcriptions, not validated equations.
    - Do not silently correct OCR-derived equations.
    - If the OCR equation has suspicious exponents, denominator changes, or unclear brackets, mark it as requiring human review.

    Variable-name formatting rule:
    - If a biological/model term is written with hyphens, subscripts, or special characters, treat it as one variable name unless the context clearly shows mathematical subtraction.
    - Convert complex variable names to readable underscore notation only when needed.
    - Do not change the biological meaning.

    Equation cleanup rule:
    - When PDF extraction splits superscripts or subscripts, reconstruct them carefully using nearby text and symbols.
    - Preserve the intended equation, but do not invent new terms.
    - Do not add artificial divisions, missing powers, or missing denominators.

    Fallback rule:
    - If the context says Gemini OCR failed, parser omitted images, or displayed equations were not retrieved: do not invent equations.
    - Do not classify parameter descriptions as equations.
    - Place the affected equations under inferred_or_missing_equations.
    - Add a human_review_flags note.

    Context:
    {equation_context}
    """

    result = llm.invoke(prompt)
    return result.content

def extract_equations_with_optional_gemini_ocr(equation_context: str) -> str:
    """
    Broad equation OCR fallback for model discovery.

    Uses Gemini for broad PDF-level equation extraction.
    Keeps GPT/pix2tex for numbered equation Q&A only.
    """

    gemini_enabled = (
        os.getenv("ENABLE_DISCOVERY_GEMINI_OCR", "false").lower()
        in {"1", "true", "yes", "on"}
    )

    if not gemini_enabled:
        return (
            "Gemini broad equation OCR skipped: "
            "set ENABLE_DISCOVERY_GEMINI_OCR=true to enable model-discovery OCR."
        )

    gemini_equations = ""

    needs_ocr = any(
        trigger in equation_context.lower()
        for trigger in [
            "picture",
            "omitted",
            "image",
            "==>",
        ]
    )

    if needs_ocr and ACTIVE_PDF_PATH is not None:
        try:
            gemini_equations = extract_equations_with_gemini(
                ACTIVE_PDF_PATH
            )
        except Exception as error:
            gemini_equations = (
                f"Gemini broad equation OCR failed: "
                f"{type(error).__name__}: {error}"
            )

    return gemini_equations

def extract_mechanisms_from_context(model_context: str) -> str:
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
Copy explicit evidence. Do not infer pathway logic or biological causality.
Prefer fewer high-confidence model mechanisms over broad background summaries.

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
- Exclude mechanisms unless the retrieved context explicitly supports them.
- Do not invent mechanisms.

Context:
{model_context}
"""

    result = llm.invoke(prompt)
    return result.content

def sanitize_mechanism_summary_for_review(mechanisms: str) -> str:
    cleaned_lines = []

    for line in mechanisms.splitlines():
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append(line)
            continue

        if stripped.startswith("|"):
            continue

        if re.match(
            r"^-?\s*(source|relation|target|evidence)\s*[:|]",
            stripped,
            re.IGNORECASE,
        ):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    return cleaned or "No clean mechanism summary extracted."

def extract_text_after_marker(text: str, marker: str) -> str:
    if marker not in text:
        return text

    return text.split(marker, 1)[1].strip()

def build_review_source_summary(
    parameters: str,
    equations: str,
    mechanisms: str,
    edges: str,
) -> str:
    clean_equations = extract_text_after_marker(
        equations,
        "EXTRACTED EQUATIONS:"
    )
    missing_equation_recovery = ""

    if "MISSING NUMBERED EQUATION RECOVERY:" in equations:
        missing_equation_recovery = equations.split(
            "MISSING NUMBERED EQUATION RECOVERY:",
            1
        )[1].split(
            "==================================================",
            1
        )[0].strip()

    return f"""
PARAMETER EXTRACTION OUTPUT:
{parameters}

EQUATION EXTRACTION OUTPUT:
{clean_equations}

MISSING NUMBERED EQUATION RECOVERY:
{missing_equation_recovery}

MECHANISM EXTRACTION OUTPUT:
{sanitize_mechanism_summary_for_review(mechanisms)}

GRAPH EDGES OUTPUT:
{edges}
"""

def clean_human_review_scaffold(text: str) -> str:
    if "# Reviewed Model Draft" in text:
        text = text[text.find("# Reviewed Model Draft"):]

    debug_markers = [
        "RAW EQUATION RETRIEVAL CONTEXT",
        "GEMINI OCR EQUATION CONTEXT",
        "PARAMETER EXTRACTION OUTPUT:",
        "EQUATION EXTRACTION OUTPUT:",
        "MECHANISM EXTRACTION OUTPUT:",
        "GRAPH EDGES OUTPUT:",
        "GRAPH GENERATION OUTPUT:",
        "Equation Search Query",
        "Parameter Query",
        "Model-building Query",
        "Chunk ID:",
    ]

    if not any(marker in text for marker in debug_markers):
        return text.strip()

    cleaned_lines = []
    skip_block = False

    for line in text.splitlines():
        if line.startswith("# Reviewed Model Draft"):
            skip_block = False

        if any(marker in line for marker in debug_markers):
            skip_block = True
            continue

        if skip_block and line.startswith("## "):
            skip_block = False

        if not skip_block:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    if "# Reviewed Model Draft" in cleaned:
        return cleaned

    return """# Reviewed Model Draft

## Scope

summary:
Clean scaffold generation produced debug-like content. See outputs/extraction_debug.md for the full trace and rerun model discovery if needed.

components:
- not extracted

status:
needs review

---

## State Variables

No state variables extracted.

---

## Parameters

No parameters extracted.

---

## Equations

No equations extracted.

---

## Mechanisms

No mechanisms extracted.

---

## Missing / Needs Review
- Clean extraction review required regeneration from debug trace.
"""


def extract_model_parameters_from_pdf(vector_store) -> str:
    global VECTOR_STORE
    VECTOR_STORE = vector_store
    parameter_context = retrieve_parameter_discovery_context()
    return extract_parameters_from_context(parameter_context)


def extract_model_equations_from_pdf(
    vector_store,
    pdf_path: str | None,
    missing_equations_path: str = "outputs/missing_equations.json",
) -> str:
    global VECTOR_STORE, ACTIVE_PDF_PATH
    VECTOR_STORE = vector_store
    ACTIVE_PDF_PATH = pdf_path

    equation_context = retrieve_equation_discovery_context()
    text_layer_equation_candidates = extract_equation_candidates_from_text_layer(pdf_path)

    needs_gemini = any(
        trigger in equation_context.lower()
        for trigger in [
            "picture",
            "omitted",
            "image",
            "==>",
        ]
    )

    print("Broad equation OCR fallback checked.")

    if needs_gemini:
        gemini_equations = (
            "Gemini/OCR equation extraction skipped during model discovery. "
            "Use Review & Validate Model for targeted equation OCR."
        )
    else:
        gemini_equations = (
            "Gemini broad equation OCR not triggered: no parser omission "
            "or image marker found in retrieved equation context."
        )

    combined_equation_context = f"""
TEXT RETRIEVAL EQUATION CONTEXT:
{equation_context}

PDF TEXT-LAYER EQUATION CANDIDATES:
{text_layer_equation_candidates}

GEMINI OCR EQUATION CONTEXT:
{gemini_equations}
"""

    extracted_equations = extract_equations_from_context(combined_equation_context)

    missing_equation_recovery = recover_missing_numbered_equations(
        extracted_equations=extracted_equations,
        equation_context=equation_context,
        pdf_path=pdf_path,
        missing_equations_path=missing_equations_path,
    )

    return f"""
RAW EQUATION RETRIEVAL CONTEXT:
{equation_context}

PDF TEXT-LAYER EQUATION CANDIDATES:
{text_layer_equation_candidates}

GEMINI OCR EQUATION CONTEXT:
{gemini_equations}

MISSING NUMBERED EQUATION RECOVERY:
{missing_equation_recovery}

==================================================

EXTRACTED EQUATIONS:
{extracted_equations}
"""


def extract_model_mechanisms_from_context(model_context: str) -> str:
    return extract_mechanisms_from_context(model_context)


def compile_reviewed_model_scaffold(extracted_summary: str) -> str:
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
Keep the scaffold conservative and evidence-preserving. Prefer copying source
extraction text over broad summarization.

SOURCE-OF-TRUTH RULES
- Use PARAMETER EXTRACTION OUTPUT as the only source for parameters.
- Use EQUATION EXTRACTION OUTPUT as the only source for equations.
- Use MECHANISM EXTRACTION OUTPUT as the only source for mechanisms.
- Use GRAPH EDGES OUTPUT as the only source for graph edges.
- Do not use prior pharmacology or biology knowledge.
- Do not invent, infer, simplify, derive, or correct equations.
- Do not add mechanisms, parameters, states, or assumptions that are not already
  present in the extracted outputs.
- Do not independently decide what is missing.
- Do not classify state variables or observed quantities as missing parameters.
- Preserve all reported values, units, symbols, and source labels exactly.

EQUATION PRESERVATION RULES
- If EQUATION EXTRACTION OUTPUT contains reported_odes, copy them exactly into section 4.
- If EQUATION EXTRACTION OUTPUT contains reported_algebraic_or_coupling_equations, copy them exactly into section 4.
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
- If an edge conflicts with mechanisms or equations, move it to section 6 as requiring review.
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
- Do not use Markdown tables.
- Do not use LaTeX display blocks.
- Put equations in inline code backticks.
- Keep each item compact and editable.
- Use YAML-like key/value fields.
- Prefer short values over long prose.

REQUIRED OUTPUT SECTIONS
Use this exact clean YAML-like model-card format.
Do NOT use Markdown tables anywhere in the output.
Do NOT use LaTeX display blocks.
Do NOT include raw extraction logs.

# Reviewed Model Draft

## Scope

summary:
<one short evidence-based sentence>

components:
- <component or process>
- <component or process>

status:
needs review

---

## State Variables

[<symbol_or_name>]
meaning: <meaning copied from extraction>
unit: <unit or unknown>
source: <page/table/source or not reported>
status: needs review

---

Repeat this compact block for each state variable or model quantity. If none
were extracted, write:

No state variables or model quantities extracted.

---

## Parameters

[<symbol_or_name>]
value: <value or not reported>
unit: <unit or unknown>
meaning: <meaning copied from extraction>
source: <page/table/source or not reported>
status: needs review

---

For derived parameters, use:

[<symbol_or_name>]
equation: <derivation copied from extraction>
unit: <unit or unknown>
meaning: <meaning copied from extraction>
source: <page/table/source or not reported>
status: derived / needs review

Repeat this compact block for reported, derived, and missing parameters. If no
parameters were extracted, write:

No parameters extracted.

---

## Equations

(eq_<number_or_label>) <short evidence-based title>
status: needs review / OCR candidate / missing
source: <page/table/source or not reported>

equation:
`<copy equation exactly as extracted>`

initial_condition:
`<initial condition if explicitly extracted, otherwise not reported>`

equivalent_form:
`<equivalent form if explicitly extracted, otherwise not reported>`

method_source:
<PDF text layer / OCR / extraction / not reported>

candidate_crop:
<image path if present, otherwise not available>

notes:
needs human validation

---

Repeat this compact block for every extracted, candidate, OCR-derived, or
missing equation. If no equations were extracted, write:

No equations extracted.

---

## Mechanisms

[mechanism_<number>]
source_entity: <source entity>
relation: <relation>
target_entity: <target entity>
source: <page/source or not reported>
evidence: <short copied evidence>
status: needs review

---

Repeat this compact block for each supported mechanism or graph edge. If no
mechanisms were extracted, write:

No mechanisms extracted.

---

## Missing / Needs Review

- <missing value, ambiguous equation, OCR warning, unsupported edge, or assumption>

IMPORTANT
- This is a draft model specification based only on extracted evidence.
- Keep it as a model card, not an extraction log.
- Do not claim the model is validated.
- Do not use Markdown tables.
- Do not use LaTeX display blocks.
- Put equations in inline code backticks only.
- Keep raw retrieval context, chunk metadata, graph JSON, internal logs, and
  debug OCR sections out of the review file.
- Preserve OCR/candidate metadata when present:
  equation_number, source page, method_source, candidate_crop, confidence,
  requires_review.

EXTRACTED INFORMATION:
{extracted_summary}
"""


    result = llm.invoke(prompt)
    return result.content

def run_model_discovery_pipeline(
    vector_store,
    pdf_path: str | None = None,
    missing_equations_path: str = "outputs/missing_equations.json",
) -> str:
    """
    Cheap model discovery stage.

    This stage builds only a candidate map of equations, tables, and figures.
    It does not run GPT/Gemini OCR, graph generation, mechanism extraction,
    or scaffold compilation.

    Precision work is deferred to Review & Validate.
    """

    if pdf_path is None:
        return "No active PDF path available for artifact discovery."

    artifact_index = save_artifact_index(
        pdf_path=pdf_path,
        output_path="outputs/artifact_index.json",
    )

    artifact_summary = format_artifact_index_for_review(
        artifact_index
    )

    result = f"""
# Lit2Model-AI Discovery Candidate Map

This is a cheap deterministic discovery step.

It does not claim to produce a validated mechanistic model.
It only maps candidate equations, tables, and figures from the PDF text layer.

---

{artifact_summary}

---

## Next Step

Use Review & Validate to inspect selected candidates.

Recommended workflow:
1. Review important equation candidates.
2. Run targeted OCR only for weak or visually omitted equations.
3. Review parameter tables.
4. Run table OCR only if text-layer extraction fails.
5. Review important figures with vision only when needed.
6. Save accepted items into the reviewed model draft.
7. Generate the Python model only from reviewed evidence.
"""

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/extraction_debug.md", "w", encoding="utf-8") as file:
        file.write(result)

    with open("outputs/extraction_review.md", "w", encoding="utf-8") as file:
        file.write(result)

    return result