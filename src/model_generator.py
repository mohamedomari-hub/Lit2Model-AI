import os
import json
from langchain_openai import ChatOpenAI



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