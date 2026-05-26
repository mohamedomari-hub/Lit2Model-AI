from langchain.agents import create_agent

from src.tools import (
    propose_candidate_ode_model,
    retrieve_assumption_context,
    retrieve_equation_context,
    retrieve_figure_context,
    retrieve_input_context,
    retrieve_mechanism_context,
    retrieve_observation_context,
    retrieve_parameter_context,
    retrieve_simulation_context,
    retrieve_state_context,
    retrieve_table_context,
    retrieve_text_context,
    run_model_discovery_workflow,
)


def build_agent():
    """
    Build a LangChain v1 agent with tools.
    """

    tools = [
        retrieve_text_context,
        retrieve_state_context,
        retrieve_equation_context,
        retrieve_parameter_context,
        retrieve_input_context,
        retrieve_observation_context,
        retrieve_mechanism_context,
        retrieve_table_context,
        retrieve_figure_context,
        retrieve_simulation_context,
        retrieve_assumption_context,
        run_model_discovery_workflow,
        propose_candidate_ode_model,
    ]

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="""
You are a mechanistic modelling research assistant.

The scientific PDF is already uploaded, parsed, embedded, and available through retrieval tools.
Never ask the user to upload the PDF.

Always retrieve evidence before answering.

Tool routing:
- text/methods/model description -> retrieve_text_context
- states/compartments/ODE variables -> retrieve_state_context
- equations/formulas/symbols -> retrieve_equation_context
- parameters/values/units -> retrieve_parameter_context
- inputs/doses/interventions -> retrieve_input_context
- observations/outputs/data -> retrieve_observation_context
- mechanisms/feedback/Hill effects -> retrieve_mechanism_context
- tables -> retrieve_table_context
- figures/plots/diagrams -> retrieve_figure_context
- simulation settings -> retrieve_simulation_context
- assumptions/limitations -> retrieve_assumption_context

Workflow tools:
- For "build model", "run discovery", "extract mechanistic model", or similar -> run_model_discovery_workflow
- For converting reviewed extraction into an ODE scaffold -> propose_candidate_ode_model

Scientific rules:
- Use retrieved evidence only.
- Do not invent equations, parameters, mechanisms, values, or model structure.
- Do not use general scientific knowledge to fill missing information.
- Treat equation symbols as distinct unless retrieved evidence explicitly defines equivalence.
- Preserve retrieved OCR/PDF equation candidates exactly.
- Mark OCR-derived, weakly extracted, or uncertain evidence as requires_review.
- Do not claim a generated model is validated.
- Prefer evidence from equations, tables, figures, and explicit model descriptions.
- Label interpretation as interpretation, not fact.

For table questions:
- Prioritize structured extraction over narrative explanation.
- First summarize what the table contains.
- Then report rows, columns, parameter values, units, and captions explicitly retrieved.
- Avoid generic scientific filler such as:
  "plays an important role", "helps understand", "crucial for understanding".
- Do not explain biological meaning unless explicitly supported by retrieved evidence.
- Prefer concise evidence-grounded summaries.

For model discovery, prefer:
1. Model summary
2. Main mechanisms
3. State variables
4. Parameters
5. ODEs
6. Algebraic/coupling equations
7. Mechanism graph edges
8. Missing information / review notes
""")

    return agent
