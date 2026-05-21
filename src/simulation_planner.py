import json
from langchain_openai import ChatOpenAI


def infer_simulation_requirements(reviewed_extraction: str) -> dict:
    """
    Infer what is needed to simulate the extracted model.
    Model-agnostic: works for PK/PD, SIR, QSP, ODE models, etc.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a mechanistic modelling simulation planner.

Given a reviewed model extraction, infer the simulation requirements.

Return ONLY valid JSON with this schema:

{{
  "model_type": "",
  "states": [
    {{
      "name": "",
      "description": "",
      "initial_value_required": true,
      "suggested_default": null,
      "unit": null,
      "requires_human_input": true
    }}
  ],
  "parameters": [
    {{
      "name": "",
      "description": "",
      "value": null,
      "unit": null,
      "source": "reported | derived | missing | user_required",
      "requires_human_input": true
    }}
  ],
  "inputs": [
    {{
      "name": "",
      "description": "",
      "value_required": true,
      "suggested_default": null,
      "unit": null
    }}
  ],
  "time_settings": {{
    "time_variable": "t",
    "start": 0,
    "end": null,
    "unit": null,
    "requires_human_input": true
  }},
  "equations_ready_for_simulation": true,
  "missing_for_simulation": [],
  "human_review_notes": []
}}

Rules:
- Use only the reviewed extraction.
- Do not invent biological mechanisms.
- If equations are incomplete or OCR-derived, mention this in human_review_notes.
- If initial conditions are not stated, put them in states with requires_human_input=true.
- If a parameter has a reported value, include it.
- If a parameter is missing, mark source as missing or user_required.
- Be model-agnostic.

REVIEWED EXTRACTION:
{reviewed_extraction}
"""

    result = llm.invoke(prompt)

    try:
        return json.loads(result.content)
    except json.JSONDecodeError:
        return {
            "model_type": "unknown",
            "states": [],
            "parameters": [],
            "inputs": [],
            "time_settings": {
                "time_variable": "t",
                "start": 0,
                "end": None,
                "unit": None,
                "requires_human_input": True
            },
            "equations_ready_for_simulation": False,
            "missing_for_simulation": [
                "Could not parse simulation requirements JSON."
            ],
            "human_review_notes": [
                result.content
            ]
        }