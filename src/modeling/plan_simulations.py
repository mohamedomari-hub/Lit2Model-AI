"""
Infers simulation setup requirements from reviewed model evidence.
"""

import json
from typing import Any


def _clean(value: Any):
    if value in [None, "", "missing", "not reported"]:
        return None
    return value


def infer_simulation_requirements_from_json(reviewed_json: dict) -> dict:
    """
    Deterministic simulation planner from reviewed JSON.
    No LLM. No hallucination.
    """

    states = []
    parameters = []
    inputs = []
    missing_for_simulation = []
    human_review_notes = []

    for ode in reviewed_json.get("odes", []):
        state_name = ode.get("state")

        if state_name:
            initial_condition = ode.get("initial_condition", {}) or {}

            states.append(
                {
                    "name": state_name,
                    "description": ode.get("meaning"),
                    "initial_value_required": True,
                    "suggested_default": _clean(initial_condition.get("value")),
                    "unit": ode.get("unit"),
                    "requires_human_input": _clean(initial_condition.get("value")) is None,
                }
            )

        for param in ode.get("parameters", []):
            name = param.get("symbol")
            value = _clean(param.get("value"))
            status = param.get("status", "missing")

            if name:
                parameters.append(
                    {
                        "name": name,
                        "description": param.get("meaning") or param.get("formula"),
                        "value": value,
                        "unit": param.get("unit"),
                        "source": status,
                        "requires_human_input": value is None or status == "missing",
                    }
                )

                if value is None or status == "missing":
                    missing_for_simulation.append(f"Missing parameter: {name}")

        for inp in ode.get("inputs", []):
            name = inp.get("symbol")

            if name:
                value = _clean(inp.get("value"))

                seen_inputs = {}

                for ode in reviewed_json.get("odes", []):
                    for inp in ode.get("inputs", []):
                        name = inp.get("symbol")

                        if not name:
                            continue

                        value = _clean(inp.get("value"))
                        unit = inp.get("unit")
                        meaning = inp.get("meaning")

                        key = name.strip().lower()

                        candidate = {
                            "name": name,
                            "description": meaning,
                            "value_required": value is None,
                            "suggested_default": value,
                            "unit": unit,
                        }

                        # Prefer the more informative input if duplicates exist
                        if key not in seen_inputs:
                            seen_inputs[key] = candidate
                        else:
                            old_text = str(seen_inputs[key])
                            new_text = str(candidate)

                            if len(new_text) > len(old_text):
                                seen_inputs[key] = candidate

                inputs = list(seen_inputs.values())

    for module in reviewed_json.get("process_modules", []):
        for param in module.get("parameters", []):
            name = param.get("symbol")
            value = _clean(param.get("value"))
            status = param.get("status", "missing")

            if name:
                parameters.append(
                    {
                        "name": name,
                        "description": param.get("meaning") or param.get("formula"),
                        "value": value,
                        "unit": param.get("unit"),
                        "source": status,
                        "requires_human_input": value is None or status == "missing",
                    }
                )

                if value is None or status == "missing":
                    missing_for_simulation.append(f"Missing parameter: {name}")

        for eq in module.get("equations", []):
            eq_text = str(eq).lower()

            if "ocr" in eq_text or "requires_review" in eq_text:
                human_review_notes.append(
                    f"Equation requires human review: {eq}"
                )

    equations_ready = len(missing_for_simulation) == 0

    return {
        "model_type": "ODE / mechanistic model",
        "states": states,
        "parameters": parameters,
        "inputs": inputs,
        "time_settings": {
            "time_variable": "t",
            "start": 0,
            "end": None,
            "unit": None,
            "requires_human_input": True,
        },
        "equations_ready_for_simulation": equations_ready,
        "missing_for_simulation": sorted(set(missing_for_simulation)),
        "human_review_notes": human_review_notes,
    }


def infer_simulation_requirements(reviewed_extraction: str | dict) -> dict:
    """
    Backward-compatible wrapper.
    Prefer reviewed JSON.
    """

    if isinstance(reviewed_extraction, dict):
        return infer_simulation_requirements_from_json(reviewed_extraction)

    try:
        parsed = json.loads(reviewed_extraction)
        if isinstance(parsed, dict):
            return infer_simulation_requirements_from_json(parsed)
    except Exception:
        pass

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
            "requires_human_input": True,
        },
        "equations_ready_for_simulation": False,
        "missing_for_simulation": [
            "Simulation planner now expects reviewed JSON."
        ],
        "human_review_notes": [
            "Use reviewed_model_draft.json or final_reviewed_model.json."
        ],
    }