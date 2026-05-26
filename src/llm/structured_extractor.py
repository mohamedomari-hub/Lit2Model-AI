from __future__ import annotations

import json
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.schemas.evidence_schema import ChunkEvidence

load_dotenv()


EXTRACTION_SYSTEM_PROMPT = """
You are an expert scientific model extraction assistant.

Your task:
Extract mechanistic-model evidence from ONE small scientific paper chunk.

This must generalize across:
- QSP
- PK/PD
- PBPK
- systems biology
- SIR/SEIR epidemiology
- ecology
- compartmental ODE models
- biochemical networks
- engineering dynamical systems

Use only the chunk evidence.
Do not invent missing equations, values, parameters, or mechanisms.
If something is incomplete, mark it as uncertain or needs_ocr.

Return ONLY valid JSON matching this structure:

{
  "equations": [
    {
      "equation_id": "Eq(1) or null",
      "equation_type": "state_equation | coupling_equation | regulatory_function | derived_definition | unknown",
      "raw_text": "exact equation or candidate text",
      "variables": ["optional symbols"],
      "page": 1,
      "confidence": "high | medium | low",
      "needs_ocr": false,
      "review_note": ""
    }
  ],
  "parameters": [
    {
      "symbol": "ka",
      "name": "absorption rate constant",
      "value": "13.4",
      "unit": "1/d",
      "meaning": "...",
      "source": "table/text/equation",
      "page": 1,
      "confidence": "high | medium | low",
      "review_note": ""
    }
  ],
  "state_variables": [
    {
      "symbol": "C",
      "name": "central compartment concentration",
      "meaning": "...",
      "unit": "ng/ml",
      "page": 1,
      "confidence": "high | medium | low"
    }
  ],
  "mechanisms": [
    {
      "source_entity": "Drug",
      "relation": "stimulates/inhibits/transfers/produces/removes/etc.",
      "target_entity": "Response",
      "evidence_text": "exact supporting text",
      "page": 1,
      "confidence": "high | medium | low",
      "review_note": ""
    }
  ],
  "tables": [
    {
      "table_id": "Table 1",
      "caption_or_title": "...",
      "purpose": "parameters/data/results/etc.",
      "page": 1,
      "likely_contains_parameters": true,
      "needs_table_extraction": true,
      "confidence": "high | medium | low"
    }
  ],
  "figures": [
    {
      "figure_id": "Figure 1",
      "caption_or_title": "...",
      "purpose": "mechanism diagram/model structure/results/etc.",
      "page": 1,
      "likely_contains_mechanism_graph": true,
      "needs_vision": true,
      "confidence": "high | medium | low"
    }
  ],
  "observations": [
    {
      "observed_quantity": "...",
      "description": "...",
      "page": 1,
      "confidence": "high | medium | low"
    }
  ],
  "missing_or_uncertain": [
    "..."
  ]
}

Important classification rules:

1. state_equation
ONLY if the expression explicitly represents a dynamic time evolution:
examples:
- dX/dt
- dx/dt
- ∂X/∂t
- X'(t)
- time derivative notation
- differential form

These represent evolving system states.

2. coupling_equation
Any mechanistic algebraic dependency WITHOUT a time derivative.

Examples:
- transfer relationships
- mechanistic modifiers
- transformed variables
- algebraic dependencies between variables
- dose-response transformations
- scaling relationships

Examples:
A = B * effect
Y = X / (X + K)
Q = alpha * X

These are NOT state equations.

3. regulatory_function
Mechanistic control functions including:

- activation
- inhibition
- stimulation
- repression
- Hill functions
- Emax models
- sigmoid functions
- Michaelis-Menten kinetics
- saturating functions
- threshold functions
- feedback control

Usually modifies another process.

Examples:
1 + Emax * X/(X+K)
X^n / (X^n + K^n)
Vmax * X / (Km + X)

4. derived_definition
Symbolic mathematical definitions.

Examples:
CL = ke * Vd
R0 = beta / gamma
k = ln(2)/t_half

5. observation_or_context
Narrative explanation, interpretation, biological discussion, results text.

6. noise
Broken OCR, incomplete fragments, unusable text.

Critical rule:
If NO time derivative exists,
DO NOT classify as state_equation.
Prefer coupling_equation or regulatory_function.
"""

def _extract_json_object(text: str) -> dict:
    """
    Robust JSON extraction from LLM output.

    Handles:
    - markdown fences
    - trailing commas
    - malformed quotes
    - retry repair using cheap LLM
    """

    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found.")

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)

    except json.JSONDecodeError:

        repair_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        repair_prompt = f"""
You repair malformed JSON.

Return ONLY valid JSON.

Fix syntax only.
Do not invent or remove scientific content.

Malformed JSON:

{candidate}
"""

        repaired = repair_llm.invoke(
            repair_prompt
        ).content.strip()

        repaired = re.sub(
            r"^```json\s*",
            "",
            repaired,
        )

        repaired = re.sub(
            r"\s*```$",
            "",
            repaired,
        )

        return json.loads(repaired)

def extract_chunk_evidence(
    chunk: dict,
    model: str = "gpt-4o-mini",
) -> ChunkEvidence:
    """
    Extract structured evidence from one scientific chunk.
    """

    llm = ChatOpenAI(
        model=model,
        temperature=0,
    )

    prompt = f"""
{EXTRACTION_SYSTEM_PROMPT}

Chunk metadata:
chunk_id: {chunk.get("chunk_id")}
page_start: {chunk.get("page_start")}
page_end: {chunk.get("page_end")}

Chunk text:
{chunk.get("text", "")}
"""

    response = llm.invoke(prompt)

    parsed = _extract_json_object(response.content)

    parsed["chunk_id"] = chunk.get("chunk_id")
    parsed["page_start"] = chunk.get("page_start")
    parsed["page_end"] = chunk.get("page_end")

    return ChunkEvidence.model_validate(parsed)