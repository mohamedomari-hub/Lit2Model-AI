"""
Classifies scientific PDF text blocks and visual candidates.
"""

from langchain_openai import ChatOpenAI


def classify_equation_candidate(candidate_text: str) -> dict:
    """
    Tiny LLM classifier for ambiguous mechanistic-model candidates.

    Called ONLY for medium-confidence candidates flagged by
    llm_review_recommended=True.

    Low-token design:
    - single candidate only
    - structured JSON output
    - deterministic temperature
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = f"""
You are a scientific mechanistic-model classifier.

Classify this candidate into ONE category only.

Allowed categories:
- state_equation
- coupling_equation
- regulatory_function
- derived_definition
- parameter_value_inline
- observation_or_context
- noise

Rules:
- State equation:
  derivative/dynamic equation.

- Coupling equation:
  algebraic mechanistic dependency.

- Regulatory function:
  Hill/Emax/EC50/effect function.

- Derived definition:
  symbolic mathematical definition
  (example: CL = ke * Vd)

- Parameter value:
  numerical quantity assignment.

- Observation/context:
  biological or narrative statement,
  not part of model structure.

- Noise:
  OCR damage or unusable fragment.

Return STRICT JSON ONLY:

{{
  "category": "...",
  "confidence": "high|medium|low",
  "reason": "short explanation"
}}

Candidate:
{candidate_text}
"""

    result = llm.invoke(prompt)

    content = result.content.strip()

    try:
        import json
        return json.loads(content)
    except Exception:
        return {
            "category": "noise",
            "confidence": "low",
            "reason": "LLM parsing failed",
        }