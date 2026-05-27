def check_model_consistency(reviewed_extraction: str) -> dict:
    """
    Run lightweight consistency checks on a reviewed model extraction.

    This deterministic placeholder intentionally reports review categories
    without pretending to validate the science. More specific checks can be
    added as the reviewed model card becomes more structured.
    """

    text = reviewed_extraction or ""
    lower_text = text.lower()
    issues = []

    expected_sections = {
        "states": ["state", "compartment"],
        "parameters": ["parameter"],
        "equations": ["equation", "ode", "d/dt"],
        "inputs": ["dose", "input", "intervention"],
        "observations": ["observation", "biomarker", "measured"],
        "simulation": ["solver", "simulation", "time grid", "initial condition"],
    }

    for section, markers in expected_sections.items():
        if not any(marker in lower_text for marker in markers):
            issues.append(
                {
                    "check": f"missing_{section}_evidence",
                    "message": f"No obvious {section} evidence found in reviewed extraction.",
                    "requires_review": True,
                }
            )

    if "ocr" in lower_text:
        issues.append(
            {
                "check": "ocr_derived_content",
                "message": "OCR-derived content is present and should remain review-flagged.",
                "requires_review": True,
            }
        )

    return {
        "status": "review" if issues else "pass",
        "issues": issues,
    }
