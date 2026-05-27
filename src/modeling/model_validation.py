"""
Equation parsing and lightweight consistency checks for reviewed models.
"""

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
import re
from sympy import sympify




def extract_equations(reviewed_extraction: str):
    """
    Extract LaTeX equations between $$...$$ blocks.
    """

    equations = re.findall(
        r"\$\$(.*?)\$\$",
        reviewed_extraction,
        re.DOTALL
    )

    equations = [
        eq.strip()
        for eq in equations
    ]

    return equations


def latex_to_sympy(eq_text: str):
    """
    Convert simple equation text into SymPy expression.

    Example:
    dCe/dt = keo*(C-Ce)
    """

    try:
        left, right = eq_text.split("=")

        expr = sympify(right)

        return {
            "lhs": left.strip(),
            "rhs": expr,
            "valid": True
        }

    except Exception as e:

        return {
            "equation": eq_text,
            "valid": False,
            "error": str(e)
        }


def parse_equations(reviewed_extraction: str):
    """
    Parse all extracted equations.
    """

    equations = extract_equations(
        reviewed_extraction
    )

    parsed = []

    for eq in equations:
        parsed.append(
            latex_to_sympy(eq)
        )

    return parsed
