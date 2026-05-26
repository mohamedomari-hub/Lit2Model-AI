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
