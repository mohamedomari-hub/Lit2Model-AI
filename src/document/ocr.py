import base64
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

_latex_ocr_model = None


def get_latex_ocr_model():
    global _latex_ocr_model

    if _latex_ocr_model is None:
        from pix2tex.cli import LatexOCR

        _latex_ocr_model = LatexOCR()

    return _latex_ocr_model


def extract_equation_with_pix2tex(image_path: str) -> str:
    """
    Extract LaTeX from an equation image using local pix2tex / LaTeX-OCR.
    """

    from PIL import Image

    model = get_latex_ocr_model()
    image = Image.open(image_path)
    return model(image)


def extract_equation_with_gpt(
    image_path: str,
    equation_number: str,
    model: str = "gpt-4o-mini",
):
    """
    OCR a specific equation from an image using GPT vision.
    """

    from openai import OpenAI

    from langsmith.wrappers import wrap_openai

    client = wrap_openai(OpenAI())

    with open(image_path, "rb") as file:
        image_base64 = base64.b64encode(file.read()).decode("utf-8")

    prompt = f"""
    You are a scientific OCR transcription engine.

    Treat this as forensic transcription,
    NOT interpretation.

    Your ONLY task is to visually transcribe
    Equation ({equation_number})
    EXACTLY as written.

    Rules:
    - Do not interpret.
    - Do not simplify.
    - Do not rewrite.
    - Do not infer.
    - Do not correct equations.
    - Never normalize notation.
    - Preserve equation numbering.
    - Preserve signs.
    - Preserve parentheses.
    - Preserve fractions exactly.
    - Preserve superscripts.
    - Preserve subscripts.
    - Preserve multiplication symbols.
    - Preserve numerator/denominator structure.
    - Pay special attention to small exponents.
    - If something is unclear,
    write [UNCLEAR] and do not guess.

    Output format:

    Equation ({equation_number}):

    <exact transcription>

    OCR confidence:
    high / medium / low
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def extract_visible_equations_with_gpt(
    image_path: str,
    model: str = "gpt-4o",
):
    """
    OCR all numbered equations visible in a crop using GPT vision.
    """

    from openai import OpenAI

    client = OpenAI()

    with open(image_path, "rb") as file:
        image_base64 = base64.b64encode(file.read()).decode("utf-8")

    prompt = """
You are a scientific OCR transcription engine.

Your ONLY task is to visually transcribe every numbered equation that is
clearly visible in this image crop.

Do not interpret, simplify, derive, or correct equations. Behave like a
literal OCR system.

Rules:
- Include only equations visibly present in the image.
- Preserve equation numbers, subscripts, superscripts, fractions, parentheses,
  signs, and numerator/denominator structure.
- If a symbol or exponent is unclear, write [UNCLEAR] instead of guessing.
- Do not invent missing equations.

Output format:

Equation (<number>):
<exact transcription>

OCR confidence:
high / medium / low

Repeat that block for each visible numbered equation.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def extract_equations_with_gemini(
    pdf_path: str,
    equation_number: str | None = None,
):
    """
    Use Gemini as an optional PDF OCR fallback for equations.
    """

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return (
            "Gemini OCR skipped: GEMINI_API_KEY or GOOGLE_API_KEY "
            "was not found in environment."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        return (
            "Gemini OCR skipped: google-genai is not installed "
            f"({type(error).__name__}: {error})."
        )

    client = genai.Client(api_key=api_key)
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        return f"Gemini OCR skipped: PDF not found at {pdf_path}"

    if equation_number is None:
        target_instruction = (
            "Find displayed mathematical equations in the PDF that are relevant "
            "to the mechanistic model. Preserve equation numbers when visible."
        )
        output_instruction = """
Return Markdown with one section per candidate:

Equation candidate:
- Equation number:
- PDF page if visible:
- Exact transcription:
- OCR confidence: high / medium / low
- Review note:
"""
    else:
        target_instruction = (
            f"Find ONLY Equation ({equation_number}) from the PDF exactly as written."
        )
        output_instruction = f"""
Return ONLY:

Equation ({equation_number}):

<exact transcription>

OCR confidence:
high / medium / low
"""

    prompt = f"""
You are an OCR transcription system.

Your ONLY task is to visually transcribe equations from the PDF exactly as written.

{target_instruction}

Rules:
- Do not interpret, simplify, reconstruct, or rewrite equations.
- Preserve signs, parentheses, fractions, superscripts, subscripts,
  multiplication symbols, and numerator/denominator structure.
- If part of the equation is unclear, write [UNCLEAR].
- If uncertain, say: OCR uncertain - requires human review.

{output_instruction}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_file.read_bytes(),
                    mime_type="application/pdf",
                ),
                prompt,
            ],
        )
        return response.text
    except Exception as error:
        return f"""
Gemini OCR fallback failed.

Reason:
{type(error).__name__}: {error}

Action:
Continue using text-retrieved equation context only.
Mark equations as requiring human review if incomplete.
"""


def describe_figure_with_gemini(image_path: str) -> str:
    """
    Use Gemini vision to describe a scientific figure.
    """

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return (
            "Gemini figure vision skipped: GEMINI_API_KEY or GOOGLE_API_KEY "
            "was not found in environment."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        return (
            "Gemini figure vision skipped: google-genai is not installed "
            f"({type(error).__name__}: {error})."
        )

    client = genai.Client(api_key=api_key)
    image_file = Path(image_path)

    if not image_file.exists():
        return f"Gemini figure vision skipped: image not found at {image_path}"

    prompt = """
You are a scientific figure interpretation assistant for mechanistic modelling papers.

Describe this figure carefully.

Focus on visible variables, labels, axes, units, legends, panel labels, curves,
trends, timing, compartments, arrows, feedbacks, stimulations, inhibitions, and
model-relevant mechanisms.

Rules:
- Do not invent information.
- Mark unclear visual evidence as uncertain.
- Separate direct visual observations from interpretation.
- Return concise Markdown.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_file.read_bytes(),
                    mime_type="image/png",
                ),
                prompt,
            ],
        )
        return response.text
    except Exception as error:
        return f"""
Gemini figure vision failed.

Reason:
{type(error).__name__}: {error}

Action:
Continue using parser/OCR-derived figure text only.
Mark figure interpretation as requiring human review.
"""


def extract_with_gpt_ocr(
    image_path: str,
    target_type: str = "equation",
    mode: str = "targeted",
    target_label: str | None = None,
    model: str = "gpt-4o",
) -> str:
    """
    Unified GPT OCR entry point for visual scientific objects.
    """

    if target_type == "equation" and mode == "all_visible":
        return extract_visible_equations_with_gpt(
            image_path=image_path,
            model=model,
        )

    if target_type == "equation":
        return extract_equation_with_gpt(
            image_path=image_path,
            equation_number=target_label or "unknown",
            model=model,
        )

    raise NotImplementedError(
        f"GPT OCR for target_type={target_type!r} is not wired yet."
    )
