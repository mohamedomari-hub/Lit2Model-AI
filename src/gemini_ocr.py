import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


def extract_equations_with_gemini(
    pdf_path: str,
    equation_number: str = None):

    """
    Use Gemini as an optional fallback to extract equations from a PDF.

    This is intended for cases where PyMuPDF4LLM returns:
    'picture intentionally omitted'
    or equation text is incomplete.
    """

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return (
            "Gemini OCR skipped: GEMINI_API_KEY or GOOGLE_API_KEY "
            "was not found in environment."
        )

    client = genai.Client(api_key=api_key)

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        return f"Gemini OCR skipped: PDF not found at {pdf_path}"

    prompt = f"""
You are an OCR transcription system.

Your ONLY task is to visually transcribe Equation ({equation_number})
from the PDF exactly as written.

IMPORTANT:
You are NOT allowed to interpret, simplify,
reconstruct, or rewrite equations.

You must behave like a scientific OCR engine.

Rules:
- Find ONLY Equation ({equation_number}).
- Copy symbols exactly as shown.
- Preserve:
  - 1+ or 1-
  - parentheses
  - fractions
  - superscripts
  - subscripts
  - multiplication symbols
  - numerator/denominator structure
- Do NOT convert equations into standard Hill forms.
- Do NOT infer missing mathematics.
- Do NOT rewrite equations using your own understanding.
- If part of the equation is unclear, write:
  [UNCLEAR]
- If uncertain, say:
  OCR uncertain – requires human review.

Return ONLY:

Equation ({equation_number}):

<exact transcription>

OCR confidence:
high / medium / low
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

    This is used for plots, diagrams, mechanism figures, and model structure figures.
    """

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return (
            "Gemini figure vision skipped: GEMINI_API_KEY or GOOGLE_API_KEY "
            "was not found in environment."
        )

    client = genai.Client(api_key=api_key)

    image_file = Path(image_path)

    if not image_file.exists():
        return f"Gemini figure vision skipped: image not found at {image_path}"

    prompt = """
You are a scientific figure interpretation assistant for mechanistic modelling papers.

Describe this figure carefully.

Focus on:
- what type of figure it is: plot, model diagram, mechanism diagram, table-like image, network, compartment model
- visible variables, labels, axes, units, legends, and panel labels
- curves, trends, peaks, delays, increases/decreases, or comparisons if clearly visible
- compartments, arrows, feedbacks, stimulations, inhibitions, or flows if shown
- model-relevant mechanisms, equations, parameters, or state variables
- timing information such as hours, days, treatment times, or dosing events

Rules:
- Do not invent information.
- If a curve, arrow, label, or relationship is unclear, mark it as uncertain.
- Separate direct visual observations from interpretation.
- If the image quality is poor, say so clearly.
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