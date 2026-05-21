import os
import re
import fitz

from langchain_core.documents import Document
from src.gemini_ocr import describe_figure_with_gemini

# --------------------------------------------------
# Cost control
# --------------------------------------------------

USE_VISION = False


# --------------------------------------------------
# Metadata helpers
# --------------------------------------------------

def detect_figure_number(text: str):
    match = re.search(
        r"\b(?:figure|fig\.?|fig)\s*(\d+)\b",
        text.lower()
    )
    return match.group(1) if match else None


def detect_table_number(text: str):
    match = re.search(
        r"\btable\s*(\d+)\b",
        text.lower()
    )
    return match.group(1) if match else None


def detect_content_type(text: str, modality: str):
    text_lower = text.lower()

    if (
        "start of picture text" in text_lower
        or (
            "picture" in text_lower
            and "intentionally omitted" in text_lower
        )
    ):
        return "figure_ocr"

    if detect_figure_number(text):
        return "figure_caption_or_reference"

    if detect_table_number(text):
        return "table"

    if any(
        token in text
        for token in [
            "\\frac",
            "$$",
            "dC",
            "d/dt",
            "ODE",
            "equation",
        ]
    ):
        return "equation_or_model_text"

    return modality or "text"


def enrich_metadata(
    content: str,
    metadata: dict,
    pdf_path: str,
    section_index=None,
    modality: str = "text",
):
    """
    Add consistent metadata to every chunk.
    This makes LangSmith traces easier to debug.
    """

    source_pdf = os.path.basename(pdf_path)
    metadata = dict(metadata) if metadata else {}

    page = (
        metadata.get("page")
        or metadata.get("page_number")
        or metadata.get("pageno")
        or "missing_from_parser"
    )

    if section_index is None:
        section_index = metadata.get("section_index", "missing_from_parser")

    figure_number = detect_figure_number(content)
    table_number = detect_table_number(content)
    content_type = detect_content_type(content, modality)

    contains_picture_text = content_type == "figure_ocr"

    metadata.update(
        {
            "source_pdf": source_pdf,
            "page": page,
            "section_index": section_index,
            "modality": modality,
            "content_type": content_type,
            "figure_number": figure_number,
            "table_number": table_number,
            "contains_picture_text": contains_picture_text,
            "chunk_preview": content[:150].replace("\n", " "),
        }
    )

    metadata["chunk_id"] = (
        f"{source_pdf}"
        f"_page_{page}"
        f"_section_{section_index}"
        f"_{content_type}"
    )

    return metadata


# --------------------------------------------------
# Text / markdown parsing
# --------------------------------------------------

def parse_pdf_with_marker(pdf_path: str):
    """
    Fast Markdown-style PDF parsing using PyMuPDF4LLM.
    Uses page_chunks=True when available so we preserve page metadata.
    """

    try:
        import pymupdf4llm
    except ImportError as error:
        raise ImportError(
            "PyMuPDF4LLM is not installed. Run: pip install pymupdf4llm"
        ) from error

    try:
        page_chunks = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True
        )
        return page_chunks

    except TypeError:
        # Fallback for older PyMuPDF4LLM versions
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        return markdown_text


def split_page_markdown_into_sections(
    page_text: str,
    pdf_path: str,
    page_number,
):
    """
    Split one page of markdown into smaller section documents.
    """

    documents = []

    sections = re.split(
        r"\n(?=# )",
        page_text
    )

    for section_index, section in enumerate(sections):

        section = section.strip()

        if not section:
            continue

        metadata = enrich_metadata(
            content=section,
            metadata={"page": page_number},
            pdf_path=pdf_path,
            section_index=section_index,
            modality="marker_markdown",
        )

        documents.append(
            Document(
                page_content=section,
                metadata=metadata,
            )
        )

    return documents


def split_marker_markdown_into_documents(
    marker_output,
    pdf_path: str,
):
    """
    Convert PyMuPDF4LLM output into LangChain Documents.

    Supports:
    - page_chunks=True output
    - fallback full markdown string output
    """

    documents = []

    # Case 1: modern PyMuPDF4LLM page_chunks output
    if isinstance(marker_output, list):

        for page_index, page_item in enumerate(marker_output, start=1):

            if isinstance(page_item, dict):
                page_text = (
                    page_item.get("text")
                    or page_item.get("markdown")
                    or ""
                )

                metadata = page_item.get("metadata", {})
                page_number = (
                    metadata.get("page")
                    or metadata.get("page_number")
                    or page_index
                )

            else:
                page_text = str(page_item)
                page_number = page_index

            page_documents = split_page_markdown_into_sections(
                page_text=page_text,
                pdf_path=pdf_path,
                page_number=page_number,
            )

            documents.extend(page_documents)

        return documents

    # Case 2: fallback full markdown string
    markdown_text = str(marker_output)

    sections = re.split(
        r"\n(?=# )",
        markdown_text
    )

    for section_index, section in enumerate(sections):

        section = section.strip()

        if not section:
            continue

        metadata = enrich_metadata(
            content=section,
            metadata={"page": "missing_from_parser"},
            pdf_path=pdf_path,
            section_index=section_index,
            modality="marker_markdown",
        )

        documents.append(
            Document(
                page_content=section,
                metadata=metadata,
            )
        )

    return documents


# --------------------------------------------------
# Vision / figure description
# --------------------------------------------------

def get_cached_or_generate_figure_description(
    image_path: str,
    page_number,
    block_index,
):
    """
    Load cached figure description if available.

    If USE_VISION=False:
        create a cheap placeholder with useful trace metadata.

    If USE_VISION=True:
        call vision model once and cache result.
    """

    description_path = image_path + ".txt"

    if os.path.exists(description_path):
        with open(description_path, "r", encoding="utf-8") as file:
            return file.read()

    if not USE_VISION:
        description = (
            f"Figure image extracted from page {page_number}, "
            f"block {block_index}. "
            f"Image path: {image_path}. "
            "Vision description skipped because USE_VISION=False."
        )

        with open(description_path, "w", encoding="utf-8") as file:
            file.write(description)

        return description

    description = describe_figure_with_gemini(image_path)

    with open(description_path, "w", encoding="utf-8") as file:
        file.write(description)

    return description


# --------------------------------------------------
# Figure extraction
# --------------------------------------------------

def extract_figures_from_page(
    page,
    page_number,
    output_dir,
):
    """
    Extract visible image/figure regions from a PDF page using PyMuPDF.
    """

    figure_items = []

    page_dict = page.get_text("dict")
    page_text = page.get_text("text")

    page_figure_numbers = re.findall(
        r"\b(?:Figure|Fig\.?|Fig)\s*(\d+)\b",
        page_text,
        flags=re.IGNORECASE
    )

    for block_index, block in enumerate(page_dict["blocks"]):

        if block["type"] != 1:
            continue

        x0, y0, x1, y1 = block["bbox"]

        width = x1 - x0
        height = y1 - y0

        if width < 100 or height < 100:
            continue

        rect = fitz.Rect(x0, y0, x1, y1)

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            clip=rect
        )

        image_path = os.path.join(
            output_dir,
            f"page_{page_number}_figure_{block_index}.png"
        )

        pix.save(image_path)

        figure_items.append(
            {
                "image_path": image_path,
                "page": page_number,
                "block_index": block_index,
                "page_text": page_text,
                "page_figure_numbers": page_figure_numbers,
            }
        )

    return figure_items


def extract_figures_with_pymupdf(pdf_path: str):
    """
    Extract figures from PDF and return them as LangChain Documents.
    """

    pdf = fitz.open(pdf_path)

    documents = []

    image_output_dir = "outputs/extracted_images"
    os.makedirs(image_output_dir, exist_ok=True)

    for page_number, page in enumerate(pdf, start=1):

        figure_items = extract_figures_from_page(
            page=page,
            page_number=page_number,
            output_dir=image_output_dir,
        )

        for figure in figure_items:

            try:
                image_description = get_cached_or_generate_figure_description(
                    image_path=figure["image_path"],
                    page_number=figure["page"],
                    block_index=figure["block_index"],
                )

            except Exception as error:
                image_description = (
                    f"Figure extracted but description failed: {error}"
                )

            figure_numbers = figure.get("page_figure_numbers", [])

            figure_number = (
                figure_numbers[0]
                if figure_numbers
                else None
            )

            content = (
                f"{image_description}\n\n"
                f"Nearby page text:\n"
                f"{figure['page_text'][:1500]}"
            )

            metadata = enrich_metadata(
                content=content,
                metadata={
                    "page": figure["page"],
                    "image_path": figure["image_path"],
                    "block_index": figure["block_index"],
                    "figure_number": figure_number,
                },
                pdf_path=pdf_path,
                section_index=figure["block_index"],
                modality="figure_image",
            )

            if figure_number is not None:
                metadata["figure_number"] = figure_number

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    return documents


# --------------------------------------------------
# Main parser
# --------------------------------------------------

def parse_pdf_multimodal(pdf_path: str):
    """
    Main parser used by the app.

    Combines:
    - PyMuPDF4LLM for text, tables, equations
    - PyMuPDF for figure extraction
    - optional GPT vision descriptions for figures
    """

    documents = []

    print("Parsing PDF with PyMuPDF4LLM...")

    marker_output = parse_pdf_with_marker(pdf_path)

    marker_documents = split_marker_markdown_into_documents(
        marker_output=marker_output,
        pdf_path=pdf_path,
    )

    documents.extend(marker_documents)

    print(
        f"PyMuPDF4LLM produced "
        f"{len(marker_documents)} text/table/equation chunks."
    )

    print("Extracting figures with PyMuPDF...")

    figure_documents = extract_figures_with_pymupdf(pdf_path)

    documents.extend(figure_documents)

    print(f"Extracted {len(figure_documents)} figure chunks.")

    return documents