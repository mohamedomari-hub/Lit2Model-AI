"""Parse scientific PDFs into documents, chunks, and extracted assets."""

import os
import re
import fitz

from langchain_core.documents import Document
from src.ingestion.ocr import describe_figure_with_gemini

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


def detect_equation_number(text: str):
    match = re.search(
        r"\(\s*(\d+)\s*\)\s*$",
        text.strip(),
        flags=re.MULTILINE
    )
    return match.group(1) if match else None


def looks_like_equation_text(text: str) -> bool:
    text = text.strip()
    text_lower = text.lower()

    if not text:
        return False

    if "picture intentionally omitted" in text_lower:
        return True

    equation_markers = [
        "\\frac",
        "d/dt",
        "dC",
        "\\dot",
        "\\partial",
        "\\sum",
        "\\int",
    ]

    if any(marker in text for marker in equation_markers):
        return True

    if "=" not in text:
        return False

    math_symbol_count = len(
        re.findall(
            r"[A-Za-zΑ-ω][A-Za-z0-9_{}\\-]*|\^|/|\+|-|\*|\(|\)",
            text
        )
    )

    return math_symbol_count >= 4


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

    if detect_equation_number(text):
        return "equation_candidate"

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
# Equation candidate extraction
# --------------------------------------------------

def extract_equation_candidates_with_pymupdf(

    pdf_path: str,

    output_dir: str,

):
    """
    Extract local, structured equation candidate documents from the PDF text
    layer and render candidate crops for human/OCR follow-up.

    This does not run OCR or call external APIs. It only preserves better
    retrieval anchors for displayed equations and parser omission placeholders.
    """

    pdf = fitz.open(pdf_path)
    documents = []
    os.makedirs(output_dir, exist_ok=True)
    seen = set()

    try:
        for page_number, page in enumerate(pdf, start=1):
            page_rect = page.rect
            raw_blocks = page.get_text("blocks")
            blocks = [
                {
                    "rect": fitz.Rect(block[:4]),
                    "text": str(block[4]).strip(),
                    "index": block_index,
                }
                for block_index, block in enumerate(raw_blocks)
                if str(block[4]).strip()
            ]
            blocks.sort(key=lambda item: (item["rect"].y0, item["rect"].x0))

            for block_index, block in enumerate(blocks):
                block_text = block["text"]

                if not looks_like_equation_text(block_text):
                    continue

                nearby_parts = []

                if block_index > 0:
                    nearby_parts.append(blocks[block_index - 1]["text"])

                nearby_parts.append(block_text)

                if block_index + 1 < len(blocks):
                    nearby_parts.append(blocks[block_index + 1]["text"])

                candidate_text = "\n".join(
                    part for part in nearby_parts if part.strip()
                )
                normalized_key = re.sub(r"\s+", " ", candidate_text)

                if normalized_key in seen:
                    continue

                seen.add(normalized_key)
                equation_number = detect_equation_number(candidate_text)

                crop_rect = fitz.Rect(
                    page_rect.x0 + page_rect.width * 0.04,
                    max(page_rect.y0, block["rect"].y0 - 80),
                    page_rect.x1 - page_rect.width * 0.03,
                    min(page_rect.y1, block["rect"].y1 + 80),
                ) & page_rect

                if crop_rect.width <= 1 or crop_rect.height <= 1:
                    continue

                image_path = os.path.join(
                    output_dir,
                    f"page_{page_number}_equation_candidate_{block['index']}.png"
                )

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(4, 4),
                    clip=crop_rect,
                    alpha=False
                )
                pix.save(image_path)

                content = f"""
Equation candidate record
- equation_number: {equation_number or "not detected"}
- page: {page_number}
- source: PDF text layer / rendered crop
- method: local parser candidate extraction
- confidence: candidate
- requires_review: true
- image_path: {image_path}

Candidate text:
```text
{candidate_text}
```
"""

                metadata = enrich_metadata(
                    content=content,
                    metadata={
                        "page": page_number,
                        "equation_number": equation_number,
                        "image_path": image_path,
                        "requires_review": True,
                    },
                    pdf_path=pdf_path,
                    section_index=f"equation_candidate_{block['index']}",
                    modality="equation_candidate",
                )

                metadata["content_type"] = "equation_candidate"
                metadata["equation_number"] = equation_number
                metadata["image_path"] = image_path
                metadata["requires_review"] = True

                documents.append(
                    Document(
                        page_content=content,
                        metadata=metadata,
                    )
                )
    finally:
        pdf.close()

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

def parse_pdf_with_pymupdf_fast(pdf_path: str):
    """
    Fast plain text parser using PyMuPDF.

    Good for:
    - speed
    - simple text-layer extraction
    - fallback comparison against PyMuPDF4LLM

    Weak for:
    - complex tables
    - equations/layout
    """

    import fitz
    from langchain_core.documents import Document

    documents = []

    pdf = fitz.open(pdf_path)

    for page_index, page in enumerate(pdf):
        page_number = page_index + 1

        text = page.get_text("text")

        if not text or not text.strip():
            continue

        metadata = enrich_metadata(
            content=text,
            metadata={"page": page_number},
            pdf_path=pdf_path,
            section_index=0,
            modality="pymupdf_fast_text",
            content_type="text",
        )

        documents.append(
            Document(
                page_content=text.strip(),
                metadata=metadata,
            )
        )

    pdf.close()

    return documents

def normalize_scientific_text(text: str) -> str:
    """
    Clean parser artifacts without changing scientific meaning.
    """

    replacements = {
        "\u000c": "\n",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\ufffe": "",
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "−": "-",
        "–": "-",
        "—": "-",
        "·": " * ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _looks_model_relevant(text: str) -> bool:
    """
    General mechanistic-model relevance detector.
    No paper-specific symbols.
    """

    patterns = [
        r"\bd\s*/\s*dt\b",
        r"\bd[A-Za-z][A-Za-z0-9_ -]*\s*/\s*dt\b",
        r"\bODE\b",
        r"\bequation\b",
        r"\bEq\.?\s*\(?\d+\)?",
        r"\bparameter\b",
        r"\bunit\b",
        r"\binitial condition\b",
        r"\bstate variable\b",
        r"\bcompartment\b",
        r"\binput\b",
        r"\bdose\b",
        r"\bforcing\b",
        r"\bintervention\b",
        r"\bmechanism\b",
        r"\bstimulat",
        r"\binhibit",
        r"\bactivation\b",
        r"\bsuppression\b",
        r"\bfeedback\b",
        r"\bHill\b",
        r"\bEmax\b",
        r"\bEC50\b",
        r"\bIC50\b",
        r"\bKm\b",
        r"\bVmax\b",
        r"=",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def _split_markdown_by_headings(markdown_text: str) -> list[str]:
    """
    Split markdown into semantic sections while preserving heading blocks.
    """

    lines = markdown_text.splitlines()
    blocks = []
    current = []

    heading_pattern = re.compile(r"^\s{0,3}#{1,4}\s+")

    for line in lines:
        if heading_pattern.match(line) and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    return [
        block for block in blocks
        if block and len(block.strip()) >= 40
    ]


def _split_large_block(
    block: str,
    max_chars: int = 3500,
    overlap_chars: int = 400,
) -> list[str]:
    """
    Split only very large blocks.
    Try to preserve paragraphs and equation groups.
    """

    if len(block) <= max_chars:
        return [block]

    paragraphs = re.split(r"\n\s*\n", block)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            current + "\n\n" + paragraph
            if current
            else paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)

            if len(paragraph) > max_chars:
                start = 0

                while start < len(paragraph):
                    end = start + max_chars
                    chunks.append(paragraph[start:end])
                    start = max(0, end - overlap_chars)

                    if start >= len(paragraph):
                        break

                current = ""
            else:
                current = paragraph

    if current:
        chunks.append(current)

    return chunks


def _extract_equation_neighborhoods(
    markdown_text: str,
    window: int = 8,
) -> list[str]:
    """
    Add extra equation-centered chunks.

    This protects linked algebraic/process equations from being separated.
    """

    lines = markdown_text.splitlines()
    selected_blocks = []

    equation_line_pattern = re.compile(
        r"("
        r"\bd\s*/\s*dt\b|"
        r"\bd[A-Za-z][A-Za-z0-9_ -]*\s*/\s*dt\b|"
        r"\bEq\.?\s*\(?\d+\)?|"
        r"="
        r")",
        flags=re.IGNORECASE,
    )

    used_ranges = []

    for i, line in enumerate(lines):
        if not equation_line_pattern.search(line):
            continue

        start = max(0, i - window)
        end = min(len(lines), i + window + 1)

        # merge overlapping equation neighborhoods
        if used_ranges and start <= used_ranges[-1][1]:
            used_ranges[-1] = (
                used_ranges[-1][0],
                max(used_ranges[-1][1], end),
            )
        else:
            used_ranges.append((start, end))

    for start, end in used_ranges:
        block = "\n".join(lines[start:end]).strip()

        if len(block) >= 40:
            selected_blocks.append(block)

    return selected_blocks


def split_docling_markdown_into_documents(
    markdown_text: str,
    pdf_path: str,
):
    """
    Structure-aware Docling chunking for mechanistic model discovery.

    Creates:
    1. semantic markdown sections
    2. extra equation-neighborhood chunks

    This improves retrieval without hardcoding a specific paper/model.
    """

    from langchain_core.documents import Document

    markdown_text = normalize_scientific_text(markdown_text)

    documents = []
    seen = set()

    semantic_blocks = _split_markdown_by_headings(markdown_text)
    equation_blocks = _extract_equation_neighborhoods(markdown_text)

    all_blocks = []

    for block in semantic_blocks:
        all_blocks.extend(
            _split_large_block(block)
        )

    for block in equation_blocks:
        all_blocks.append(block)

    for section_index, block in enumerate(all_blocks):
        block = block.strip()

        if len(block) < 40:
            continue

        key = re.sub(r"\s+", " ", block[:800]).lower()

        if key in seen:
            continue

        seen.add(key)

        content_type = (
            "model_evidence"
            if _looks_model_relevant(block)
            else "text"
        )

        metadata = enrich_metadata(
            content=block,
            metadata={
                "page": "docling",
                "content_type": content_type,
                "parser": "docling",
            },
            pdf_path=pdf_path,
            section_index=section_index,
            modality="docling_markdown",
        )

        documents.append(
            Document(
                page_content=block,
                metadata=metadata,
            )
        )

    return documents

def parse_pdf_with_docling(pdf_path: str):
    """
    Docling parser.

    Converts PDF to Markdown using Docling,
    then splits markdown into retrieval-friendly sections.
    """

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    markdown_text = result.document.export_to_markdown()

    documents = split_docling_markdown_into_documents(
        markdown_text=markdown_text,
        pdf_path=pdf_path,
    )

    return documents


def parse_pdf_multimodal(
    pdf_path: str,
    parser_mode: str = "pymupdf4llm",
    equation_candidates_dir: str = "outputs/equation_candidates",
):
    """
    Main parser used by the app.

    Combines:
    - PyMuPDF4LLM for text, tables, equations
    - PyMuPDF for figure extraction
    - optional GPT vision descriptions for figures
    """

    documents = []

    if parser_mode == "pymupdf4llm":
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

    elif parser_mode == "pymupdf_fast":
        print("Parsing PDF with PyMuPDF fast text parser...")

        fast_documents = parse_pdf_with_pymupdf_fast(
            pdf_path
        )

        documents.extend(fast_documents)

        print(
            f"PyMuPDF fast produced "
            f"{len(fast_documents)} text chunks."
        )

    elif parser_mode == "docling":
        print("Parsing PDF with Docling...")

        docling_documents = parse_pdf_with_docling(
            pdf_path
        )

        documents.extend(docling_documents)

        print(
            f"Docling produced "
            f"{len(docling_documents)} markdown chunks."
        )

    else:
        raise ValueError(f"Unknown parser_mode: {parser_mode}")

    print("Extracting local equation candidate records...")

    equation_candidate_documents = extract_equation_candidates_with_pymupdf(
        pdf_path=pdf_path,
        output_dir=equation_candidates_dir,
    )

    documents.extend(equation_candidate_documents)

    print(
        f"Extracted {len(equation_candidate_documents)} "
        "equation candidate chunks."
    )

    print("Extracting figures with PyMuPDF...")

    figure_documents = extract_figures_with_pymupdf(pdf_path)

    documents.extend(figure_documents)

    print(f"Extracted {len(figure_documents)} figure chunks.")

    return documents
