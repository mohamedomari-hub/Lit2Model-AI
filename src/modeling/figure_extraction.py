"""
Figure retrieval and OCR description helpers.
"""

import os
import re

import fitz
from langchain_openai import ChatOpenAI

from src.paper_processing.crops import render_pdf_page_to_image
from src.paper_processing.ocr import describe_figure_with_gemini
from src.retrieval import format_doc_metadata, make_chunk_key

VECTOR_STORE = None
ACTIVE_PDF_PATH = None


def configure_figure_extraction_runtime(vector_store=None, pdf_path: str | None = None):
    global VECTOR_STORE, ACTIVE_PDF_PATH
    VECTOR_STORE = vector_store
    ACTIVE_PDF_PATH = pdf_path


def _figure_caption_regex(figure_number: str) -> str:
    escaped_number = re.escape(str(figure_number))
    return rf"^\s*(?:figure|fig\.?)\s*{escaped_number}\s*([:.])\s*(.*)$"

def is_figure_caption_match(content: str, figure_number: str) -> bool:
    """
    True only when the text looks like the actual caption heading, e.g.
    'Figure 7:' or 'Fig. 7.'. Prose references such as 'shown in Fig 7(E)'
    are not enough.
    """

    if not content:
        return False

    caption_pattern = re.compile(
        _figure_caption_regex(figure_number),
        flags=re.IGNORECASE | re.MULTILINE
    )

    for match in caption_pattern.finditer(content):
        punctuation = match.group(1)
        remainder = (match.group(2) or "").strip()

        if punctuation == ":":
            return True

        if len(remainder) >= 10:
            return True

    return False

def extract_figure_caption_from_pdf(
    pdf_path: str,
    figure_number: str,
    max_caption_lines: int = 12
):
    """
    Scan the PDF text layer for the requested figure caption.

    This is stricter than semantic retrieval: it requires a caption-style
    anchor like 'Figure 7:' or 'Fig. 7.' and ignores prose references.
    """

    if not pdf_path or not os.path.exists(pdf_path):
        return None

    caption_pattern = re.compile(
        _figure_caption_regex(figure_number),
        flags=re.IGNORECASE | re.MULTILINE
    )
    next_caption_pattern = re.compile(
        r"^\s*(?:figure|fig\.?)\s*\d+\s*[:.]",
        flags=re.IGNORECASE
    )

    pdf = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(pdf):
            lines = [
                line.strip()
                for line in page.get_text("text").splitlines()
                if line.strip()
            ]

            for line_index, line in enumerate(lines):
                caption_match = caption_pattern.search(line)

                if not caption_match:
                    continue

                punctuation = caption_match.group(1)
                remainder = (caption_match.group(2) or "").strip()

                if punctuation != ":" and len(remainder) < 10:
                    continue

                caption_lines = [line]

                for next_line in lines[line_index + 1:]:
                    if next_caption_pattern.search(next_line):
                        break

                    caption_lines.append(next_line)

                    if len(caption_lines) >= max_caption_lines:
                        break

                return {
                    "page": page_index + 1,
                    "caption": " ".join(caption_lines),
                }
    finally:
        pdf.close()

    return None

def is_exact_figure_match(content: str, metadata: dict, figure_number: str) -> bool:
    """
    Check whether retrieved content explicitly corresponds to the requested figure.
    """

    metadata = metadata or {}
    if str(metadata.get("figure_number")) == str(figure_number):
        return True

    return is_figure_caption_match(
        content=content,
        figure_number=figure_number
    )

def retrieve_figure_context_from_active_pdf(query: str) -> str:
    """
    Retrieve figure-specific evidence:
    captions, OCR picture text, nearby discussion, and figure references.

    Uses exact figure-number matches first.
    Falls back to semantic figure retrieval only if no exact match is found.
    """

    if VECTOR_STORE is None:
        return "Vector store is not initialized."

    query_lower = query.lower()

    figure_match = re.search(
        r"(figure|fig\.?|fig)\s*(\d+)",
        query_lower
    )

    search_queries = [query]

    figure_number = None

    if figure_match:
        figure_number = figure_match.group(2)

        search_queries.extend([
            f"Figure {figure_number}",
            f"Fig {figure_number}",
            f"Fig. {figure_number}",
            f"Figure {figure_number} caption",
            f"Fig. {figure_number} caption",
            f"description of Figure {figure_number}",
            f"results shown in Figure {figure_number}",
            f"text discussing Figure {figure_number}",
        ])

    else:
        search_queries.extend([
            "figure caption",
            "figure description",
            "figure results",
            "diagram caption",
            "model diagram",
        ])

    search_queries = list(dict.fromkeys(search_queries))

    exact_results = []
    fallback_results = []
    seen = set()

    pdf_caption_match = None

    if figure_number is not None and ACTIVE_PDF_PATH is not None:
        pdf_caption_match = extract_figure_caption_from_pdf(
            pdf_path=ACTIVE_PDF_PATH,
            figure_number=figure_number
        )

        if pdf_caption_match is not None:
            exact_results.append(
                "[STRICT FIGURE CAPTION MATCH: "
                f"Figure {figure_number} | "
                f"Source: {os.path.basename(ACTIVE_PDF_PATH)} | "
                f"Page: {pdf_caption_match['page']}]\n"
                f"{pdf_caption_match['caption']}"
            )

    for search_query in search_queries:

        docs = VECTOR_STORE.similarity_search(
            search_query,
            k=6
        )

        for doc in docs:
            chunk_key = make_chunk_key(doc)

            if chunk_key in seen:
                continue

            seen.add(chunk_key)

            content = doc.page_content

            if "Start of picture text" in content:
                content = (
                    "[FIGURE OCR TEXT: visual figure was omitted by parser, "
                    "but OCR extracted visible labels, axes, legends, or units.]\n"
                    + content
                )

            header = format_doc_metadata(
                doc=doc,
                search_query=search_query,
                label="Figure Search Query"
            )

            entry = f"{header}\n{content}"

            if figure_number is not None:
                if is_exact_figure_match(
                    content=content,
                    metadata=doc.metadata or {},
                    figure_number=figure_number
                ):
                    exact_results.append(entry)
                else:
                    fallback_results.append(entry)
            else:
                fallback_results.append(entry)

    if exact_results:
        return "\n\n---\n\n".join(exact_results)

    if figure_number is not None and fallback_results:
        warning = (
            f"No exact Figure {figure_number} match was found. "
            "The following chunks were retrieved semantically and may not correspond "
            "to the requested figure."
        )

        return warning + "\n\n---\n\n" + "\n\n---\n\n".join(fallback_results)

    if fallback_results:
        return "\n\n---\n\n".join(fallback_results)

    return "No figure-specific context was retrieved from the PDF."

def should_use_gemini_for_figure(figure_context: str) -> bool:
    """
    Decide whether Gemini vision should be used for figure interpretation.
    """

    context_lower = figure_context.lower()

    triggers = [
        "picture intentionally omitted",
        "figure image extracted",
        "vision description skipped",
        "no exact figure",
        "may not correspond to the requested figure",
        "ocr",
        "visual figure was omitted",
    ]

    return any(trigger in context_lower for trigger in triggers)

def extract_first_page_number_from_context(figure_context: str):
    match = re.search(r"Page:\s*(\d+)", figure_context)

    if match:
        return int(match.group(1))

    return None

def extract_best_figure_page_from_context(
    figure_context: str,
    figure_number: str | None = None
):
    """
    Prefer the page tied to the strict caption match. Fallback to a chunk that
    contains a caption-style figure anchor, then finally to the first page.
    """

    strict_match = re.search(
        r"\[STRICT FIGURE CAPTION MATCH:.*?\|\s*Page:\s*(\d+)\]",
        figure_context,
        flags=re.IGNORECASE | re.DOTALL
    )

    if strict_match:
        return int(strict_match.group(1))

    if figure_number is not None:
        chunks = figure_context.split("---")

        for chunk in chunks:
            if not is_figure_caption_match(chunk, figure_number):
                continue

            page_match = re.search(r"Page:\s*(\d+)", chunk)

            if page_match:
                return int(page_match.group(1))

    return extract_first_page_number_from_context(figure_context)

def get_gemini_figure_fallback(
    figure_context: str,
    figure_number: str | None = None
) -> str:
    """
    Use Gemini vision on the retrieved figure page if figure context is incomplete.
    """

    if ACTIVE_PDF_PATH is None:
        return "Gemini figure fallback skipped: active PDF path is not set."

    page_number = extract_best_figure_page_from_context(
        figure_context=figure_context,
        figure_number=figure_number
    )

    if page_number is None:
        return "Gemini figure fallback skipped: no page number found in retrieved context."

    try:
        image_path = render_pdf_page_to_image(
            pdf_path=ACTIVE_PDF_PATH,
            page_number=page_number
        )

        gemini_description = describe_figure_with_gemini(image_path)

        return f"""
GEMINI FIGURE VISION FALLBACK
Rendered PDF page: {page_number}
Image path: {image_path}

{gemini_description}
"""

    except Exception as error:
        return f"""
Gemini figure fallback failed.

Reason:
{type(error).__name__}: {error}
"""

def extract_figure_explanation(
    figure_context: str,
    user_question: str
) -> str:
    """
    Convert retrieved figure context into a cautious scientific explanation.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a scientific figure interpretation assistant.

Use only the retrieved figure context to answer the user's question.

User question:
{user_question}

Retrieved figure context:
{figure_context}

Rules:
- If the context contains "STRICT FIGURE CAPTION MATCH", treat that caption
  and page as the primary evidence for the requested numbered figure.
- Do not use prose references like "shown in Fig. 7" as proof that the chunk is
  the actual Figure 7 caption.
- If the context contains "FIGURE OCR TEXT", treat it as OCR-derived visual evidence.
- Extract visible axis labels, units, variables, legends, and panel labels.
- Separate:
  1. Directly retrieved evidence
  2. Interpretation
  3. Limitations
- Do not invent curve shapes or trends if they are not visible in the retrieved context.
- Do not say no figure context was found if OCR figure text is available.
- If the original visual image is unavailable, clearly say that detailed visual interpretation is limited.
- Be scientifically cautious.

Critical limitation wording rule:
- If GEMINI VISUAL FIGURE CONTEXT contains a non-empty Gemini response, NEVER write:
  "the visual representation was absent"
  "the actual figure image was absent"
  "without the visual representation"
- Instead write:
  "The text parser omitted the original figure image, but Gemini analyzed a rendered PDF page image."
- Mention that fine visual details still require human review.

Limitations section rule:
- If Gemini visual context is available, the limitation is NOT absence of the image.
- The limitation is that the figure was interpreted from a rendered PDF page and should be human-reviewed for fine details.

Return a clear explanation.
"""

    result = llm.invoke(prompt)

    return result.content


def retrieve_figure_context_service(
    vector_store,
    pdf_path: str | None,
    query: str,
    figure_number: str | None = None,
) -> str:
    configure_figure_extraction_runtime(vector_store, pdf_path)
    if figure_number:
        return retrieve_figure_context_from_active_pdf(f"{query} Figure {figure_number}")
    return retrieve_figure_context_from_active_pdf(query)


def explain_figure_from_pdf_service(vector_store, pdf_path: str | None, query: str) -> str:
    configure_figure_extraction_runtime(vector_store, pdf_path)
    figure_match = re.search(
        r"(figure|fig\.?|fig)\s*(\d+)",
        query.lower()
    )
    figure_number = figure_match.group(2) if figure_match else None

    figure_context = retrieve_figure_context_from_active_pdf(query)

    gemini_context = ""

    if should_use_gemini_for_figure(figure_context):
        gemini_context = get_gemini_figure_fallback(
            figure_context=figure_context,
            figure_number=figure_number
        )

    combined_context = f"""
                        RETRIEVED FIGURE CONTEXT:
                        {figure_context}

                        GEMINI VISUAL FIGURE CONTEXT:
                        {gemini_context}
                        """

    answer = extract_figure_explanation(
        figure_context=combined_context,
        user_question=query
    )

    return f"""
            FIGURE RETRIEVAL CONTEXT:
            {figure_context}

            GEMINI VISUAL FIGURE CONTEXT:
            {gemini_context}

            ==================================================

            FIGURE EXPLANATION:
            {answer}
            """
