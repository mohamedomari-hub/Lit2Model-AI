"""
PDF page rendering and crop helpers for equations and figures.
"""

import os

import fitz


def render_pdf_page_for_equation(
    pdf_path: str,
    page_number: int,
    equation_number: str,
) -> str:
    """
    Render an equation-specific crop for OCR.

    The function searches nearby candidate pages first, then scans the PDF for
    the requested equation anchor to avoid cropping the wrong equation.
    """

    os.makedirs("outputs/equation_pages", exist_ok=True)
    pdf = fitz.open(pdf_path)

    anchor_patterns = [
        f"({equation_number})",
        f"Equation {equation_number}",
        f"Equation ({equation_number})",
        f"Eq. {equation_number}",
        f"Eq. ({equation_number})",
        f"Eq {equation_number}",
        f"Eq ({equation_number})",
    ]

    def collect_anchors(page):
        anchors = []
        for pattern in anchor_patterns:
            for rect in page.search_for(pattern):
                anchors.append((pattern, rect))
        return anchors

    def score_anchor(page_rect, item):
        pattern, rect = item
        right_margin_score = rect.x0 / max(page_rect.width, 1)
        compact_score = 2.0 if pattern == f"({equation_number})" else 0.5
        left_penalty = 2.0 if rect.x0 < page_rect.width * 0.50 else 0.0
        return (5.0 * right_margin_score) + compact_score - left_penalty

    candidate_indices = [
        page_number - 1,
        page_number,
        page_number - 2,
        page_number + 1,
    ]
    candidate_indices = [
        index for index in candidate_indices
        if 0 <= index < len(pdf)
    ]
    all_indices = candidate_indices + [
        index for index in range(len(pdf))
        if index not in candidate_indices
    ]

    best = None

    for page_index in all_indices:
        page = pdf[page_index]
        page_rect = page.rect
        anchors = collect_anchors(page)

        if not anchors:
            continue

        anchor = max(anchors, key=lambda item: score_anchor(page_rect, item))
        anchor_score = score_anchor(page_rect, anchor)

        if best is None or anchor_score > best["score"]:
            best = {
                "page_index": page_index,
                "page": page,
                "anchor": anchor,
                "score": anchor_score,
            }

    if best is None:
        pdf.close()
        raise ValueError(
            f"No visual/text anchor found for Equation {equation_number} in the PDF."
        )

    page_index = best["page_index"]
    page = best["page"]
    page_rect = page.rect
    _, anchor_rect = best["anchor"]
    rendered_page_number = page_index + 1

    crop_rect = fitz.Rect(
        page_rect.x0 + page_rect.width * 0.04,
        max(page_rect.y0, anchor_rect.y0 - 70),
        page_rect.x1 - page_rect.width * 0.03,
        min(page_rect.y1, anchor_rect.y1 + 70),
    )
    crop_rect = crop_rect & page_rect

    if crop_rect.width <= 1 or crop_rect.height <= 1:
        pdf.close()
        raise ValueError(
            f"Invalid crop for Equation {equation_number} on PDF page "
            f"{rendered_page_number}."
        )

    pix = page.get_pixmap(
        matrix=fitz.Matrix(5, 5),
        clip=crop_rect,
        alpha=False,
    )

    image_path = (
        "outputs/equation_pages/"
        f"equation_{equation_number}"
        f"_page_{rendered_page_number}.png"
    )

    pix.save(image_path)
    pdf.close()

    return image_path


def render_pdf_page_to_image(
    pdf_path: str,
    page_number: int,
) -> str:
    """
    Render a full PDF page to an image for figure-level OCR.
    """

    os.makedirs("outputs/figure_pages", exist_ok=True)

    pdf = fitz.open(pdf_path)
    page_index = page_number - 1

    if page_index < 0 or page_index >= len(pdf):
        pdf.close()
        raise ValueError(f"Invalid page number: {page_number}")

    page = pdf[page_index]
    pix = page.get_pixmap(
        matrix=fitz.Matrix(3, 3),
        alpha=False,
    )

    image_path = "outputs/figure_pages/" f"page_{page_number}.png"

    pix.save(image_path)
    pdf.close()

    return image_path
