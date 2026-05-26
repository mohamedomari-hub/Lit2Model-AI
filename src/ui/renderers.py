import base64
import os
import re

import fitz
import streamlit as st
import streamlit.components.v1 as components


def render_markdown_with_latex(text: str):
    text = text.replace("\\[", "$$")
    text = text.replace("\\]", "$$")

    text = re.sub(
        r"\\\((.*?)\\\)",
        r"$\1$",
        text,
        flags=re.DOTALL,
    )

    st.markdown(text, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_pdf_pages_for_viewer(
    pdf_path: str,
    modified_time: float,
    zoom: float = 1.35,
):
    pages = []
    pdf = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(pdf):
            pix = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                alpha=False,
            )
            pages.append(
                {
                    "page": page_index + 1,
                    "image": base64.b64encode(pix.tobytes("png")).decode("utf-8"),
                }
            )
    finally:
        pdf.close()

    return pages


def render_pdf_viewer(pdf_path: str, height: int = 780, zoom: float = 1.35):
    if not pdf_path or not os.path.exists(pdf_path):
        st.warning("PDF file not found.")
        return

    pages = load_pdf_pages_for_viewer(
        pdf_path=pdf_path,
        modified_time=os.path.getmtime(pdf_path),
        zoom=zoom,
    )
    visual_width = int(zoom * 100)

    page_html = "\n".join(
        f"""
        <section class="pdf-page">
            <div class="pdf-page-label">Page {page["page"]}</div>
            <img src="data:image/png;base64,{page["image"]}" />
        </section>
        """
        for page in pages
    )

    components.html(
        f"""
        <style>
            .pdf-scroll {{
                height: {height}px;
                overflow: auto;
                padding: 12px;
                border: 1px solid #d8e2e7;
                border-radius: 8px;
                background: #eef2f6;
            }}

            .pdf-page {{
                margin: 0 auto 16px;
                width: {visual_width}%;
                min-width: 65%;
                max-width: none;
            }}

            .pdf-page-label {{
                margin: 0 0 6px;
                color: #64748b;
                font: 600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                text-align: center;
            }}

            .pdf-page img {{
                display: block;
                width: 100%;
                height: auto;
                background: white;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                box-shadow: 0 1px 8px rgba(15, 23, 42, 0.12);
            }}
        </style>
        <div class="pdf-scroll">
            {page_html}
        </div>
        """,
        height=height + 28,
    )


def render_mermaid(mermaid_code: str):
    components.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{startOnLoad:true}});
        </script>

        <div class="mermaid">
        {mermaid_code}
        </div>
        """,
        height=500,
        scrolling=True,
    )


def split_review_model_card_sections(text: str) -> dict[str, str]:
    section_names = [
        "Scope",
        "State Variables",
        "Parameters",
        "Equations",
        "Mechanisms",
        "Missing / Needs Review",
    ]
    sections = {}

    for index, section_name in enumerate(section_names):
        next_names = section_names[index + 1:]
        next_pattern = "|".join(re.escape(name) for name in next_names)
        pattern = (
            rf"^##\s+{re.escape(section_name)}\s*$"
            rf"(.*?)"
            rf"(?=^##\s+(?:{next_pattern})\s*$|\Z)"
            if next_pattern
            else rf"^##\s+{re.escape(section_name)}\s*$(.*)\Z"
        )
        match = re.search(
            pattern,
            text or "",
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        if match:
            sections[section_name] = match.group(1).strip()

    return sections


def render_review_model_card_preview(text: str):
    sections = split_review_model_card_sections(text)

    if not sections:
        st.info("No model-card sections found yet.")
        return

    st.markdown("### Section preview")

    for section_name in [
        "Scope",
        "State Variables",
        "Parameters",
        "Equations",
        "Mechanisms",
        "Missing / Needs Review",
    ]:
        content = sections.get(section_name)

        if not content:
            continue

        with st.expander(section_name, expanded=section_name == "Equations"):
            st.code(content, language="yaml")
