"""
Streamlit rendering helpers for equations, PDF previews, and discovery reviews.
"""

import base64
import html
import json
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


def format_equation_for_display(equation_text: str) -> str:
    """
    Make common LaTeX equation text easier to read in the UI.
    This only changes display text, not saved extraction data.
    """
    def replace_fractions(value: str) -> str:
        result = []
        index = 0

        while index < len(value):
            frac_match = re.match(r"\\?frac\s*\{", value[index:])

            if not frac_match:
                result.append(value[index])
                index += 1
                continue

            numerator_start = index + frac_match.end() - 1
            numerator, after_numerator = read_braced_text(value, numerator_start)

            if numerator is None:
                result.append(value[index])
                index += 1
                continue

            denominator_start = after_numerator

            while denominator_start < len(value) and value[denominator_start].isspace():
                denominator_start += 1

            denominator, after_denominator = read_braced_text(value, denominator_start)

            if denominator is None:
                result.append(value[index])
                index += 1
                continue

            numerator = replace_fractions(numerator)
            denominator = replace_fractions(denominator)
            if any(operator in denominator for operator in ["+", "-", "*", "/"]):
                denominator = f"({denominator})"
            result.append(f"({numerator} / {denominator})")
            index = after_denominator

        return "".join(result)

    def read_braced_text(value: str, start_index: int):
        if start_index >= len(value) or value[start_index] != "{":
            return None, start_index

        depth = 0

        for index in range(start_index, len(value)):
            character = value[index]

            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1

                if depth == 0:
                    return value[start_index + 1:index], index + 1

        return None, start_index

    text = str(equation_text or "")

    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\times", "*")
    text = text.replace("\\,", " ")
    text = text.replace("\\;", " ")

    text = replace_fractions(text)

    text = re.sub(
        r"\(\s*d\s*/\s*dt\s*\)\s*([A-Za-z][A-Za-z0-9_{}^-]*)",
        r"d\1/dt",
        text,
    )

    text = re.sub(r"\\mathrm\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\text\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"_([A-Za-z0-9]+)", r"\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([A-Za-z0-9])-([A-Za-z0-9])", r"\1§HYPHEN§\2", text)
    text = re.sub(r"\s*([=+\-*/()])\s*", r" \1 ", text)
    text = text.replace("§HYPHEN§", "-")
    text = re.sub(r"d([A-Za-z0-9]+)\s*/\s*dt", r"d\1/dt", text)
    text = re.sub(r"\s*\^\s*", "^", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+", " ", text)

    if text.count("=") > 1:
        parts = [part.strip() for part in text.split("=")]
        text = f"{parts[0]} = {parts[1]}"

    return text.strip()


def render_equation_block(equation_text: str):
    """
    Render one equation in a readable review style without changing saved content.
    """
    equation_text = format_equation_for_display(equation_text)

    if not equation_text:
        st.markdown(
            '<div class="discovery-equation-block">not reported</div>',
            unsafe_allow_html=True,
        )
        return

    safe_text = html.escape(equation_text)
    st.markdown(
        f'<div class="discovery-equation-block">{safe_text}</div>',
        unsafe_allow_html=True,
    )


def _last_non_empty_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines()]

    for line in reversed(lines):
        if line:
            return line.lower()

    return ""


def render_discovery_review(text: str):
    """
    Render discovery markdown with polished equation blocks.
    The underlying extracted markdown/JSON is not changed.
    """
    if not text:
        return

    st.markdown(
        '<span class="discovery-review-marker"></span>',
        unsafe_allow_html=True,
    )

    text = re.sub(
        r"^\s*#\s+Model Discovery Review\s*\n+",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*status:\s*draft\s*/\s*requires human review\s*\n+",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    fence_pattern = re.compile(
        r"```(?P<language>[A-Za-z0-9_-]*)\n(?P<body>.*?)\n```",
        flags=re.DOTALL,
    )

    position = 0

    for match in fence_pattern.finditer(text):
        before = text[position:match.start()]
        language = (match.group("language") or "").strip().lower()
        body = match.group("body").strip()
        if language == "text":
            render_markdown_with_latex(before)
            render_equation_block(body)
        else:
            render_markdown_with_latex(
                before
                + f"```{language}\n{body}\n```"
            )

        position = match.end()

    remaining = text[position:]

    if remaining.strip():
        render_markdown_with_latex(remaining)


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


def render_pdf_viewer(
    pdf_path: str,
    height: int = 780,
    zoom: float = 1.35,
    selected_page: int | None = None,
):
    if not pdf_path or not os.path.exists(pdf_path):
        st.warning("PDF file not found.")
        return

    pages = load_pdf_pages_for_viewer(
        pdf_path=pdf_path,
        modified_time=os.path.getmtime(pdf_path),
        zoom=zoom,
    )
    visual_width = int(zoom * 100)
    page_count = len(pages)
    storage_key = (
        "lit2model_pdf_scroll_"
        + re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.abspath(pdf_path))
    )
    storage_key_json = json.dumps(storage_key)

    if page_count == 0:
        st.warning("PDF has no rendered pages.")
        return

    if selected_page is not None:
        if selected_page < 1:
            selected_page = 1

        if selected_page > page_count:
            selected_page = page_count

    page_html = "\n".join(
        f"""
        <section class="pdf-page" id="pdf-page-{page["page"]}">
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
        <div class="pdf-scroll" id="pdf-scroll-container">
            {page_html}
        </div>
        <script>
            const storageKey = {storage_key_json};
            const scrollContainer = document.getElementById("pdf-scroll-container");
            const selectedPage = {
                "null"
                if selected_page is None
                else f'document.getElementById("pdf-page-{selected_page}")'
            };

            if (selectedPage && scrollContainer) {{
                setTimeout(function() {{
                    scrollContainer.scrollTop = selectedPage.offsetTop - scrollContainer.offsetTop;
                }}, 80);
            }}

            if (!selectedPage && scrollContainer) {{
                function savePdfScroll() {{
                    const maxScroll = Math.max(
                        scrollContainer.scrollHeight - scrollContainer.clientHeight,
                        1
                    );
                    const scrollData = {{
                        top: scrollContainer.scrollTop,
                        ratio: scrollContainer.scrollTop / maxScroll
                    }};
                    window.localStorage.setItem(
                        storageKey,
                        JSON.stringify(scrollData)
                    );
                }}

                function restorePdfScroll() {{
                    const rawScrollData = window.localStorage.getItem(storageKey);

                    if (!rawScrollData) {{
                        return;
                    }}

                    try {{
                        const scrollData = JSON.parse(rawScrollData);
                        const maxScroll = Math.max(
                            scrollContainer.scrollHeight - scrollContainer.clientHeight,
                            1
                        );
                        const restoredTop = Math.round(
                            (scrollData.ratio || 0) * maxScroll
                        );
                        scrollContainer.scrollTop = restoredTop || scrollData.top || 0;
                    }} catch (error) {{
                        scrollContainer.scrollTop = 0;
                    }}
                }}

                scrollContainer.addEventListener(
                    "scroll",
                    savePdfScroll,
                    {{ passive: true }}
                );

                setTimeout(restorePdfScroll, 80);
                setTimeout(restorePdfScroll, 350);
            }}
        </script>
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
