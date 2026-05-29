"""
Global Streamlit CSS theme and typography settings.
"""

import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
            html, body, .stApp, p, label, table, textarea, input,
            [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {
                font-family: "Times New Roman", Georgia, serif !important;
            }

            h1, h2, h3, h4, h5, h6 {
                font-family: "Times New Roman", Georgia, serif !important;
                font-weight: 750 !important;
                line-height: 1.2 !important;
                margin-top: 0.8rem !important;
                margin-bottom: 0.45rem !important;
            }

            button,
            button p,
            .stButton button {
                font-family: "Times New Roman", Georgia, serif !important;
            }

            .material-icons,
            .material-symbols-rounded,
            .material-symbols-outlined,
            [data-testid="stIconMaterial"],
            [data-testid="stIconMaterial"] span {
                font-family: "Material Symbols Rounded", "Material Icons" !important;
                font-weight: normal !important;
                font-style: normal !important;
                line-height: 1 !important;
                letter-spacing: normal !important;
                text-transform: none !important;
                white-space: nowrap !important;
                word-wrap: normal !important;
                direction: ltr !important;
                -webkit-font-feature-settings: "liga" !important;
                -webkit-font-smoothing: antialiased !important;
                font-feature-settings: "liga" !important;
            }

            svg,
            svg * {
                font-family: initial !important;
            }

            /*
            CURRENT_AURORA_THEME_BACKUP
            Previous palette before stronger ciel-blue refinement:
            app background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 24rem),
            radial-gradient(circle at top right, rgba(252, 231, 243, 0.72), transparent 26rem),
            linear-gradient(180deg, #ffffff 0%, #fdf2f8 48%, #eef8ff 100%)
            sidebar: #fdf2f8 -> #eef8ff -> #f4f0ff
            rose border: #f5bfd1
            selected button: #0ea5d8 -> #38bdf8 -> #fda4af
            */

            /* Main interface theme (sidebar intentionally excluded) */
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(230, 234, 240, 0.34), transparent 26rem),
                    linear-gradient(180deg, #fafafa 0%, #f8f9fb 52%, #f7f8fa 100%) !important;
                color: #242936 !important;
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(230, 234, 240, 0.28), transparent 25rem),
                    linear-gradient(180deg, #fafafa 0%, #f8f9fb 100%) !important;
            }

            [data-testid="stAppViewContainer"] .block-container {
                color: #242936 !important;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(
                    180deg,
                    #d7eeff 0%,
                    #cfefff 44%,
                    #f9d2e3 100%
                ) !important;
                border-right: 1px solid #f6afcf !important;
            }

            [data-testid="stSidebar"] > div:first-child {
                padding: 0.55rem 0.6rem 0.85rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.46rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploader"] {
                background: #ffffff !important;
                border: 1px solid #f6afcf !important;
                border-radius: 10px !important;
                padding: 0.32rem !important;
                margin-bottom: 0.35rem !important;
                box-shadow: 0 5px 14px rgba(56, 189, 248, 0.12), 0 2px 10px rgba(246, 175, 207, 0.16) !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                min-height: 58px !important;
                padding: 0.2rem 0.28rem !important;
                border: 1px dashed #f6afcf !important;
                border-radius: 8px !important;
                background: linear-gradient(180deg, #ffffff 0%, #d7eeff 100%) !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
                padding: 0.05rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
                min-height: 1.45rem !important;
                padding: 0.05rem 0.4rem !important;
                font-size: 0.68rem !important;
                border-radius: 6px !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploader"] small {
                display: block !important;
                margin-top: 0.18rem !important;
                font-size: 0.58rem !important;
                line-height: 1.15 !important;
                color: #64748b !important;
            }

            [data-testid="stSidebar"] h3 {
                font-size: 0.84rem !important;
                margin-top: 0.25rem !important;
                margin-bottom: 0.2rem !important;
            }

            .sidebar-section-label {
                display: block !important;
                margin-top: 0.72rem !important;
                margin-bottom: 0.72rem !important;
                color: #0f6f8f !important;
                font-size: 0.64rem !important;
                font-weight: 900 !important;
                letter-spacing: 0.08em !important;
                text-transform: uppercase !important;
                line-height: 1.15 !important;
            }

            .workflow-label {
                margin-top: 0.7rem !important;
            }

            [data-testid="stSidebar"] .stButton > button {
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: flex-start !important;
                min-height: 2rem !important;
                margin-bottom: 0.36rem !important;
                padding: 0.25rem 0.65rem !important;
                border-radius: 9px !important;
                border: 1px solid #f6afcf !important;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(215, 238, 255, 0.6) 100%) !important;
                color: #153243 !important;
                font-size: 0.8rem !important;
                font-weight: 650 !important;
                line-height: 1.2 !important;
                text-align: left !important;
                box-shadow: 0 2px 8px rgba(56, 189, 248, 0.08), 0 1px 8px rgba(246, 175, 207, 0.12) !important;
            }

            [data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {
                width: 100% !important;
                display: block !important;
                text-align: left !important;
            }

            [data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] > div {
                width: 100% !important;
                display: flex !important;
                justify-content: flex-start !important;
                align-items: center !important;
                text-align: left !important;
            }

            [data-testid="stSidebar"] .stButton > button p {
                width: 100% !important;
                display: block !important;
                margin: 0 !important;
                text-align: left !important;
                justify-content: flex-start !important;
            }

            [data-testid="stSidebar"] .stButton > button *:not(svg):not(path):not([data-testid="stIconMaterial"]) {
                text-align: left !important;
                justify-content: flex-start !important;
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                border-color: #f6afcf !important;
                background: linear-gradient(135deg, #ffffff 0%, #cfefff 58%, #f9d2e3 100%) !important;
                box-shadow: 0 5px 14px rgba(56, 189, 248, 0.16), 0 3px 12px rgba(246, 175, 207, 0.18) !important;
            }

            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #3bb4f2 0%, #a8d8ff 46%, #f6afcf 100%) !important;
                border-color: #f6afcf !important;
                color: white !important;
                box-shadow: 0 5px 16px rgba(56, 189, 248, 0.24), 0 4px 14px rgba(246, 175, 207, 0.24) !important;
            }

            [data-testid="stSidebar"] .stButton > button[kind="primary"] * {
                color: white !important;
            }

            [data-testid="stFileUploader"] button {
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 0.25rem !important;
                white-space: nowrap !important;
            }

            [data-testid="stSidebar"] hr {
                margin: 0.07rem 0 0.55rem 0 !important;
                border-color: #f6afcf !important;
            }

            .sidebar-reset-note {
                color: #64748b !important;
                font-size: 0.6rem !important;
                line-height: 1.2 !important;
                margin-top: -0.05rem !important;
                margin-bottom: 0.7rem !important;
            }

            .workflow-status-row {
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
                gap: 0.35rem !important;
                min-height: 1.45rem !important;
                padding: 0.24rem 0.45rem !important;
                margin: 0.2rem 0 !important;
                background: rgba(255, 255, 255, 0.9) !important;
                border: 1px solid #f6afcf !important;
                border-radius: 8px !important;
                font-size: 0.68rem !important;
                line-height: 1.15 !important;
                box-shadow: 0 2px 8px rgba(56, 189, 248, 0.08), 0 1px 8px rgba(246, 175, 207, 0.1) !important;
            }

            .workflow-status-row span:first-child {
                min-width: 0 !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                white-space: nowrap !important;
            }

            .status-done,
            .status-pending {
                flex: 0 0 auto !important;
                border-radius: 999px !important;
                padding: 0.1rem 0.34rem !important;
                font-size: 0.54rem !important;
                font-weight: 800 !important;
                line-height: 1 !important;
            }

            .status-done {
                color: #065f46 !important;
                background: #bbf7d0 !important;
            }

            .status-pending {
                color: #991b1b !important;
                background: #fecaca !important;
            }

            .sticky-right-panel-marker {
                display: none !important;
            }

            .review-page-scroll-marker {
                display: none !important;
            }

            div[data-testid="stVerticalBlock"]:has(.review-page-scroll-marker),
            div[data-testid="stVerticalBlock"]:has(.review-page-scroll-marker) [data-testid="stVerticalBlock"],
            div[data-testid="stVerticalBlock"]:has(.review-page-scroll-marker) [data-testid="stExpander"],
            div[data-testid="stVerticalBlock"]:has(.review-page-scroll-marker) [data-testid="stExpander"] details {
                height: auto !important;
                min-height: auto !important;
                max-height: none !important;
                overflow: visible !important;
            }

            div[data-testid="stVerticalBlock"]:has(.sticky-right-panel-marker) {
                position: sticky !important;
                top: 1rem !important;
                max-height: calc(100vh - 2rem) !important;
                overflow-y: auto !important;
                padding-right: 0.25rem !important;
            }

            .page-title {
                margin-top: 0.65rem !important;
                margin-bottom: 0.45rem !important;
                color: #0f172a !important;
                font-family: "Times New Roman", Georgia, serif !important;
                font-size: 1.75rem !important;
                font-weight: 750 !important;
                line-height: 1.16 !important;
            }

            .page-title-note {
                color: #0f172a !important;
                font-size: inherit !important;
                font-weight: inherit !important;
                white-space: nowrap !important;
            }

            .section-title {
                margin-top: 0.85rem !important;
                margin-bottom: 0.35rem !important;
                color: #12364a !important;
                font-family: "Times New Roman", Georgia, serif !important;
                font-size: 1.18rem !important;
                font-weight: 750 !important;
                line-height: 1.22 !important;
            }

            .small-title {
                margin-top: 0.45rem !important;
                margin-bottom: 0.2rem !important;
                color: #334155 !important;
                font-family: "Times New Roman", Georgia, serif !important;
                font-size: 0.98rem !important;
                font-weight: 750 !important;
                line-height: 1.2 !important;
            }

            [data-testid="stExpander"] details summary p {
                font-size: 1rem !important;
                font-weight: 700 !important;
                text-align: left !important;
            }

            .discovery-equation-block {
                margin: 0.35rem 0 0.75rem 0 !important;
                padding: 0.78rem 0.9rem !important;
                background: #ffffff !important;
                border: 1px solid #e6eaf0 !important;
                border-left: 4px solid #94a3b8 !important;
                border-radius: 7px !important;
                color: #0f172a !important;
                font-family: "Cambria Math", "Times New Roman", "STIX Two Text", serif !important;
                font-size: 1.12rem !important;
                line-height: 1.58 !important;
                white-space: pre-wrap !important;
                overflow-x: auto !important;
                box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
            }

            [data-testid="stChatMessage"] .katex-display {
                max-width: 100% !important;
                overflow-x: auto !important;
                overflow-y: hidden !important;
                padding-bottom: 0.25rem !important;
            }

            [data-testid="stChatMessage"] .katex-display > .katex {
                white-space: nowrap !important;
            }

            [data-testid="stChatMessage"] pre {
                max-width: 100% !important;
                overflow-x: auto !important;
            }

            [data-testid="stChatMessage"] code {
                white-space: pre-wrap !important;
                overflow-wrap: anywhere !important;
            }

            .discovery-review-marker {
                display: none !important;
            }

            div[data-testid="stVerticalBlock"]:has(.discovery-review-marker) {
                font-family: "Times New Roman", Georgia, serif !important;
                color: #172033 !important;
            }

            div[data-testid="stVerticalBlock"]:has(.discovery-review-marker) h1 {
                display: none !important;
            }

            div[data-testid="stVerticalBlock"]:has(.discovery-review-marker) h2 {
                margin-top: 1.15rem !important;
                margin-bottom: 0.45rem !important;
                color: #0f3f56 !important;
                font-family: "Times New Roman", Georgia, serif !important;
                font-size: 1.48rem !important;
                line-height: 1.2 !important;
                font-weight: 750 !important;
            }

            div[data-testid="stVerticalBlock"]:has(.discovery-review-marker) h3 {
                margin-top: 0.95rem !important;
                margin-bottom: 0.32rem !important;
                color: #12364a !important;
                font-family: "Times New Roman", Georgia, serif !important;
                font-size: 1.18rem !important;
                line-height: 1.22 !important;
                font-weight: 750 !important;
            }

            div[data-testid="stVerticalBlock"]:has(.discovery-review-marker) p,
            div[data-testid="stVerticalBlock"]:has(.discovery-review-marker) li {
                font-family: "Times New Roman", Georgia, serif !important;
                font-size: 0.98rem !important;
                line-height: 1.42 !important;
            }

            div[data-testid="stVerticalBlock"]:has(.discovery-review-marker) strong {
                font-size: 0.98rem !important;
                font-weight: 750 !important;
            }

            div[data-testid="stVerticalBlock"]:has(.discovery-review-marker) table {
                font-family: "Times New Roman", Georgia, serif !important;
                font-size: 0.9rem !important;
                line-height: 1.25 !important;
                margin-top: 0.25rem !important;
                margin-bottom: 0.75rem !important;
            }

            div[data-testid="stVerticalBlock"]:has(.discovery-review-marker) hr {
                margin: 0.8rem 0 !important;
                border-color: #c7e3f3 !important;
            }

            [data-testid="stExpander"] {
                margin-top: 0.55rem !important;
                margin-bottom: 0.85rem !important;
                border-color: #e6eaf0 !important;
                background: #ffffff !important;
            }

            [data-testid="stExpander"] details summary {
                min-height: 2.1rem !important;
                align-items: center !important;
            }

            [data-testid="stTextArea"] textarea {
                line-height: 1.35 !important;
            }

            .voice-question-row {
                margin-top: 0.45rem !important;
                margin-bottom: 0.2rem !important;
                padding: 0.45rem 0.55rem !important;
                background: #ffffff !important;
                border: 1px solid #e6eaf0 !important;
                border-radius: 9px !important;
            }

            .voice-question-label {
                margin: 0 0 0.2rem 0 !important;
                color: #406377 !important;
                font-size: 0.86rem !important;
                font-weight: 700 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
