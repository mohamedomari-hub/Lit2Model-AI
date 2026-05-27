import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 24rem),
                    linear-gradient(180deg, #ffffff 0%, #eef8ff 100%) !important;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(
                    180deg,
                    #eef8ff 0%,
                    #dff4ff 48%,
                    #f4f0ff 100%
                ) !important;
                border-right: 1px solid #b9dff4 !important;
            }

            [data-testid="stSidebar"] > div:first-child {
                padding: 0.55rem 0.6rem 0.85rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.46rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploader"] {
                background: #ffffff !important;
                border: 1px solid #b9dff4 !important;
                border-radius: 10px !important;
                padding: 0.32rem !important;
                margin-bottom: 0.35rem !important;
                box-shadow: 0 4px 12px rgba(14, 116, 144, 0.08) !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                min-height: 58px !important;
                padding: 0.2rem 0.28rem !important;
                border: 1px dashed #b9dff4 !important;
                border-radius: 8px !important;
                background: #f8fdff !important;
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
                min-height: 2rem !important;
                margin-bottom: 0.36rem !important;
                padding: 0.25rem 0.65rem !important;
                border-radius: 9px !important;
                border: 1px solid #b9dff4 !important;
                background: rgba(255, 255, 255, 0.92) !important;
                color: #153243 !important;
                font-size: 0.8rem !important;
                font-weight: 650 !important;
                line-height: 1.2 !important;
                justify-content: flex-start !important;
                text-align: left !important;
                box-shadow: 0 2px 8px rgba(14, 116, 144, 0.05) !important;
            }

            [data-testid="stSidebar"] .stButton > button p,
            [data-testid="stSidebar"] .stButton > button div {
                width: 100% !important;
                margin: 0 !important;
                text-align: left !important;
                justify-content: flex-start !important;
            }

            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: linear-gradient(180deg, #2bb8e8 0%, #0ea5d8 100%) !important;
                border-color: #0ea5d8 !important;
                color: white !important;
                box-shadow: 0 4px 12px rgba(14, 165, 216, 0.24) !important;
            }

            [data-testid="stSidebar"] .stButton > button[kind="primary"] * {
                color: white !important;
            }

            [data-testid="stSidebar"] hr {
                margin: 0.07rem 0 0.55rem 0 !important;
                border-color: #b9dff4 !important;
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
                border: 1px solid #b9dff4 !important;
                border-radius: 8px !important;
                font-size: 0.68rem !important;
                line-height: 1.15 !important;
                box-shadow: 0 2px 8px rgba(14, 116, 144, 0.04) !important;
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

            div[data-testid="stVerticalBlock"]:has(.sticky-right-panel-marker) {
                position: sticky !important;
                top: 1rem !important;
                max-height: calc(100vh - 2rem) !important;
                overflow-y: auto !important;
                padding-right: 0.25rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
