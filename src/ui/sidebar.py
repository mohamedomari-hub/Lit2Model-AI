import os

import streamlit as st

from src.app.config import (
    DRAFT_REVIEWED_MODEL_PATH,
    FINAL_REVIEWED_MODEL_PATH,
    GENERATED_MODEL_PATH,
    REVIEW_PATH,
    SIMULATION_REQUIREMENTS_PATH,
)


def nav_button(label: str, target_mode: str):
    is_active = st.session_state.mode == target_mode

    if st.sidebar.button(
        label,
        key=f"nav_{target_mode}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.mode = target_mode


def render_sidebar_navigation():
    st.sidebar.markdown(
        '<div class="sidebar-section-label">Explore paper</div>',
        unsafe_allow_html=True,
    )
    nav_button("Ask paper questions", "Ask paper questions")

    st.sidebar.markdown(
        '<div class="sidebar-section-label workflow-label">Model workflow</div>',
        unsafe_allow_html=True,
    )
    nav_button("1. Run model discovery", "Run model discovery")
    nav_button("2. Review & Validate Model", "Review & Validate Model")
    nav_button("3. Simulation setup", "Simulation setup")
    nav_button("4. Generate Python Model", "Generate Python Model")


def render_workflow_status(project_paths=None):
    if project_paths is None:
        status_items = [
            ("Raw extraction", [REVIEW_PATH]),
            ("Draft review", [DRAFT_REVIEWED_MODEL_PATH]),
            ("Validated model", [FINAL_REVIEWED_MODEL_PATH]),
            ("Simulation setup", [SIMULATION_REQUIREMENTS_PATH]),
            ("Python model", [GENERATED_MODEL_PATH]),
        ]
    else:
        status_items = [
            ("Raw extraction", [project_paths["review_path"]]),
            ("Draft review", [project_paths["draft_reviewed_model_path"]]),
            (
                "Validated model",
                [
                    project_paths["final_reviewed_model_path"],
                    project_paths["final_reviewed_json_path"],
                ],
            ),
            ("Simulation setup", [project_paths["simulation_requirements_path"]]),
            ("Python model", [project_paths["generated_model_path"]]),
        ]

    st.sidebar.markdown("### Status")

    for label, paths in status_items:
        is_done = any(os.path.exists(path) for path in paths)
        status_class = "status-done" if is_done else "status-pending"
        status_text = "Done" if is_done else "Open"

        st.sidebar.markdown(
            f"""
            <div class="workflow-status-row">
                <span>{label}</span>
                <span class="{status_class}">{status_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )