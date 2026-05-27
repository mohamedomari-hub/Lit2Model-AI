"""
Sidebar navigation and workflow status display for the Streamlit app.
"""

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
    discovery_running = st.session_state.get("discovery_running", False)
    disable_navigation = (
        discovery_running
        and target_mode != "Run model discovery"
    )

    if st.sidebar.button(
        label,
        key=f"nav_{target_mode}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
        disabled=disable_navigation,
    ):
        st.session_state.mode = target_mode


def render_sidebar_navigation():
    if st.session_state.get("discovery_running", False):
        st.sidebar.warning(
            "Model discovery is running. Please do not switch workflow pages until it finishes."
        )

    st.sidebar.markdown("### Explore paper")
    nav_button("Ask paper questions", "Ask paper questions")

    st.sidebar.markdown("### Model workflow")
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
