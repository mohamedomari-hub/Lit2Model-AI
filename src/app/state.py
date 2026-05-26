import os
import shutil

import streamlit as st

from src.app.config import (
    DEBUG_EXTRACTION_PATH,
    DRAFT_REVIEWED_MODEL_PATH,
    FINAL_REVIEWED_MODEL_PATH,
    GENERATED_MODEL_PATH,
    HIGH_ACCURACY_OCR_DIR,
    MISSING_EQUATIONS_PATH,
    REVIEW_NOTES_PATH,
    REVIEW_PATH,
    SIMULATION_REQUIREMENTS_PATH,
)


def reset_workflow_state():
    workflow_files = [
        REVIEW_PATH,
        DEBUG_EXTRACTION_PATH,
        DRAFT_REVIEWED_MODEL_PATH,
        FINAL_REVIEWED_MODEL_PATH,
        REVIEW_NOTES_PATH,
        GENERATED_MODEL_PATH,
        SIMULATION_REQUIREMENTS_PATH,
        MISSING_EQUATIONS_PATH,
    ]

    for file_path in workflow_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    for dir_path in [
        "outputs/equation_candidates",
        "outputs/equation_pages",
        HIGH_ACCURACY_OCR_DIR,
    ]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    for key in [
        "reviewed_extraction",
        "review_editor_text",
        "pending_review_editor_text",
        "review_validated",
        "review_notes",
        "final_reviewed_model",
        "latest_scaffold",
        "latest_flowchart",
        "model_discovery_result",
        "generated_python_model",
        "generated_code_editor",
        "simulation_requirements",
        "validated_model_loaded_for_edit",
        "uploaded_review_model_name",
        "high_accuracy_ocr_result",
        "high_accuracy_ocr_cache_path",
        "missing_equation_ocr_candidates",
        "targeted_equation_ocr_enabled",
        "targeted_gpt4o_equation_ocr_enabled",
    ]:
        st.session_state.pop(key, None)

    st.session_state.reviewed_extraction = None
    st.session_state.review_editor_text = ""
    st.session_state.review_validated = False
    st.session_state.review_notes = ""
    st.session_state.final_reviewed_model = None
    st.session_state.latest_scaffold = None
    st.session_state.latest_flowchart = None
    st.session_state.model_discovery_result = None
    st.session_state.generated_python_model = None
    st.session_state.simulation_requirements = None
    st.session_state.validated_model_loaded_for_edit = False
    st.session_state.uploaded_review_model_name = None
    st.session_state.high_accuracy_ocr_result = None
    st.session_state.high_accuracy_ocr_cache_path = None
    st.session_state.missing_equation_ocr_candidates = {}
    st.session_state.targeted_equation_ocr_enabled = False
    st.session_state.targeted_gpt4o_equation_ocr_enabled = False
