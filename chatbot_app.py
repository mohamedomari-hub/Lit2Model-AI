import os
import re
import json
import shutil
import hashlib
import subprocess
import sys
import difflib
import glob
from datetime import datetime

from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

import streamlit as st
from dotenv import load_dotenv

from src.app.config import (
    DEBUG_EXTRACTION_PATH,
    DEFAULT_PDF_PATH,
    DRAFT_REVIEWED_MODEL_PATH,
    FINAL_REVIEWED_MODEL_PATH,
    GENERATED_MODEL_PATH,
    GENERATED_SCAFFOLD_PATH,
    HIGH_ACCURACY_OCR_DIR,
    MISSING_EQUATIONS_PATH,
    OUTPUT_DIR,
    REVIEW_NOTES_PATH,
    REVIEW_PATH,
    SIMULATION_REQUIREMENTS_PATH,
    UPLOAD_DIR,
    EXTRACTED_EVIDENCE_JSON_PATH,
    DRAFT_REVIEWED_JSON_PATH,
    FINAL_REVIEWED_JSON_PATH,

)

from src.discovery.run_model_discovery import run_controlled_discovery
from src.app.io import (
    append_latest_answer_to_review_draft,
    load_review_file,
    load_text_file,
    save_text_file,
)
from src.app.state import reset_workflow_state
from src.app.theme import apply_theme
from src.ingestion.pdf_parser import parse_pdf_multimodal
from src.ingestion.ocr import (
    extract_visible_equations_with_gpt,
)
from src.retrieval.vector_store import build_vector_store, load_vector_store, get_chroma_dir
from src.chat.chat_tools import (
    set_vector_store,
    set_active_pdf_path,
    propose_candidate_ode_model,
)
from src.chat.chat_agent import build_agent
from src.modelling.plan_simulations import infer_simulation_requirements
from src.modelling.generate_model import save_generated_python_model

from src.modelling.validate_model import parse_equations
from src.ui.renderers import (
    render_discovery_review,
    render_markdown_with_latex,
    render_mermaid,
    render_pdf_viewer,
    render_review_model_card_preview,
)
from src.ui.sidebar import render_sidebar_navigation, render_workflow_status
from src.discovery.discovery_pipeline import run_llm_first_discovery_markdown



load_dotenv()

client = OpenAI()

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    audio_path = "outputs/voice_question.wav"
    os.makedirs("outputs", exist_ok=True)

    with open(audio_path, "wb") as file:
        file.write(audio_bytes)

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

    return transcript.text


def text_to_speech(answer_text: str) -> str:
    audio_path = "outputs/voice_answer.mp3"
    os.makedirs("outputs", exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=answer_text[:4000],
    ) as response:
        response.stream_to_file(audio_path)

    return audio_path

LAST_ACTIVE_PDF_PATH = "outputs/last_active_pdf.txt"

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def save_last_active_pdf(pdf_path: str):
    os.makedirs("outputs", exist_ok=True)

    with open(LAST_ACTIVE_PDF_PATH, "w", encoding="utf-8") as file:
        file.write(pdf_path)


def load_last_active_pdf():
    if not os.path.exists(LAST_ACTIVE_PDF_PATH):
        return None

    with open(LAST_ACTIVE_PDF_PATH, "r", encoding="utf-8") as file:
        pdf_path = file.read().strip()

    if pdf_path and os.path.exists(pdf_path):
        return pdf_path

    return None


def get_pdf_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()

def get_project_dir(pdf_path: str, parser_mode: str) -> str:
    """
    Stable project directory based on PDF filename + parser.
    Avoids hash mismatches during development.
    """

    pdf_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    safe_pdf_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        pdf_name,
    )

    safe_parser = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        parser_mode,
    )

    return os.path.join(
        "outputs",
        "projects",
        f"{safe_pdf_name}_{safe_parser}"
    )


def clear_project_review_files_only():
    """
    Clear project review/discovery files but keep parser-generated assets
    such as equation crops and OCR cache.
    """

    paths = current_project_paths()

    files_to_remove = [
        paths["review_path"],
        paths["debug_extraction_path"],
        paths["draft_reviewed_model_path"],
        paths["final_reviewed_model_path"],
        paths["review_notes_path"],
        paths["generated_model_path"],
        paths["simulation_requirements_path"],
        paths["missing_equations_path"],
        paths["extracted_evidence_json_path"],
        paths["draft_reviewed_json_path"],
        paths["final_reviewed_json_path"],
    ]

    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)

    os.makedirs(paths["project_dir"], exist_ok=True)
    os.makedirs(paths["equation_candidates_dir"], exist_ok=True)
    os.makedirs(paths["equation_pages_dir"], exist_ok=True)
    os.makedirs(paths["equation_ocr_dir"], exist_ok=True)

def get_project_paths(pdf_path: str, parser_mode: str) -> dict:
    project_dir = get_project_dir(pdf_path, parser_mode)

    return {
        "project_dir": project_dir,
        "chroma_dir": os.path.join(project_dir, "chroma_db"),
        "review_path": os.path.join(project_dir, "extraction_review.md"),
        "debug_extraction_path": os.path.join(project_dir, "extraction_debug.md"),
        "draft_reviewed_model_path": os.path.join(project_dir, "reviewed_model_draft.md"),
        "final_reviewed_model_path": os.path.join(project_dir, "reviewed_model.md"),
        "review_notes_path": os.path.join(project_dir, "review_notes.md"),
        "generated_model_path": os.path.join(project_dir, "generated_model.py"),
        "simulation_requirements_path": os.path.join(project_dir, "simulation_requirements.json"),
        "missing_equations_path": os.path.join(project_dir, "missing_equations.json"),
        "extracted_evidence_json_path": os.path.join(project_dir, "extracted_evidence.json"),
        "draft_reviewed_json_path": os.path.join(project_dir, "reviewed_model_draft.json"),
        "final_reviewed_json_path": os.path.join(project_dir, "final_reviewed_model.json"),
        "equation_candidates_dir": os.path.join(project_dir, "equation_candidates"),
        "equation_pages_dir": os.path.join(project_dir, "equation_pages"),
        "ocr_dir": os.path.join(project_dir, "cache", "ocr"),
        "equation_ocr_dir": os.path.join(project_dir, "cache", "ocr", "equations"),
    }


def current_project_paths() -> dict:
    return get_project_paths(
        st.session_state.pdf_path,
        st.session_state.parser_mode,
    )

def clean_mermaid_flowchart(raw_text: str) -> str:
    """
    Convert extracted mechanism lines into styled Mermaid.

    States/compartments get one color.
    Processes/outputs get another color.
    Parameters/rate constants are avoided as boxes where possible.
    """

    def node_id(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", name.strip())

    def is_rate_or_parameter(name: str) -> bool:
        name_lower = name.strip().lower()

        parameter_like = {
            "ka", "ke", "keo", "ke1", "cl", "vd", "emax",
            "ca", "cb", "dose", "f"
        }

        return (
            name_lower in parameter_like
            or name_lower.startswith("k_")
            or name_lower.startswith("k")
            and len(name_lower) <= 4
        )

    def node_class(name: str) -> str:
        name_lower = name.strip().lower()

        if name_lower in {"c", "ce"}:
            return "state"

        if "compartment" in name_lower:
            return "state"

        if "dose" in name_lower or "dexamethasone" in name_lower:
            return "input"

        if (
            "glucose" in name_lower
            or "insulin" in name_lower
            or "glucagon" in name_lower
            or "secretion" in name_lower
            or "uptake" in name_lower
            or "production" in name_lower
            or "signaling" in name_lower
            or "glulv" in name_lower
            or "glumilk" in name_lower
            or "glucasec" in name_lower
        ):
            return "process"

        return "process"

    lines = []

    for line in raw_text.splitlines():
        line = line.strip()

        if not line or "-->" not in line:
            continue

        match_labeled = re.match(r"(.+?)-->\|(.+?)\|(.+)", line)
        match_unlabeled = re.match(r"(.+?)-->\s*(.+)", line)

        if match_labeled:
            source = match_labeled.group(1).strip()
            relation = match_labeled.group(2).strip()
            target = match_labeled.group(3).strip()

        elif match_unlabeled:
            source = match_unlabeled.group(1).strip()
            relation = "affects"
            target = match_unlabeled.group(2).strip()

        else:
            continue

        # Avoid drawing pure parameters/rates as standalone boxes
        if is_rate_or_parameter(source) or is_rate_or_parameter(target):
            continue

        source_id = node_id(source)
        target_id = node_id(target)

        source_class = node_class(source)
        target_class = node_class(target)

        lines.append(
            f'{source_id}["{source}"]:::{source_class} '
            f'-->|{relation}| '
            f'{target_id}["{target}"]:::{target_class}'
        )

    if not lines:
        return raw_text

    return (
        "flowchart TD\n"
        + "\n".join(lines)
        + """

classDef state fill:#DDEBFF,stroke:#2B5AA8,stroke-width:2px,color:#111;
classDef process fill:#EAF7EA,stroke:#2E7D32,stroke-width:2px,color:#111;
classDef input fill:#FFF3CD,stroke:#B8860B,stroke-width:2px,color:#111;
classDef review fill:#FDE2E2,stroke:#B00020,stroke-width:2px,color:#111;
"""
    )

def clear_output_files():
    """
    Clear only session/UI state files.
    Do NOT delete project-specific outputs.
    Project outputs are now isolated by PDF hash.
    """

    transient_files = [
        "outputs/agent_result.md",
        "outputs/sanity_check_report.md",
        "outputs/mechanism_graph.html",
        "outputs/simulation_plot.png",
        "outputs/simulation_results.csv",
    ]

    for file_path in transient_files:
        if os.path.exists(file_path):
            os.remove(file_path)

def clear_current_project_outputs():
    """
    Delete outputs only for the currently selected PDF/parser project.
    Also clears legacy/global workflow files used by sidebar status.
    """

    paths = current_project_paths()
    project_dir = paths["project_dir"]

    # Delete project folder
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)

    # Remove legacy/global workflow artifacts
    legacy_files = [
        REVIEW_PATH,
        DRAFT_REVIEWED_MODEL_PATH,
        FINAL_REVIEWED_MODEL_PATH,
        SIMULATION_REQUIREMENTS_PATH,
        GENERATED_MODEL_PATH,
    ]

    for file_path in legacy_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    # Recreate required directories
    os.makedirs(paths["project_dir"], exist_ok=True)
    os.makedirs(paths["equation_candidates_dir"], exist_ok=True)
    os.makedirs(paths["equation_pages_dir"], exist_ok=True)
    os.makedirs(paths["equation_ocr_dir"], exist_ok=True)

def show_project_mechanism_flowchart():
    """
    Show mechanism flowchart from the current project's JSON draft
    if available.
    """

    extracted_json = (
        load_json_file(PROJECT_DRAFT_JSON_PATH)
        or load_json_file(PROJECT_EVIDENCE_JSON_PATH)
    )

    if not extracted_json:
        st.info("No mechanism flowchart JSON found yet.")
        return

    mermaid_code = extracted_json.get("mechanism_flowchart", "")

    if not mermaid_code:
        st.info("No mechanism flowchart found in JSON.")
        return

    st.markdown("## Mechanism Flowchart")
    render_mermaid(
        clean_mermaid_flowchart(mermaid_code)
    )


def sync_global_discovery_artifacts_to_project():
    """
    Temporary compatibility bridge:
    Some discovery functions may still write to old global outputs/.
    After discovery, copy those artifacts into the current PDF project folder.
    """

    paths = current_project_paths()

    file_map = {
        "outputs/extracted_evidence.json": paths["extracted_evidence_json_path"],
        "outputs/reviewed_model_draft.json": paths["draft_reviewed_json_path"],
        "outputs/missing_equations.json": paths["missing_equations_path"],
    }

    for source, target in file_map.items():
        if os.path.exists(source):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(source, target)

    dir_map = {
        "outputs/equation_candidates": paths["equation_candidates_dir"],
        "outputs/equation_pages": paths["equation_pages_dir"],
    }

    for source_dir, target_dir in dir_map.items():
        if os.path.exists(source_dir):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)

def generate_compartment_flowchart(reviewed_text: str) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a scientific diagram generator.

Create a Mermaid flowchart from the reviewed mechanistic extraction.

Node rules:
- State variables, compartments, biomarkers, observed outputs, and biological entities may be nodes.
- Parameters, rate constants, clearances, coefficients, and effect functions should usually be edge labels, not nodes.
- Examples of edge labels: ka, ke, keo, CL, Emax, effect_dxm_gluca.
- Do not create separate boxes for rate constants unless the paper explicitly treats them as biological entities.

Color/style rules:
- Use Mermaid classDef.
- State/compartment nodes use class `state`.
- Process/output nodes use class `process`.
- External input/intervention nodes use class `input`.
- Uncertain nodes or edges use class `review`.

flowchart LR
    Dose["IM dexamethasone dose"]:::input -->|ka| C["C: central compartment"]:::state
    C -->|ke1 / keo| Ce["Ce: effect compartment"]:::state
    Ce -->|effect_dxm_gluca| GlucaSec["glucagon secretion"]:::process
    Ce -->|effect_dxm_bt| GlucoseUptake["glucose uptake"]:::process

    classDef state fill:#DDEBFF,stroke:#2B5AA8,stroke-width:2px,color:#111;
    classDef process fill:#EAF7EA,stroke:#2E7D32,stroke-width:2px,color:#111;
    classDef input fill:#FFF3CD,stroke:#B8860B,stroke-width:2px,color:#111;
    classDef review fill:#FDE2E2,stroke:#B00020,stroke-width:2px,color:#111;

REVIEWED EXTRACTION:
{reviewed_text}
"""

    result = llm.invoke(prompt)
    return result.content.strip()


def initialize_system(
    pdf_path: str,
    parser_mode: str = "pymupdf4llm",
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    current_pdf_hash = get_pdf_hash(pdf_path)
    project_paths = get_project_paths(
        pdf_path,
        parser_mode,
    )

    chroma_dir = project_paths["chroma_dir"]

    hash_file = os.path.join(chroma_dir, "current_pdf_hash.txt")

    previous_hash = None

    if os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as file:
            previous_hash = file.read().strip()

    if current_pdf_hash != previous_hash:
        print("New PDF detected. Rebuilding Chroma DB.")

        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)

        project_paths = get_project_paths(
            pdf_path,
            parser_mode,
        )

        documents = parse_pdf_multimodal(
            pdf_path,
            parser_mode=parser_mode,
            equation_candidates_dir=project_paths["equation_candidates_dir"],
        )

        vector_store = build_vector_store(
            documents,
            pdf_path=pdf_path,
            reset_db=False,
            parser_mode=parser_mode,
            chroma_dir=chroma_dir,
        )
        os.makedirs(chroma_dir, exist_ok=True)

        with open(hash_file, "w", encoding="utf-8") as file:
            file.write(current_pdf_hash)

    else:
        print("Same PDF detected. Reusing Chroma DB.")
        vector_store = load_vector_store(
            pdf_path,
            parser_mode=parser_mode,
            chroma_dir=chroma_dir,
        )
    
    paths = current_project_paths()

    if not glob.glob(os.path.join(paths["equation_candidates_dir"], "*.png")):
        print("No project equation crops found. Regenerating parser assets.")

        documents = parse_pdf_multimodal(
            pdf_path,
            parser_mode=parser_mode,
        )

    set_vector_store(vector_store)
    set_active_pdf_path(pdf_path)

    st.session_state.vector_store = vector_store

    return build_agent()


def looks_like_debug_extraction(text: str) -> bool:
    debug_markers = [
        "RAW EQUATION RETRIEVAL CONTEXT",
        "GEMINI OCR EQUATION CONTEXT",
        "PARAMETER EXTRACTION OUTPUT:",
        "EQUATION EXTRACTION OUTPUT:",
        "MECHANISM EXTRACTION OUTPUT:",
        "GRAPH EDGES OUTPUT:",
        "GRAPH GENERATION OUTPUT:",
        "Chunk ID:",
    ]

    return any(marker in text for marker in debug_markers)


def append_to_review_section(text: str, heading: str, addition: str) -> str:
    heading_pattern = re.compile(
        rf"^##\s+.*{re.escape(heading)}.*$",
        flags=re.IGNORECASE | re.MULTILINE
    )

    match = heading_pattern.search(text)

    if not match:
        return text.rstrip() + f"\n\n## {heading}\n{addition}\n"

    next_heading = re.search(
        r"^##\s+",
        text[match.end():],
        flags=re.MULTILINE
    )

    if next_heading:
        insert_pos = match.end() + next_heading.start()
        return (
            text[:insert_pos].rstrip()
            + f"\n\n{addition}\n\n"
            + text[insert_pos:].lstrip()
        )

    return text.rstrip() + f"\n\n{addition}\n"

def build_review_queue(review_text: str) -> list[dict]:
    """
    Build a simple deterministic review queue from the current draft.
    """

    queue = []

    # Missing parameters from markdown tables
    missing_pattern = re.compile(
        r"\|\s*([^|\n]+?)\s*\|\s*(missing|not reported)\s*\|\s*([^|\n]*?)\s*\|\s*missing\s*\|",
        flags=re.IGNORECASE,
    )

    for match in missing_pattern.finditer(review_text):
        symbol = match.group(1).strip()
        unit = match.group(3).strip()

        queue.append(
            {
                "type": "missing_parameter",
                "label": f"Missing parameter: {symbol}",
                "symbol": symbol,
                "unit": unit,
                "action": "add_or_update_parameter",
            }
        )
    
    # Inputs marked missing / not reported
    input_sections = re.findall(
        r"\*\*Inputs\*\*(.*?)(?:\n\*\*|\n---|\Z)",
        review_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    for section in input_sections:
        if "not reported" in section.lower():
            queue.append(
                {
                    "type": "missing_input",
                    "label": "Missing or unclear input",
                    "symbol": "input",
                    "unit": "",
                    "action": "recover_input",
                }
            )

    # OCR candidate blocks
    if "OCR Candidate" in review_text:
        queue.append(
            {
                "type": "ocr_candidate",
                "label": "OCR candidate found in draft",
                "symbol": "OCR",
                "unit": "",
                "action": "review_ocr_candidate",
            }
        )

    # Equations requiring review
    suspicious_terms = [
        "requires review",
        "requires_review",
        "[UNCLEAR]",
        "OCR confidence:\nlow",
        "not validated",
    ]

    for term in suspicious_terms:
        if term.lower() in review_text.lower():
            queue.append(
                {
                    "type": "requires_review",
                    "label": f"Requires review: {term}",
                    "symbol": term,
                    "unit": "",
                    "action": "manual_review",
                }
            )


    # Suspicious / review-needed equations
    equation_lines = []

    for line in review_text.splitlines():
        line_clean = line.strip()

        if not line_clean:
            continue

        looks_like_equation = (
            "=" in line_clean
            or "\\frac" in line_clean
            or "d/dt" in line_clean
            or "\\frac{d}" in line_clean
        )

        if looks_like_equation:
            equation_lines.append(line_clean)

    suspicious_equation_markers = [
        "missing",
        "not reported",
        "unclear",
        "requires review",
        "requires_review",
        "ocr",
        "candidate",
        "effectdxm",
        "\\frac",
        "^",
    ]

    for eq in equation_lines:
        eq_lower = eq.lower()

        if any(marker in eq_lower for marker in suspicious_equation_markers):
            queue.append(
                {
                    "type": "equation_review",
                    "label": f"Equation requires review: {eq[:90]}",
                    "symbol": eq,
                    "unit": "",
                    "action": "review_or_replace_equation",
                }
            )

    # Remove duplicates by label
    unique = {}
    for item in queue:
        unique[item["label"]] = item

    return list(unique.values())


def list_equation_crop_paths():
    """
    Return equation crop paths only for the current active project/PDF.
    Prevents showing crops from old papers.
    """

    paths = current_project_paths()
    crop_paths = []

    project_crop_dirs = [
        paths.get("equation_candidates_dir"),
        paths.get("equation_pages_dir"),
    ]

    for crop_dir in project_crop_dirs:

        if crop_dir and os.path.exists(crop_dir):
            found = glob.glob(os.path.join(crop_dir, "*.png"))
            crop_paths.extend(found)

    return sorted(set(crop_paths))

def show_workflow_diagram():
    """
    Display Lit2Model-AI workflow diagram.
    """

    st.subheader("Lit2Model-AI Workflow")

    st.graphviz_chart("""
    digraph {
        rankdir=LR;

        node [shape=box, style="rounded"];

        "Upload PDF" -> "Parse + Index";
        "Parse + Index" -> "Run Model Discovery";
        "Run Model Discovery" -> "Review & Validate Model";
        "Review & Validate Model" -> "Simulation Setup";
        "Simulation Setup" -> "Generate Python Model";
        "Generate Python Model" -> "Run / Diagnose Simulation";
        "Run / Diagnose Simulation" -> "Review & Validate Model";
    }
    """)

def load_missing_equations():
    if not os.path.exists(MISSING_EQUATIONS_PATH):
        return []

    with open(MISSING_EQUATIONS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def load_json_file(path: str):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def initialize_review_json_draft():
    paths = current_project_paths()

    draft_path = paths["draft_reviewed_json_path"]
    extracted_path = paths["extracted_evidence_json_path"]

    if os.path.exists(draft_path):
        return load_json_file(draft_path)

    extracted = load_json_file(extracted_path)

    if extracted is None:
        return None

    save_json_file(draft_path, extracted)

    return extracted
# ---------------------------------------
# ADD BATCH 1 HERE
# ---------------------------------------

def build_review_queue_from_json(data: dict | None) -> list[dict]:
    if not data:
        return []

    queue = []

    for ode_index, ode in enumerate(data.get("odes", [])):
        state = ode.get("state", f"ode_{ode_index}")

        for param_index, param in enumerate(ode.get("parameters", [])):
            status = str(param.get("status", "")).lower()
            value = str(param.get("value", "")).lower()

            if status == "missing" or value in {"missing", "not reported", ""}:
                queue.append({
                    "type": "missing_parameter",
                    "location": "ode",
                    "ode_index": ode_index,
                    "param_index": param_index,
                    "label": f"Missing parameter `{param.get('symbol')}` in ODE `{state}`",
                    "symbol": param.get("symbol"),
                })

    for module_index, module in enumerate(data.get("process_modules", [])):
        module_name = module.get("name", f"module_{module_index}")

        for equation_index, equation in enumerate(module.get("equations", [])):

            if isinstance(equation, dict):
                equation_text = str(equation.get("equation", ""))
                equation_status = str(equation.get("status", "")).lower()
                equation_review = str(equation.get("review", "")).lower()
            else:
                equation_text = str(equation)
                equation_status = ""
                equation_review = ""

            equation_lower = equation_text.lower()

            already_reviewed = (
                equation_status in {
                    "human_reviewed",
                    "ocr_reviewed",
                    "reported",
                    "corrected",
                }
                or equation_review in {
                    "human_updated",
                    "human_reviewed",
                }
            )

            suspicious = (
                not already_reviewed
                and (
                    "requires_review" in equation_lower
                    or "not reported" in equation_lower
                    or "missing" in equation_lower
                    or equation_lower.strip().startswith("effectdxm")
                    or equation_lower.strip().startswith("effect_{dxm")
                )
            )

            if suspicious:
                queue.append({
                    "type": "equation_review",
                    "location": "process_module",
                    "module_index": module_index,
                    "equation_index": equation_index,
                    "label": (
                                f"Review equation {equation_index} in `{module_name}`: "
                                f"{equation_text[:80]}"
                            ),
                    "symbol": equation_text,
                })

        for param_index, param in enumerate(module.get("parameters", [])):
            status = str(param.get("status", "")).lower()
            value = str(param.get("value", "")).lower()

            if status == "missing" or value in {"missing", "not reported", ""}:
                queue.append({
                    "type": "missing_parameter",
                    "location": "process_module",
                    "module_index": module_index,
                    "param_index": param_index,
                    "label": f"Missing parameter `{param.get('symbol')}` in `{module_name}`",
                    "symbol": param.get("symbol"),
                })

    return queue

def update_json_parameter(
    data: dict,
    item: dict,
    value: str,
    unit: str,
    status: str,
    formula: str = "",
) -> dict:
    data = json.loads(json.dumps(data))

    location = item.get("location")

    if location == "ode":
        param = data["odes"][item["ode_index"]]["parameters"][item["param_index"]]

    elif location == "process_module":
        param = data["process_modules"][item["module_index"]]["parameters"][item["param_index"]]

    else:
        return data

    param["value"] = value
    param["unit"] = unit
    param["status"] = status

    if formula:
        param["formula"] = formula

    param["review"] = "human_updated"

    return data

def apply_ocr_candidate_to_json(
    data: dict,
    item: dict,
    equation_text: str,
) -> dict:
    data = json.loads(json.dumps(data))

    location = item.get("location")

    equation_record = {
        "equation": equation_text.strip(),
        "status": "ocr_reviewed",
        "review": "human_updated",
        "source": "ocr_candidate",
    }

    if location == "process_module":
        module = data["process_modules"][item["module_index"]]
        equation_index = item["equation_index"]

        module["equations"][equation_index] = equation_record

    elif location == "ode":
        ode = data["odes"][item["ode_index"]]
        ode["equation"] = equation_text.strip()
        ode["equation_status"] = "ocr_reviewed"
        ode["review"] = "human_updated"

    return data

def add_json_input_to_ode(
    data: dict,
    ode_index: int,
    symbol: str,
    value: str,
    unit: str,
    meaning: str,
) -> dict:
    data = json.loads(json.dumps(data))

    ode = data["odes"][ode_index]

    if "inputs" not in ode or ode["inputs"] is None:
        ode["inputs"] = []

    ode["inputs"].append(
        {
            "symbol": symbol,
            "value": value,
            "unit": unit,
            "meaning": meaning,
            "review": "human_added",
        }
    )

    return data

def get_current_review_text_for_append() -> str:
    """
    Return the best available review text before appending OCR/manual edits.
    Priority:
    1. editor text
    2. saved draft
    3. raw extraction review
    4. latest discovery result in session
    """

    candidates = [
        st.session_state.get("review_editor_text", ""),
        load_text_file(DRAFT_REVIEWED_MODEL_PATH),
        load_review_file(),
        st.session_state.get("model_discovery_result", ""),
    ]

    for text in candidates:
        if text and text.strip():
            return text

    return ""

def format_ocr_candidate_record(
    equation_number: str,
    image_path: str,
    method: str,
    result: str,
    cache_path: str,
):
    return f"""#### Equation {equation_number or "unknown"} OCR Candidate
- Status: OCR candidate generated; not validated
- Source page:
- Method/source: {method}
- Candidate crop: {image_path}
- Cache file: {cache_path}
- Requires review: true
- Candidate equation:

```text
{result}
```

- Review notes: OCR candidate accepted into draft, not yet validated.
"""

def extract_equation_text_from_ocr_record(text: str) -> str:
    """
    Extract the useful equation body from an OCR candidate record.
    Keeps the function conservative: if it cannot detect a clean block,
    it returns the original text.
    """

    if not text:
        return ""

    # Prefer fenced code block content
    code_block_match = re.search(
        r"```(?:text|latex|math)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if code_block_match:
        return code_block_match.group(1).strip()

    # Fallback: remove obvious metadata lines
    lines = []

    skip_prefixes = (
        "###",
        "####",
        "- Status:",
        "- Source page:",
        "- Method/source:",
        "- Candidate crop:",
        "- Cache file:",
        "- Requires review:",
        "- Candidate equation:",
        "- Review notes:",
    )

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(skip_prefixes):
            continue

        lines.append(line)

    cleaned = "\n".join(lines).strip()

    return cleaned if cleaned else text.strip()

def split_ocr_equations(clean_text: str) -> list[str]:
    """
    Split OCR output into separate equations when one crop contains multiple equations.
    """

    if not clean_text:
        return []

    parts = re.split(
        r"\n\s*(?=Equation\s*\(\d+\)\s*:)",
        clean_text,
        flags=re.IGNORECASE,
    )

    equations = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    return equations if equations else [clean_text.strip()]

def run_cached_visible_equations_ocr(
    image_path: str,
    model: str = "gpt-4o-mini",
):
    project_paths = current_project_paths()
    ocr_dir = project_paths["equation_ocr_dir"]

    os.makedirs(ocr_dir, exist_ok=True)

    with open(image_path, "rb") as file:
        image_hash = hashlib.md5(file.read()).hexdigest()

    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)

    cache_path = os.path.join(
        ocr_dir,
        (
            f"{os.path.basename(image_path)}"
            f"_visible_equations_"
            f"{safe_model}_"
            f"{image_hash[:12]}.md"
        )
    )

    if os.path.exists(cache_path):
        return load_text_file(cache_path), cache_path, True

    result = extract_visible_equations_with_gpt(
        image_path=image_path,
        model=model,
    )

    ocr_record = format_ocr_candidate_record(
        equation_number="visible equations",
        image_path=image_path,
        method=f"{model} visible-equations OCR",
        result=result,
        cache_path=cache_path,
    )

    save_text_file(cache_path, ocr_record)

    return ocr_record, cache_path, False

def load_cached_ocr_candidates_for_crop(
    image_path: str,
    equation_number: str,
):
    project_paths = current_project_paths()
    ocr_dir = project_paths["equation_ocr_dir"]

    if not image_path or not os.path.exists(ocr_dir):
        return []

    image_basename = os.path.basename(image_path)
    safe_equation_number = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        equation_number or "unknown"
    )
    cache_paths = glob.glob(
        os.path.join(ocr_dir, f"{image_basename}*.md")
    )

    candidates = []

    for cache_path in cache_paths:
        cache_name = os.path.basename(cache_path)
        is_visible_candidate = "_visible_equations_" in cache_name
        is_selected_equation_candidate = (
            f"_eq_{safe_equation_number}_" in cache_name
        )

        if not is_visible_candidate and not is_selected_equation_candidate:
            continue

        if "_visible_equations_" in cache_name:

            cache_lower = cache_name.lower()

            if "gpt-4o-mini" in cache_lower:
                method = "gpt-4o-mini OCR"

            elif (
                "gpt-4o-mini" in cache_lower
                or "gpt_4_1_mini" in cache_lower
            ):
                method = "gpt-4o-mini OCR"

            elif (
                "gpt-4o" in cache_lower
                or "gpt_4o" in cache_lower
            ):
                method = "gpt-4o OCR"

            else:
                method = "visible-equations OCR"

            priority = 0
        elif "gpt-4o-mini" in cache_name or "gpt_4o" in cache_name:
            method = "gpt-4o-mini OCR"
            priority = 1
        elif "pix2tex" in cache_name:
            method = "local pix2tex OCR"
            priority = 2
        else:
            method = "cached OCR"
            priority = 9

        candidates.append(
            {
                "label": f"{method} ({cache_name})",
                "text": load_text_file(cache_path),
                "cache_path": cache_path,
                "method": method,
                "priority": priority,
            }
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["priority"],
            candidate["cache_path"],
        )
    )


def apply_simple_review_command(command: str, current_text: str) -> tuple[str, str]:
    command_clean = command.strip()
    command_lower = command_clean.lower()

    def append_to_section(text: str, heading: str, addition: str) -> str:
        heading_pattern = re.compile(
            rf"^##\s+.*{re.escape(heading)}.*$",
            flags=re.IGNORECASE | re.MULTILINE
        )

        match = heading_pattern.search(text)

        if not match:
            return text.rstrip() + f"\n\n## {heading}\n{addition}\n"

        next_heading = re.search(
            r"^##\s+",
            text[match.end():],
            flags=re.MULTILINE
        )

        if next_heading:
            insert_pos = match.end() + next_heading.start()
            return (
                text[:insert_pos].rstrip()
                + f"\n\n{addition}\n\n"
                + text[insert_pos:].lstrip()
            )

        return text.rstrip() + f"\n\n{addition}\n"

    if command_lower.startswith("remove "):
        target = command_clean[7:].strip()

        if target in current_text:
            updated = current_text.replace(target, "")
            return updated, f"Removed: `{target}`"

        return current_text, f"I could not find exact text to remove: `{target}`"

    if command_lower.startswith("replace ") and " with " in command_lower:
        body = command_clean[8:]
        split_match = re.search(r"\s+with\s+", body, flags=re.IGNORECASE)

        if split_match is None:
            return current_text, "Use: `replace <old> with <new>`."

        old = body[:split_match.start()]
        new = body[split_match.end():]

        old = old.strip()
        new = new.strip()

        if old in current_text:
            updated = current_text.replace(old, new)
            return updated, f"Replaced:\n\n`{old}`\n\nwith:\n\n`{new}`"

        return current_text, f"I could not find exact text to replace: `{old}`"

    if command_lower.startswith("add note:"):
        addition = command_clean.split(":", 1)[1].strip()
        updated = append_to_section(
            current_text,
            "Missing / Needs Review",
            f"- {addition}"
        )
        return updated, "Added note to human review notes."

    if command_lower.startswith("add to ") and ":" in command_clean:
        header_part, addition = command_clean.split(":", 1)
        section_name = header_part[7:].strip().lower()
        addition = addition.strip()

        if section_name == "equations":
            formatted_addition = (
                "(eq_added) Added equation\n"
                "status: needs review\n"
                "source: not reported\n\n"
                "equation:\n"
                f"`{addition}`\n\n"
                "notes:\n"
                "added from review chat\n"
            )
            updated = append_to_section(
                current_text,
                "Equations",
                formatted_addition
            )
            return updated, "Added to equations."

        section_headings = {
            "parameters": "Parameters",
            "mechanisms": "Mechanisms",
        }

        if section_name in section_headings:
            updated = append_to_section(
                current_text,
                section_headings[section_name],
                f"- {addition}"
            )
            return updated, f"Added to {section_name}."

        updated = append_to_section(
            current_text,
            section_name,
            f"- {addition}"
        )
        return updated, f"Added to section `{section_name}`."

    return current_text, (
        "I did not recognize this as an edit command. "
        "Try: `add to equations: ...`, `add to parameters: ...`, "
        "`add to mechanisms: ...`, `add note: ...`, "
        "`replace ... with ...`, or `remove ...`."
    )


def scientific_review_edit(review_command: str, current_extraction: str) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a scientific model review assistant.

CURRENT REVIEWED EXTRACTION:
{current_extraction}

USER CORRECTION:
{review_command}

Your task:
Update the reviewed extraction conservatively.

Rules:
- Preserve existing validated content.
- Apply only the requested scientific correction.
- Preserve section structure and markdown formatting.
- Do not delete unrelated content.
- Do not invent biology, parameters, or equations.
- If the user correction is ambiguous, add a short "Requires human review" note rather than guessing.
- If the user provides an equation in plain notation, convert it into clean readable mathematical notation.
- Accept plain scientific notation such as:
  * Ce^10
  * Emax * Ce
  * dCe/dt
  * ka * exp(-ka*t)
  * dose * F * ka * exp(-ka*t) - CL*C

- ALWAYS format equations using display LaTeX with dollar blocks.

Example:
$$
\\frac{{dC_e}}{{dt}}
=
k_{{e1}} C - k_{{eo}} C_e
$$

- Never write equations as markdown text using underscores.
- Never output equations like:
_Ce_ = _C_ * _ke_1 - _keo_ * _Ce_
- Preserve mathematical notation.
- Mark OCR-derived or corrected equations as requiring human review unless the user explicitly says they verified them.
- If the user says an equation is "weird", "wrong", "unclear", or "suspicious" without providing the corrected equation, do NOT rewrite the equation.
- Instead, keep the original equation unchanged and add "(requires human review / OCR uncertain)".

Return ONLY the updated reviewed extraction text.
"""

    result = llm.invoke(prompt)
    return result.content


def sanity_check_reviewed_extraction(reviewed_text: str) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a scientific consistency checker for mechanistic modelling papers.

Check the reviewed extraction for:
- contradictions between mechanisms
- equations inconsistent with mechanisms
- graph edges with unsupported directionality
- parameter/state/quantity misclassification
- reported vs derived vs missing parameter errors
- OCR-derived equations that should require human review
- overconfident claims
- missing information needed for simulation

Do NOT rewrite the full extraction.
Do NOT invent missing facts.

Return exactly:

1. PASS_OR_REVIEW
- PASS if the extraction is scientifically coherent enough for a draft scaffold.
- REVIEW if issues need human attention.

2. Issues found
- Bullet list.

3. Suggested corrections
- Concrete edits the user can apply.

4. Safe-to-use summary
- What parts seem reliable.

REVIEWED EXTRACTION:
{reviewed_text}
"""

    result = llm.invoke(prompt)
    return result.content


def generate_final_from_review(reviewed_text: str) -> str:
    sanity_report = sanity_check_reviewed_extraction(reviewed_text)

    final_answer = propose_candidate_ode_model.invoke({
        "extracted_summary": f"""
SCIENTIFIC SANITY CHECK REPORT:
{sanity_report}

REVIEWED EXTRACTION:
{reviewed_text}

Instruction:
Use the reviewed extraction as the source of truth.
Use the sanity-check report to add human-review notes.
Do not invent information.
If the sanity checker marked an item as problematic, do not present it as fully validated.
"""
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open("outputs/sanity_check_report.md", "w", encoding="utf-8") as file:
        file.write(sanity_report)

    with open(GENERATED_SCAFFOLD_PATH, "w", encoding="utf-8") as file:
        file.write(final_answer)

    return final_answer


# --------------------------------------------------
# Streamlit setup
# --------------------------------------------------

st.set_page_config(
    page_title="Lit2Model-AI",
    page_icon="🧬",
    layout="wide"
)


apply_theme()

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        display: none;
    }

    .block-container {
        padding-top: 0rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)





# --------------------------------------------------
# PDF upload
# --------------------------------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)
st.sidebar.markdown("### Upload paper")

uploaded_file = st.sidebar.file_uploader(
    "PDF file",
    type=["pdf"],
    label_visibility="collapsed",
)

st.sidebar.markdown("### Parser")

parser_mode = st.sidebar.selectbox(
    "PDF parser",
    [
        "docling",
        "pymupdf4llm",
        "pymupdf_fast",
    ],
    index=0,
    label_visibility="collapsed",
    help=(
        "Docling = best scientific parser (recommended). "
        "PyMuPDF4LLM = fast baseline. "
        "PyMuPDF_fast = lightweight fallback."
    ),
)


if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = load_last_active_pdf() or DEFAULT_PDF_PATH


if "parser_mode" not in st.session_state:
    st.session_state.parser_mode = parser_mode

if st.session_state.parser_mode != parser_mode:
    st.session_state.parser_mode = parser_mode

    clear_output_files()

    st.session_state.pop("agent", None)
    st.session_state.pop("vector_store", None)
    st.session_state.pop("reviewed_extraction", None)
    st.session_state.pop("model_discovery_result", None)
    st.session_state.pop("review_editor_text", None)
    st.session_state.pop("latest_scaffold", None)
    st.session_state.pop("latest_flowchart", None)

    st.info(f"Parser changed to: {parser_mode}. Rebuilding parser-specific vector DB.")
    st.rerun()

if uploaded_file is not None:

    uploaded_bytes = uploaded_file.getbuffer()
    uploaded_hash = hashlib.md5(uploaded_bytes).hexdigest()

    uploaded_pdf_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name,
    )

    upload_signature = f"{uploaded_file.name}_{uploaded_hash}"

    if st.session_state.get("upload_signature") != upload_signature:

        with open(uploaded_pdf_path, "wb") as file:
            file.write(uploaded_bytes)

        st.session_state.upload_signature = upload_signature
        st.session_state.pdf_path = uploaded_pdf_path
        save_last_active_pdf(uploaded_pdf_path)

        set_active_pdf_path(uploaded_pdf_path)

        st.session_state.pop("vector_store", None)
        set_vector_store(None)

        clear_output_files()

        for key in [
            "agent",
            "reviewed_extraction",
            "messages",
            "review_chat_messages",
            "final_reviewed_model",
            "latest_scaffold",
            "latest_flowchart",
            "model_discovery_result",
            "generated_python_model",
            "generated_code_editor",
            "simulation_requirements",
            "review_editor_text",
            "pending_review_editor_text",
            "review_validated",
            "review_notes",
            "latest_paper_answer",
            "validated_model_loaded_for_edit",
            "uploaded_review_model_name",
            "high_accuracy_ocr_result",
            "high_accuracy_ocr_cache_path",
            "missing_equation_ocr_candidates",
            "targeted_equation_ocr_enabled",
            "targeted_gpt4o_equation_ocr_enabled",
        ]:
            st.session_state.pop(key, None)

        uploaded_chroma_dir = get_chroma_dir(
            uploaded_pdf_path,
            parser_mode=st.session_state.parser_mode,
        )

        if os.path.exists(uploaded_chroma_dir):
            shutil.rmtree(uploaded_chroma_dir)

        st.success(f"Uploaded: {uploaded_file.name}")
        st.rerun()

# --------------------------------------------------
# Initialize system
# --------------------------------------------------

if "agent" not in st.session_state:
    with st.spinner("Loading paper and vector database..."):
        st.session_state.agent = initialize_system(
            st.session_state.pdf_path,
            parser_mode=st.session_state.parser_mode,
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "review_chat_messages" not in st.session_state:
    st.session_state.review_chat_messages = []

if "reviewed_extraction" not in st.session_state:
    st.session_state.reviewed_extraction = None

if "final_reviewed_model" not in st.session_state:
    st.session_state.final_reviewed_model = None

if "latest_scaffold" not in st.session_state:
    st.session_state.latest_scaffold = None

if "latest_flowchart" not in st.session_state:
    st.session_state.latest_flowchart = None

if "model_discovery_result" not in st.session_state:
    st.session_state.model_discovery_result = None

if "generated_python_model" not in st.session_state:
    st.session_state.generated_python_model = None

if "simulation_requirements" not in st.session_state:
    st.session_state.simulation_requirements = None

if "review_editor_text" not in st.session_state:
    st.session_state.review_editor_text = ""

if "review_validated" not in st.session_state:
    st.session_state.review_validated = os.path.exists(
        FINAL_REVIEWED_MODEL_PATH
    )

if "review_notes" not in st.session_state:
    st.session_state.review_notes = load_text_file(REVIEW_NOTES_PATH) or ""

if "latest_paper_answer" not in st.session_state:
    st.session_state.latest_paper_answer = None

if "validated_model_loaded_for_edit" not in st.session_state:
    st.session_state.validated_model_loaded_for_edit = False

if "uploaded_review_model_name" not in st.session_state:
    st.session_state.uploaded_review_model_name = None

if "high_accuracy_ocr_result" not in st.session_state:
    st.session_state.high_accuracy_ocr_result = None

if "high_accuracy_ocr_cache_path" not in st.session_state:
    st.session_state.high_accuracy_ocr_cache_path = None

if "missing_equation_ocr_candidates" not in st.session_state:
    st.session_state.missing_equation_ocr_candidates = {}

if "targeted_equation_ocr_enabled" not in st.session_state:
    st.session_state.targeted_equation_ocr_enabled = False

if "targeted_gpt4o_equation_ocr_enabled" not in st.session_state:
    st.session_state.targeted_gpt4o_equation_ocr_enabled = False

if "pending_review_editor_text" in st.session_state:
    st.session_state.review_editor_text = st.session_state.pop(
        "pending_review_editor_text"
    )

if "mode" not in st.session_state:
    st.session_state.mode = "Ask paper questions"

PROJECT_PATHS = current_project_paths()

PROJECT_REVIEW_PATH = PROJECT_PATHS["review_path"]
PROJECT_DRAFT_JSON_PATH = PROJECT_PATHS["draft_reviewed_json_path"]
PROJECT_FINAL_JSON_PATH = PROJECT_PATHS["final_reviewed_json_path"]
PROJECT_EVIDENCE_JSON_PATH = PROJECT_PATHS["extracted_evidence_json_path"]
PROJECT_SIMULATION_PATH = PROJECT_PATHS["simulation_requirements_path"]
PROJECT_GENERATED_MODEL_PATH = PROJECT_PATHS["generated_model_path"]
PROJECT_MISSING_EQUATIONS_PATH = PROJECT_PATHS["missing_equations_path"]
PROJECT_DRAFT_MODEL_PATH = PROJECT_PATHS["draft_reviewed_model_path"]
PROJECT_FINAL_MODEL_PATH = PROJECT_PATHS["final_reviewed_model_path"]
PROJECT_REVIEW_NOTES_PATH = PROJECT_PATHS["review_notes_path"]
PROJECT_OCR_DIR = PROJECT_PATHS["equation_ocr_dir"]

os.makedirs(PROJECT_PATHS["project_dir"], exist_ok=True)
os.makedirs(PROJECT_PATHS["equation_candidates_dir"], exist_ok=True)
os.makedirs(PROJECT_PATHS["equation_pages_dir"], exist_ok=True)
os.makedirs(PROJECT_PATHS["equation_ocr_dir"], exist_ok=True)



# --------------------------------------------------
# Sidebar
# --------------------------------------------------

render_sidebar_navigation()

st.sidebar.divider()
if st.sidebar.button("Clear current project", use_container_width=True):
    clear_current_project_outputs()
    st.session_state.workflow_reset_notice = True
    st.rerun()

    st.session_state.pop("reviewed_extraction", None)
    st.session_state.pop("model_discovery_result", None)
    st.session_state.pop("review_editor_text", None)
    st.session_state.pop("latest_scaffold", None)
    st.session_state.pop("latest_flowchart", None)
    st.session_state.pop("generated_python_model", None)
    st.session_state.pop("generated_code_editor", None)
    st.session_state.pop("simulation_requirements", None)
    st.session_state.pop("missing_equation_ocr_candidates", None)

    st.session_state.workflow_reset_notice = True
    st.success("Current PDF project cleared.")
    st.rerun()

if st.session_state.get("workflow_reset_notice"):
    st.sidebar.markdown(
        '<div class="sidebar-reset-success">Workflow reset successfully.</div>',
        unsafe_allow_html=True,
    )




st.sidebar.markdown("---")
mode = st.session_state.mode

if mode == "Ask paper questions":

    chat_col,  pdf_col = st.columns([0.85, 1.15], gap="large")
    # ==========================
    # RIGHT COLUMN: PDF ONLY
    # ==========================
    with pdf_col:
        # Keep PDF viewer visible while left content scrolls.
        st.markdown(
            '<span class="sticky-right-panel-marker"></span>',
            unsafe_allow_html=True,
        )

        pdf_zoom = st.slider(
            "PDF zoom",
            min_value=1.0,
            max_value=2.2,
            value=1.35,
            step=0.05,
            key="paper_viewer_zoom",
        )

        render_pdf_viewer(
            st.session_state.pdf_path,
            zoom=pdf_zoom
        )

    # ==========================
    # Left COLUMN: APP + CHAT
    # ==========================
    with chat_col:
        st.title("🧬 Lit2Model-AI")
        st.caption(
            "Scientific paper chatbot + mechanistic model discovery assistant"
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                render_markdown_with_latex(message["content"])

        prompt = None


        audio = mic_recorder(
            start_prompt="🎙️ Start recording",
            stop_prompt="⏹️ Stop recording",
            just_once=True,
            use_container_width=True,
            key="voice_question_recorder",
        )

        if audio and "bytes" in audio:
            with st.spinner("Transcribing voice question..."):
                prompt = transcribe_audio_bytes(audio["bytes"])

            st.success(f"You asked: {prompt}")

        text_prompt = st.chat_input(
            "Ask questions about the paper, i.e. equations, tables, figure, or mechanism ..."

        )

        if text_prompt:
            prompt = text_prompt

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with chat_col:
            with st.chat_message("user"):
                st.markdown(prompt)


        user_question = f"""
Answer the user's question using retrieved paper evidence only.

User question:
{prompt}

Tool routing:
- text/methods/model description -> retrieve_text_context
- equations/formulas/numbered equations/symbols -> retrieve_equation_context
- parameters/values/units -> retrieve_parameter_context
- states/compartments/initial conditions -> retrieve_state_context
- inputs/doses/interventions -> retrieve_input_context
- observations/outputs/data -> retrieve_observation_context
- mechanisms/feedback/Hill effects -> retrieve_mechanism_context
- tables -> retrieve_table_context
- figures/plots/diagrams -> retrieve_figure_context
- simulation settings -> retrieve_simulation_context
- assumptions/limitations/missing info -> retrieve_assumption_context

Evidence rules:
- Retrieve evidence before answering.
- Use only retrieved paper context.
- Do not use general scientific knowledge to fill missing definitions, values, mechanisms, or causal explanations.
- Quote values and units only when explicitly retrieved.
- If evidence is missing, ambiguous, or incomplete, say so clearly.

Equation rules:
- For numbered equations, including typos like "equa 7", use retrieve_equation_context first.
- Preserve OCR/PDF equation candidates exactly; do not rewrite, simplify, or fix them.
- Treat OCR/PDF candidates as requires_review unless explicitly validated.
- Define symbols only when the exact symbol is explicitly defined in retrieved context.
- Treat symbols as distinct (e.g., Ce, Ca, C10e, C10a, C10, C) unless retrieved evidence explicitly states equivalence.
- Preserve equation symbols exactly as written.
- Do not assign numerical values unless explicitly retrieved for that exact symbol.
- Describe only what is mathematically visible or explicitly retrieved.
- Do not infer biological mechanisms, saturation, inhibition, stimulation, asymptotes, pharmacological meaning, or causality unless explicitly supported.

For equation answers use:
1. Retrieved equation
2. Explicit symbol definitions
3. Mathematical structure visible in the equation
4. Evidence-supported interpretation
5. Missing / requires review

Calculation rules:
- If the user asks to calculate a derived quantity, you may calculate it ONLY if:
  1. the formula is explicitly retrieved from the paper or reviewed context, and
  2. all required numerical values are explicitly retrieved.
- Show the formula, substitution, result, and units.
- Do not invent missing values.
- If one value is missing, say which value is missing.
- Example:
  If CL = ke * Vd, ke = 2.7 1/d, and Vd = 1.105 L/kg:
  CL = 2.7 * 1.105 = 2.9835 L/kg/day.

Figure rules:
- Use retrieve_figure_context first.
- Separate:
    1. Visible figure content
    2. Caption-supported meaning
    3. Cautious interpretation
- Describe only retrieved visual evidence.
- Prefer describing arrows, labels, axes, symbols, legends, trends, interactions, and captions.
- Do not invent trends, values, mechanisms, regulation, feedback, causality, or physiological implications.
- If uncertain, prefer wording like:
"The figure appears to show..."

Table rules:
- Use retrieve_table_context first.
- Prioritize explicit retrieved rows, values, units, captions, and sources/status.
- If parameter names, descriptions, values, units, or sources are retrieved, extract them explicitly.
- Do not say values are missing if they appear in retrieved context.
- If only partial rows are retrieved, extract only visible rows.
- Prefer structured tables over narrative explanation.
- Do not summarize obvious retrieved values into generic prose.
- Avoid generic sections such as "Source Context", "Summary", or "Cautious Interpretation" unless uncertainty exists.
- For parameter tables:
    - Preserve symbols exactly as written (ka, ke, Vd, keo, Emax, Ca, Cb, Ce).
    - Rename first column to "Parameter" when symbols are reported.
    - Return rows as:
- After presenting a retrieved parameter table, stop.
- Do not add explanatory paragraphs unless the user explicitly asks for interpretation.

Parameter | Description | Value | Unit | Source

- Use concise scientific table titles:
"Table X: <caption>"

Style:
- Be concise, scientific, and evidence-grounded.
- Prefer structured outputs over narrative text.
- Avoid repeating obvious retrieved content.
- Keep review notes short and factual.
- Use a short Review note only for OCR-derived, uncertain, incomplete, or weakly supported content.
"""
        with chat_col:
            with st.chat_message("assistant"):
                with st.spinner("Searching the paper..."):
                    response = st.session_state.agent.invoke({
                        "messages": [
                            {
                                "role": "user",
                                "content": user_question
                            }
                        ]
                    })

                    answer = response["messages"][-1].content

                    # Show text answer
                    render_markdown_with_latex(answer)

                    # ---------------------------
                    # Voice answer
                    # ---------------------------
                    with st.spinner("Generating voice answer..."):
                        answer_audio_path = text_to_speech(answer)

                    st.audio(answer_audio_path)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.session_state.latest_paper_answer = answer

    if st.session_state.latest_paper_answer:
        with chat_col:
            if st.button("Send this answer to review draft"):
                append_latest_answer_to_review_draft(
                    st.session_state.latest_paper_answer
                )
                st.success(
                    f"Latest paper Q&A answer appended to {DRAFT_REVIEWED_MODEL_PATH}"
                )
                st.rerun()


# --------------------------------------------------
# Mode 2: Run model discovery
# --------------------------------------------------

elif mode == "Run model discovery":

    st.markdown(
        """
        <div class="page-title">
            Run Mechanistic Model Discovery
            <span class="page-title-note">- requires human review</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.model_discovery_result is not None:
        render_discovery_review(
            st.session_state.model_discovery_result
        )
    
    show_project_mechanism_flowchart()

    existing_discovery = load_text_file(PROJECT_REVIEW_PATH)
    existing_draft_json = load_json_file(PROJECT_DRAFT_JSON_PATH)

    existing_discovery_is_clean = (
    existing_discovery is not None
    and not looks_like_debug_extraction(existing_discovery)
    )

    run_discovery = False

    has_project_outputs = (
        existing_discovery_is_clean
        or existing_draft_json is not None
    )

    if has_project_outputs:
        st.info("Existing discovery/project files found for this PDF.")

        reuse_col, fresh_col = st.columns(2)

        with reuse_col:
            if st.button(
                "Continue existing discovery",
                key="continue_existing_discovery",
                use_container_width=True,
            ):
                if existing_discovery_is_clean:
                    st.session_state.model_discovery_result = existing_discovery
                    st.session_state.reviewed_extraction = existing_discovery

                st.success("Loaded existing project state for this PDF.")
                st.rerun()

        with fresh_col:
            run_discovery = st.button(
                "Run fresh discovery",
                key="run_fresh_model_discovery",
                use_container_width=True,
            )

    else:
        run_discovery = st.button(
            "Run model discovery",
            key="run_model_discovery",
            use_container_width=True,
        )

    if run_discovery:

        if st.session_state.get("discovery_running", False):
            st.warning("Discovery is already running.")
            if st.button("Force unlock discovery"):
                st.session_state.discovery_running = False
                st.rerun()
            st.stop()

        st.session_state.discovery_running = True
        st.warning(
            "Model discovery is running. Please do not switch workflow pages until it finishes."
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.info("Preparing model discovery...")
            progress_bar.progress(10)

            clear_project_review_files_only()
            clear_output_files()

            st.session_state.latest_scaffold = None
            st.session_state.latest_flowchart = None
            st.session_state.reviewed_extraction = None
            st.session_state.review_editor_text = ""
            st.session_state.review_validated = False
            st.session_state.generated_python_model = None
            st.session_state.pop("generated_code_editor", None)
            st.session_state.simulation_requirements = None

            vector_store = st.session_state.get("vector_store")

            if vector_store is None:
                st.error("Vector store not initialized. Please reload the PDF.")
                st.session_state.discovery_running = False
                st.stop()

            status_text.info("Generating equation crops locally...")
            progress_bar.progress(25)

            status_text.info("Retrieving model-relevant context...")
            progress_bar.progress(45)

            status_text.info("LLM extracting structured model evidence...")
            progress_bar.progress(65)

            answer = run_controlled_discovery(
                vector_store=vector_store,
                pdf_path=st.session_state.pdf_path,
                equation_candidates_dir=PROJECT_PATHS["equation_candidates_dir"],
            )

            sync_global_discovery_artifacts_to_project()

            status_text.info("Saving and formatting discovery review...")
            progress_bar.progress(90)

            st.session_state.model_discovery_result = answer
            st.session_state.reviewed_extraction = answer
            st.session_state.latest_flowchart = None

            save_text_file(PROJECT_REVIEW_PATH, answer)
            save_text_file(PROJECT_DRAFT_MODEL_PATH, answer)

            extracted_json = load_json_file(PROJECT_EVIDENCE_JSON_PATH)

            if extracted_json is not None:
                save_json_file(PROJECT_DRAFT_JSON_PATH, extracted_json)


            progress_bar.progress(100)
            status_text.success("Model discovery completed.")

            render_discovery_review(answer)

            # --------------------------------------------------
            # Mechanism flowchart
            # --------------------------------------------------
            extracted_json = load_json_file(
                PROJECT_DRAFT_JSON_PATH
            )

            if extracted_json:
                mermaid_code = extracted_json.get(
                    "mechanism_flowchart", ""
                )

                if mermaid_code:
                    st.markdown("## Mechanism Flowchart")
                    render_mermaid(
                        clean_mermaid_flowchart(mermaid_code)
                    )

        except Exception as error:
            st.session_state.discovery_running = False
            status_text.error(
                f"Model discovery failed: {type(error).__name__}: {error}"
            )
            raise

        finally:
            st.session_state.discovery_running = False

# --------------------------------------------------
# Mode 3: Review & Validate Model
# --------------------------------------------------

elif mode == "Review & Validate Model":

    st.markdown(
        '<div class="page-title">Review & Validate Model</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(
            '<div class="section-title">Structured reviewed model</div>',
            unsafe_allow_html=True,
        )
        draft_col, validated_col, download_col = st.columns(3)


        with draft_col:
            if st.button("Load working draft", use_container_width=True):
                draft_review = load_text_file(PROJECT_DRAFT_MODEL_PATH)

                if draft_review is None:
                    st.warning("No saved draft review found yet.")
                else:
                    st.session_state.review_editor_text = draft_review
                    st.session_state.reviewed_extraction = draft_review
                    st.session_state.validated_model_loaded_for_edit = False
                    st.success(f"Loaded {PROJECT_DRAFT_MODEL_PATH}")

        with validated_col:
            if st.button("Load validated model", use_container_width=True):
                validated_review = load_text_file(PROJECT_FINAL_MODEL_PATH)

                if validated_review is None:
                    st.warning("No validated reviewed model found yet.")
                else:
                    st.session_state.review_editor_text = validated_review
                    st.session_state.reviewed_extraction = validated_review
                    st.session_state.review_validated = False
                    st.session_state.validated_model_loaded_for_edit = True
                    st.success("Loaded validated reviewed model for editing.")

        with download_col:

            if os.path.exists(PROJECT_FINAL_MODEL_PATH):
                validated_model_text = load_text_file(
                    PROJECT_FINAL_MODEL_PATH
                )

                st.download_button(
                    "Download validated model",
                    validated_model_text,
                    file_name="validated_reviewed_model.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.button(
                    "Download validated model",
                    disabled=True,
                    use_container_width=True,
                )



        if st.session_state.validated_model_loaded_for_edit:
            st.warning(
                "You are editing the validated source of truth. Re-validate after changes before simulation/code generation."
            )

        review_json = load_json_file(PROJECT_DRAFT_JSON_PATH)

        with st.expander("Structured JSON draft", expanded=True):
            if review_json is None:
                st.info("No JSON draft found. Run model discovery first.")
            else:
                edited_json_text = st.text_area(
                    "JSON draft editor",
                    value=json.dumps(review_json, indent=2, ensure_ascii=False),
                    height=520,
                    key="json_draft_editor",
                )

                if st.button("Save edited JSON draft", use_container_width=True):
                    try:
                        edited_json = json.loads(edited_json_text)

                        save_json_file(
                            PROJECT_DRAFT_JSON_PATH,
                            edited_json,
                        )

                        st.success("Edited JSON draft saved.")
                        st.rerun()

                    except json.JSONDecodeError as error:
                        st.error(f"Invalid JSON: {error}")

        validate_json_col, download_json_col = st.columns(2)

        with validate_json_col:
            if st.button(
                "Validate JSON model",
                use_container_width=True,
                disabled=review_json is None,
            ):
                save_json_file(
                    PROJECT_FINAL_JSON_PATH,
                    review_json,
                )

                st.success(
                    f"Validated JSON model saved to {PROJECT_FINAL_JSON_PATH}"
                )
                st.rerun()

        with download_json_col:
            if review_json is not None:
                st.download_button(
                    "Download JSON draft",
                    json.dumps(review_json, indent=2, ensure_ascii=False),
                    file_name="reviewed_model_draft.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.button(
                    "Download JSON draft",
                    disabled=True,
                    use_container_width=True,
                )



        # ---------------------------------------
        # ADD BATCH 2 HERE
        # ---------------------------------------
        with st.expander("JSON Review Queue", expanded=True):
            review_queue = build_review_queue_from_json(review_json)

            if not review_queue:
                st.success("No obvious JSON review items detected.")

            else:
                st.warning(f"{len(review_queue)} JSON review item(s) detected.")

                labels = [
                    f"{i + 1}. {item['label']}"
                    for i, item in enumerate(review_queue)
                ]

                selected_label = st.selectbox(
                    "Select item to fix",
                    labels,
                    key="selected_json_review_item",
                )

                selected_index = labels.index(selected_label)
                item = review_queue[selected_index]

                st.caption(
                    f"type: {item['type']} | location: {item['location']}"
                )

                if item["type"] == "missing_parameter":

                    st.markdown(
                        f'<div class="small-title">Fix parameter: <code>{item.get("symbol")}</code></div>',
                        unsafe_allow_html=True,
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        new_value = st.text_input("Value")

                    with col2:
                        new_unit = st.text_input(
                            "Unit",
                            value="not reported",
                        )

                    with col3:
                        new_status = st.selectbox(
                            "Status",
                            ["reported", "estimated", "fixed", "derived", "missing"],
                            index=0,
                        )

                    new_formula = st.text_input("Formula / note")

                    if st.button("Apply to JSON draft"):
                        if not new_value.strip():
                            st.warning("Please enter a value.")
                        else:
                            updated_json = update_json_parameter(
                                data=review_json,
                                item=item,
                                value=new_value,
                                unit=new_unit,
                                status=new_status,
                                formula=new_formula,
                            )

                            save_json_file(
                                PROJECT_DRAFT_JSON_PATH,
                                updated_json,
                            )

                            st.success("Parameter updated in JSON draft.")
                            st.rerun()

                elif item["type"] == "equation_review":
                    st.info("Equation replacement will be added next.")

                elif item["type"] == "missing_input":
                    st.info("Input editor will be added next.")

        with st.expander("Add extra parameter / input", expanded=False):

            add_type = st.selectbox(
                "What do you want to add?",
                ["parameter", "input"],
                key="extra_item_type",
            )

            target_kind = st.selectbox(
                "Where should it be added?",
                ["ode", "process_module"],
                key="extra_item_target_kind",
            )

            if review_json is None:
                st.info("No JSON draft available.")
            else:
                if target_kind == "ode":
                    targets = [
                        f"{i}: ODE {ode.get('state', i)}"
                        for i, ode in enumerate(review_json.get("odes", []))
                    ]
                else:
                    targets = [
                        f"{i}: {module.get('name', i)}"
                        for i, module in enumerate(review_json.get("process_modules", []))
                    ]

                selected_target = st.selectbox(
                    "Target",
                    targets,
                    key="extra_item_target",
                )

                target_index = int(selected_target.split(":", 1)[0])

                symbol = st.text_input("Symbol", key="extra_symbol")
                value = st.text_input("Value", key="extra_value")
                unit = st.text_input("Unit", key="extra_unit")
                meaning = st.text_input("Meaning / note", key="extra_meaning")

                status = st.selectbox(
                    "Status",
                    ["reported", "estimated", "fixed", "derived", "missing"],
                    key="extra_status",
                )

                if st.button("Add to JSON draft", key="add_extra_to_json"):

                    if not symbol.strip():
                        st.warning("Please enter a symbol.")
                    else:
                        updated_json = json.loads(json.dumps(review_json))

                        new_item = {
                            "symbol": symbol.strip(),
                            "value": value.strip() if value.strip() else "not reported",
                            "unit": unit.strip() if unit.strip() else "not reported",
                            "meaning": meaning.strip() if meaning.strip() else "not reported",
                            "status": status,
                            "review": "human_added",
                        }

                        if target_kind == "ode":
                            target = updated_json["odes"][target_index]

                            if add_type == "parameter":
                                target.setdefault("parameters", []).append(new_item)
                            else:
                                target.setdefault("inputs", []).append(new_item)

                        else:
                            target = updated_json["process_modules"][target_index]

                            if add_type == "parameter":
                                target.setdefault("parameters", []).append(new_item)
                            else:
                                target.setdefault("inputs", []).append(new_item)

                        save_json_file(PROJECT_DRAFT_JSON_PATH, updated_json)

                        st.success(f"Added `{symbol}` to JSON draft.")
                        st.rerun()

    with right:
        # Keep PDF/crop viewer visible while left content scrolls.
        st.markdown(
            '<span class="sticky-right-panel-marker"></span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Equation recovery</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Missing Equation Recovery", expanded=True):
            missing_equations = load_json_file(PROJECT_MISSING_EQUATIONS_PATH) or []
            show_all_crops = st.checkbox(
                "Show all equation crops",
                value=False,
                help=(
                    "Use this when an equation is present in the draft but garbled. "
                    "Missing equations are shown by default."
                ),
                key="show_all_equation_crops"
            )

            if show_all_crops:
                crop_paths = list_equation_crop_paths()
                selectable_missing = []

                for crop_path in crop_paths:

                    filename = os.path.basename(crop_path)

                    page_match = re.search(
                        r"(?:page_|_page_)(\d+)",
                        filename
                    )

                    candidate_match = re.search(
                        r"(?:equation_candidate_|candidate_)(\d+)",
                        filename
                    )

                    equation_match = re.search(
                        r"(?:equation_|eq_)(\d+)",
                        filename
                    )

                    page_number = (
                        page_match.group(1)
                        if page_match
                        else "unknown"
                    )

                    candidate_number = (
                        candidate_match.group(1)
                        if candidate_match
                        else (
                            equation_match.group(1)
                            if equation_match
                            else "unknown"
                        )
                    )

                    selectable_missing.append(
                        {
                            "equation_number": candidate_number,
                            "status": "available crop",
                            "source_page": page_number,
                            "candidate_crop": crop_path,
                            "filename": filename,
                        }
                    )
            else:
                selectable_missing = [
                    item for item in missing_equations
                    if item.get("candidate_crop")
                ]

            if not selectable_missing:
                st.info(
                    "No equation recovery crops found yet. Run model discovery first."
                )
            else:
                selected_label = st.selectbox(
                    (
                        "Equation crop"
                        if show_all_crops
                        else "Detected missing equation"
                    ),
                    [
                        (
                            f"Page {item.get('source_page')} | "
                            f"Candidate {item.get('equation_number')} | "
                            f"{item.get('filename', os.path.basename(item.get('candidate_crop', '')))}"
                        )
                        for item in selectable_missing
                    ],
                    key="missing_equation_recovery_select"
                )
                selected_index = [
                    (
                        f"Page {item.get('source_page')} | "
                        f"Candidate {item.get('equation_number')} | "
                        f"{item.get('filename', os.path.basename(item.get('candidate_crop', '')))}"
                    )
                    for item in selectable_missing
                ].index(selected_label)
                selected_missing = selectable_missing[selected_index]
                equation_number = str(
                    selected_missing.get("equation_number") or ""
                )
                crop_path = selected_missing.get("candidate_crop")
                candidate_key = f"{equation_number}:{crop_path}"
                cache_picker_key = (
                    "cached_ocr_candidate_"
                    + hashlib.md5(candidate_key.encode("utf-8")).hexdigest()
                )

                st.write(f"Status: {selected_missing.get('status')}")
                st.write(f"Crop: `{crop_path}`")
                st.caption(
                    "Choose an OCR model and run extraction on all visible equations "
                    "inside the crop. Results are cached automatically."
                )

                if crop_path and os.path.exists(crop_path):
                    st.image(crop_path, caption=f"Equation {equation_number} crop")

                # -----------------------------------------
                # OCR model selector
                # -----------------------------------------

                ocr_model = st.selectbox(
                    "OCR model",
                    [
                        "gpt-4o",
                        "gpt-4.1-mini",
                        "gpt-4o-mini",
                    ],
                    index=1,
                    help=(
                        "gpt-4o = strongest but expensive\n"
                        "gpt-4.1-mini = good balance\n"
                        "gpt-4o-mini = cheapest"
                    ),
                    key=f"ocr_model_{candidate_key}"
                )

                st.caption(
                    "Runs OCR on all visible equations in this crop. "
                    "Results are cached automatically."
                )

                if st.button(
                    f"Run OCR ({ocr_model})",
                    use_container_width=True,
                    key=f"run_ocr_{candidate_key}"
                ):
                    try:
                        ocr_result, cache_path, from_cache = (
                            run_cached_visible_equations_ocr(
                                image_path=crop_path,
                                model=ocr_model,
                            )
                        )

                        st.session_state.missing_equation_ocr_candidates[
                            candidate_key
                        ] = {
                            "text": ocr_result,
                            "cache_path": cache_path,
                            "method": f"{ocr_model} visible-equations OCR",
                        }

                        st.success(
                            (
                                "Loaded cached OCR candidate."
                                if from_cache
                                else f"{ocr_model} OCR candidate created."
                            )
                        )

                    except Exception as error:
                        st.error(
                            f"{ocr_model} OCR failed. "
                            f"{type(error).__name__}: {error}"
                        )

                cached_candidates = load_cached_ocr_candidates_for_crop(
                    image_path=crop_path,
                    equation_number=equation_number,
                )

                if cached_candidates:

                    cached_labels = [
                        candidate["label"]
                        for candidate in cached_candidates
                    ]

                    selected_cached_label = st.selectbox(
                        "Cached OCR candidates",
                        cached_labels,
                        key=cache_picker_key,
                    )

                    selected_cached = cached_candidates[
                        cached_labels.index(selected_cached_label)
                    ]

                    st.session_state.missing_equation_ocr_candidates[
                        candidate_key
                    ] = {
                        "text": selected_cached["text"],
                        "cache_path": selected_cached["cache_path"],
                        "method": selected_cached["method"],
                    }

                candidate = st.session_state.missing_equation_ocr_candidates.get(
                    candidate_key
                )


                if candidate:
                    st.markdown("### Suggested equation candidate")

                    candidate_text = candidate.get("text", "")

                    st.code(
                        candidate_text,
                        language="text"
                    )

                    if candidate:

                        st.markdown("### Suggested equation candidate")

                        candidate_text = candidate.get("text", "")

                        st.code(
                            candidate_text,
                            language="text"
                        )

                        review_queue = build_review_queue_from_json(review_json)

                        equation_items = [
                            item for item in review_queue
                            if item["type"] == "equation_review"
                        ]

                        if equation_items:

                            equation_labels = [
                                item["label"]
                                for item in equation_items
                            ]

                            selected_equation_label = st.selectbox(
                                "Apply OCR result to which equation?",
                                equation_labels,
                                key=f"ocr_target_equation_{candidate_key}"
                            )

                            selected_item = equation_items[
                                equation_labels.index(selected_equation_label)
                            ]

                            if st.button(
                                "Accept OCR candidate → update JSON",
                                use_container_width=True,
                                key=f"accept_ocr_json_{candidate_key}"
                            ):

                                clean_equation_text = extract_equation_text_from_ocr_record(
                                    candidate_text
                                )

                                equation_candidates = split_ocr_equations(clean_equation_text)

                                selected_clean_equation = st.selectbox(
                                    "Select equation text to apply",
                                    equation_candidates,
                                    key=f"selected_clean_equation_{candidate_key}",
                                )

                                updated_json = apply_ocr_candidate_to_json(
                                    data=review_json,
                                    item=selected_item,
                                    equation_text=selected_clean_equation,
                                )

                                save_json_file(
                                    PROJECT_DRAFT_JSON_PATH,
                                    updated_json,
                                )

                                st.success(
                                    "OCR equation applied to JSON draft."
                                )

                                st.rerun()

                        else:
                            st.info(
                                "No suspicious equations currently detected in review queue."
                            )


# --------------------------------------------------
# Mode 4: Simulation setup
# --------------------------------------------------

elif mode == "Simulation setup":

    st.markdown(
        '<div class="page-title">Simulation Setup</div>',
        unsafe_allow_html=True,
    )

    validated_review_json = load_json_file(PROJECT_FINAL_JSON_PATH)
    draft_review_json = load_json_file(PROJECT_DRAFT_JSON_PATH)

    review_source = validated_review_json or draft_review_json

    if review_source is None:
        st.warning(
            "Please create or validate a reviewed JSON model before simulation setup."
        )
        st.stop()

    if validated_review_json is None:
        st.info(
            "Using JSON draft for simulation setup. "
            "Validate it when ready."
        )
    else:
        st.success("Using validated JSON model for simulation setup.")

    if st.button("Validate equations"):
        st.info(
            "Equation validation will be refactored to read from reviewed JSON next. "
            "For now, use Infer simulation requirements."
        )
    if st.button("Infer simulation requirements"):

        with st.spinner("Inferring simulation requirements..."):
            requirements = infer_simulation_requirements(
                review_source
            )

        st.session_state.simulation_requirements = requirements

    if st.session_state.simulation_requirements is not None:

        req = st.session_state.simulation_requirements

        st.markdown("### Model type")
        st.write(req.get("model_type", "unknown"))

        st.markdown("### States / initial conditions")

        for i, state in enumerate(req.get("states", [])):

            state_name = state.get("name", f"state_{i}")

            state["user_value"] = st.text_input(
                f"{state_name} initial value",
                value="" if state.get("suggested_default") is None else str(state["suggested_default"]),
                help=state.get("description", ""),
                key=f"simulation_state_{i}_{state_name}",
            )

        st.markdown("### Parameters")

        for i, param in enumerate(req.get("parameters", [])):

            param_name = param.get("name", f"param_{i}")
            param_unit = param.get("unit", "")

            param["user_value"] = st.text_input(
                f"{param_name} [{param_unit}]",
                value="" if param.get("value") is None else str(param["value"]),
                help=param.get("description", ""),
                key=f"simulation_param_{i}_{param_name}",
            )

        st.markdown("### Inputs")

        for i, inp in enumerate(req.get("inputs", [])):

            inp_name = inp.get("name", f"input_{i}")
            inp_unit = inp.get("unit", "")

            inp["user_value"] = st.text_input(
                f"{inp_name} [{inp_unit}]",
                value="" if inp.get("suggested_default") is None else str(inp["suggested_default"]),
                help=inp.get("description", ""),
                key=f"simulation_input_{i}_{inp_name}",
            )

        st.markdown("### Time settings")
        time_settings = req.get("time_settings", {})

        time_settings["start"] = st.text_input(
            "Simulation start",
            value=str(time_settings.get("start", 0))
        )

        time_settings["end"] = st.text_input(
            "Simulation end",
            value="" if time_settings.get("end") is None else str(time_settings["end"])
        )

        time_settings["unit"] = st.text_input(
            "Time unit",
            value="" if time_settings.get("unit") is None else str(time_settings["unit"])
        )

        st.markdown("### Missing for simulation")
        st.write(req.get("missing_for_simulation", []))

        st.markdown("### Human-review notes")
        st.write(req.get("human_review_notes", []))

        if st.button("Save simulation setup"):

            os.makedirs(OUTPUT_DIR, exist_ok=True)

            with open(PROJECT_SIMULATION_PATH, "w", encoding="utf-8") as file:
                json.dump(req, file, indent=2)

            st.success(f"Saved to {PROJECT_SIMULATION_PATH}")
            st.rerun()


# --------------------------------------------------
# Mode 5: Generate Python model
# --------------------------------------------------

elif mode == "Generate Python Model":

    st.markdown(
        '<div class="page-title">Generate Python Model</div>',
        unsafe_allow_html=True,
    )

    validated_review_json = load_json_file(PROJECT_FINAL_JSON_PATH)
    draft_review_json = load_json_file(PROJECT_DRAFT_JSON_PATH)

    review_source = validated_review_json or draft_review_json

    if review_source is None:
        st.warning(
            "Please create or validate a reviewed JSON model before generating Python code."
        )
        st.stop()

    if os.path.exists(PROJECT_SIMULATION_PATH):
        simulation_requirements_from_file = load_json_file(
            PROJECT_SIMULATION_PATH
        )
    else:
        simulation_requirements_from_file = None

    if (
        st.session_state.generated_python_model is None
        and os.path.exists(PROJECT_GENERATED_MODEL_PATH)
    ):
        st.session_state.generated_python_model = load_text_file(
            PROJECT_GENERATED_MODEL_PATH
        )

    if (
        "generated_code_editor" not in st.session_state
        and st.session_state.generated_python_model
    ):
        st.session_state.generated_code_editor = (
            st.session_state.generated_python_model
        )

    settings_col, code_col = st.columns([0.9, 1.25], gap="large")

    with settings_col:
        st.markdown(
            '<div class="section-title">Model inputs</div>',
            unsafe_allow_html=True,
        )

        simulation_time = st.text_input(
            "Simulation end time",
            value="",
            placeholder="Leave empty if unknown"
        )

        initial_conditions = st.text_area(
            "Initial conditions (JSON format)",
            value='{"C": 0, "Ce": 0}'
        )

        extra_parameters = st.text_area(
            "Extra parameter values (JSON format)",
            value='{"dose": 0.02}'
        )

        if st.button("Generate Python Model", use_container_width=True):

            st.session_state.generated_python_model = None

            simulation_setup = {
                "simulation_end": simulation_time,
                "initial_conditions": initial_conditions,
                "extra_parameters": extra_parameters
            }

            st.session_state.simulation_requirements = simulation_setup

            os.makedirs(OUTPUT_DIR, exist_ok=True)

            with open(PROJECT_SIMULATION_PATH, "w", encoding="utf-8") as file:
                json.dump(simulation_setup, file, indent=2)

            with st.spinner("Generating executable model..."):

                code = save_generated_python_model(
                    path=PROJECT_GENERATED_MODEL_PATH,
                    reviewed_extraction=json.dumps(review_source, indent=2),
                    simulation_setup=simulation_requirements_from_file or simulation_setup,
                )

                st.session_state.generated_python_model = code
                st.session_state.generated_code_editor = code

            st.success("Python model generated.")
            st.rerun()

    if st.session_state.generated_python_model:

        with code_col:
            st.markdown(
                '<div class="section-title">Generated Python model</div>',
                unsafe_allow_html=True,
            )

            edited_generated_code = st.text_area(
                "Generated Python code",
                height=650,
                key="generated_code_editor"
            )

            code_save_col, code_download_col = st.columns(2)

            with code_save_col:
                if st.button("Save edited generated code", use_container_width=True):
                    save_text_file(PROJECT_GENERATED_MODEL_PATH, edited_generated_code)
                    st.session_state.generated_python_model = edited_generated_code
                    st.success(f"Edited generated code saved to {PROJECT_GENERATED_MODEL_PATH}")

            with code_download_col:
                st.download_button(
                    "Download generated Python model",
                    st.session_state.generated_code_editor,
                    file_name="generated_model.py",
                    mime="text/x-python",
                    use_container_width=True,
                )

        # -----------------------------------------
        # Simulation controls
        # -----------------------------------------

        with settings_col:
            st.markdown(
                '<div class="section-title">Simulation options</div>',
                unsafe_allow_html=True,
            )

            solver = st.selectbox(
                "Choose solver",
                ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
                index=5
            )

            rtol = st.number_input(
                "Relative tolerance",
                value=1e-6,
                format="%.1e"
            )

            atol = st.number_input(
                "Absolute tolerance",
                value=1e-9,
                format="%.1e"
            )

            run_simulation = st.button("Run simulation", use_container_width=True)

        if run_simulation:
            if not os.path.exists(PROJECT_GENERATED_MODEL_PATH):
                st.error(
                    "No generated model found. "
                    "Generate the Python model first."
                )
                st.stop()

            with st.spinner("Running simulation..."):

                for file_path in [
                    "outputs/simulation_plot.png",
                    "outputs/simulation_results.csv",
                ]:
                    if os.path.exists(file_path):
                        os.remove(file_path)

                env = os.environ.copy()

                env["SOLVER_METHOD"] = solver
                env["RTOL"] = str(rtol)
                env["ATOL"] = str(atol)

                result = subprocess.run(
                    [sys.executable, PROJECT_GENERATED_MODEL_PATH],
                    capture_output=True,
                    text=True,
                    env=env
                )

            if result.returncode != 0:

                st.error("Simulation failed.")

                st.markdown("### Error log")
                st.code(result.stderr)

            else:

                st.success("Simulation completed.")

                if result.stdout:
                    st.markdown("### Simulation log")
                    st.code(result.stdout)

        # Always show latest simulation outputs if they exist
        if os.path.exists("outputs/simulation_plot.png"):
            st.markdown("### Latest simulation plot")
            st.image("outputs/simulation_plot.png")

        if os.path.exists("outputs/simulation_results.csv"):
            with open("outputs/simulation_results.csv", "rb") as file:
                st.download_button(
                    "Download latest simulation results CSV",
                    file,
                    file_name="simulation_results.csv",
                    mime="text/csv",
                )

# --------------------------------------------------
# Sidebar status - render last so it reflects latest files
# --------------------------------------------------
render_workflow_status(PROJECT_PATHS)
