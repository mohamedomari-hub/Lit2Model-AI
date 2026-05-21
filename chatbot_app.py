import os
import re
import json
import shutil
import hashlib
import subprocess
import sys

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.parser import parse_pdf_multimodal
from src.rag import build_vector_store, load_vector_store, CHROMA_DIR
from src.tools import (
    set_vector_store,
    set_active_pdf_path,
    propose_candidate_ode_model,
)
from src.agent import build_agent
from src.simulation_planner import infer_simulation_requirements
from src.model_generator import save_generated_python_model

from src.equation_parser import parse_equations


load_dotenv()

DEFAULT_PDF_PATH = "data/sample_papers/MetRep_Dexamethasone_Adm.pdf"
UPLOAD_DIR = "data/uploaded_papers"
OUTPUT_DIR = "outputs"

REVIEW_PATH = "outputs/extraction_review.md"
FINAL_REVIEWED_MODEL_PATH = "outputs/final_reviewed_model.md"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_pdf_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def clear_output_files():
    output_files = [
        "outputs/extraction_review.md",
        "outputs/agent_result.md",
        "outputs/candidate_model.py",
        "outputs/final_reviewed_model.md",
        "outputs/sanity_check_report.md",
        "outputs/mechanism_graph.html",
        "outputs/generated_model.py",
        "outputs/simulation_requirements.json",
    ]

    for file_path in output_files:
        if os.path.exists(file_path):
            os.remove(file_path)


def render_markdown_with_latex(text: str):
    text = text.replace("\\[", "$$")
    text = text.replace("\\]", "$$")

    text = re.sub(
        r"\\\((.*?)\\\)",
        r"$\1$",
        text,
        flags=re.DOTALL
    )

    st.markdown(text, unsafe_allow_html=True)


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
        scrolling=True
    )


def generate_compartment_flowchart(reviewed_text: str) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a scientific diagram generator.

Create a Mermaid flowchart from the reviewed mechanistic extraction.

Rules:
- Use only relationships explicitly present in the reviewed extraction.
- Do not invent compartments, rates, or biological mechanisms.
- Work for any mechanistic model type.
- Prefer model structure, compartments, states, flows, regulations, and feedbacks.
- If direction is uncertain, label the edge "requires review".
- Keep the diagram small and readable.
- Use Mermaid syntax only.
- Do not include markdown fences.

Use this Mermaid style:
flowchart TD
    A[Entity A] -->|relation| B[Entity B]

REVIEWED EXTRACTION:
{reviewed_text}
"""

    result = llm.invoke(prompt)
    return result.content.strip()


def initialize_system(pdf_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    current_pdf_hash = get_pdf_hash(pdf_path)
    hash_file = os.path.join(CHROMA_DIR, "current_pdf_hash.txt")

    previous_hash = None

    if os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as file:
            previous_hash = file.read().strip()

    if current_pdf_hash != previous_hash:
        print("New PDF detected. Rebuilding Chroma DB.")

        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)

        documents = parse_pdf_multimodal(pdf_path)

        vector_store = build_vector_store(
            documents,
            reset_db=False
        )

        os.makedirs(CHROMA_DIR, exist_ok=True)

        with open(hash_file, "w", encoding="utf-8") as file:
            file.write(current_pdf_hash)

    else:
        print("Same PDF detected. Reusing Chroma DB.")
        vector_store = load_vector_store()

    set_vector_store(vector_store)
    set_active_pdf_path(pdf_path)

    return build_agent()


def load_review_file():
    if os.path.exists(REVIEW_PATH):
        with open(REVIEW_PATH, "r", encoding="utf-8") as file:
            return file.read()

    return None


def save_review_file(text: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(REVIEW_PATH, "w", encoding="utf-8") as file:
        file.write(text)


def apply_simple_review_command(command: str, current_text: str) -> tuple[str, str]:
    command_clean = command.strip()

    if command_clean.lower().startswith("remove "):
        target = command_clean[7:].strip()

        if target in current_text:
            updated = current_text.replace(target, "")
            return updated, f"Removed: `{target}`"

        return current_text, f"I could not find exact text to remove: `{target}`"

    if command_clean.lower().startswith("replace ") and " with " in command_clean:
        body = command_clean[8:]
        old, new = body.split(" with ", 1)

        old = old.strip()
        new = new.strip()

        if old in current_text:
            updated = current_text.replace(old, new)
            return updated, f"Replaced:\n\n`{old}`\n\nwith:\n\n`{new}`"

        return current_text, f"I could not find exact text to replace: `{old}`"

    if command_clean.lower().startswith("add to ") and ":" in command_clean:
        header_part, addition = command_clean.split(":", 1)
        section_name = header_part[7:].strip()
        addition = addition.strip()

        pattern = re.compile(
            rf"(#+\s*.*{re.escape(section_name)}.*\n)",
            re.IGNORECASE
        )

        match = pattern.search(current_text)

        if match:
            insert_pos = match.end()
            updated = (
                current_text[:insert_pos]
                + f"\n- {addition}\n"
                + current_text[insert_pos:]
            )
            return updated, f"Added to section `{section_name}`."

        updated = current_text + f"\n\n## {section_name}\n- {addition}\n"
        return updated, f"Section `{section_name}` was not found, so I added it at the end."

    return current_text, (
        "I did not recognize this as an edit command. "
        "Try: `remove ...`, `replace ... with ...`, or `add to section: ...`."
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

    with open(FINAL_REVIEWED_MODEL_PATH, "w", encoding="utf-8") as file:
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

st.title("🧬 Lit2Model-AI")
st.caption("Scientific paper chatbot + mechanistic model discovery assistant")


# --------------------------------------------------
# PDF upload
# --------------------------------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.sidebar.markdown("## Upload paper")

uploaded_file = st.sidebar.file_uploader(
    "Upload a scientific PDF",
    type=["pdf"]
)

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = DEFAULT_PDF_PATH

if uploaded_file is not None:

    uploaded_pdf_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name
    )

    with open(uploaded_pdf_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    if st.session_state.pdf_path != uploaded_pdf_path:

        st.session_state.pdf_path = uploaded_pdf_path

        clear_output_files()

        st.session_state.pop("agent", None)
        st.session_state.pop("reviewed_extraction", None)
        st.session_state.pop("messages", None)
        st.session_state.pop("review_chat_messages", None)
        st.session_state.pop("final_reviewed_model", None)
        st.session_state.pop("latest_scaffold", None)
        st.session_state.pop("latest_flowchart", None)
        st.session_state.pop("model_discovery_result", None)
        st.session_state.pop("generated_python_model", None)
        st.session_state.pop("simulation_requirements", None)

        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)

        st.success(f"Uploaded: {uploaded_file.name}")
        st.rerun()


# --------------------------------------------------
# Initialize system
# --------------------------------------------------

if "agent" not in st.session_state:
    with st.spinner("Loading paper and vector database..."):
        st.session_state.agent = initialize_system(
            st.session_state.pdf_path
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "review_chat_messages" not in st.session_state:
    st.session_state.review_chat_messages = []

if "reviewed_extraction" not in st.session_state:
    st.session_state.reviewed_extraction = load_review_file()

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


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

mode = st.sidebar.radio(
    "Mode",
    [
        "Ask paper questions",
        "Run model discovery",
        "Review with chatbot",
        "Simulation setup",
        "Generate Python Model",
        "Manual review editor",
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Current PDF:")
st.sidebar.code(st.session_state.pdf_path)


# --------------------------------------------------
# Mode 1: Ask paper questions
# --------------------------------------------------

if mode == "Ask paper questions":

    st.subheader("Ask questions about the paper")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_markdown_with_latex(message["content"])

    prompt = st.chat_input(
        "Ask about an equation, table, figure, mechanism, or parameter..."
    )

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        user_question = f"""
Use retrieve_pdf_context to answer the user's question.

User question:
{prompt}

Evidence rules:
- First retrieve relevant evidence from the paper.
- Use only the retrieved paper context.
- If the answer is not found, say so clearly.
- Prefer evidence from tables, equations, figure captions, methods, and explicit model descriptions.
- Quote parameter values and units when present.

Figure rule:
- If the user asks about a figure, plot, diagram, or visual result, use explain_figure_from_pdf first.
- Use retrieve_pdf_context only as supplementary context if needed.
- If figure OCR text is available, interpret it cautiously instead of saying no context was found.
- If OCR text strongly suggests variables, axes, or timing, infer the most plausible scientific interpretation while clearly labeling it as interpretation.
- Prefer concrete observations over generic statements.
- Avoid vague wording such as: "likely contains multiple curves" when axes/variables are available.
- Combine evidence across modalities when relevant (text, OCR figure text, tables, captions, equations).

Scientific framing rules:
- Separate what the paper explicitly states from your interpretation.
- Do not overstate uncertainty, validation, causality, or biological meaning.
- Preserve the terminology used in the paper.
- Do not reinterpret model symbols beyond retrieved evidence.
- If the exact role of a symbol, mechanism, or parameter is ambiguous, say it is ambiguous.
- If you make an inference, label it as an interpretation.
- For equations, do not say the full equation set is available unless the retrieved context explicitly contains the complete equations.
- If equations are referenced but not fully visible in retrieved text, classify them as "requires review / possible OCR issue".

Answer style:
- Explain clearly for a scientific audience.
- For equations, render them in LaTeX when possible.
- End with a short “Review note” only if something is uncertain.
"""

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
                render_markdown_with_latex(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# --------------------------------------------------
# Mode 2: Run model discovery
# --------------------------------------------------

elif mode == "Run model discovery":

    st.subheader("Run mechanistic model discovery")

    st.write(
        "This extracts parameters, equations, mechanisms, graph edges, "
        "generates the graph, and creates a draft scaffold."
    )

    if st.session_state.model_discovery_result is not None:
        st.markdown("### Latest discovery result")
        render_markdown_with_latex(
            st.session_state.model_discovery_result
        )

    if st.session_state.latest_scaffold is not None:
        st.markdown("### Latest reviewed scaffold")
        render_markdown_with_latex(
            st.session_state.latest_scaffold
        )

    if st.session_state.latest_flowchart is not None:
        st.markdown("### Latest compartment flowchart")
        render_mermaid(st.session_state.latest_flowchart)

    if st.button("Run model discovery"):

        clear_output_files()
        st.session_state.latest_scaffold = None
        st.session_state.latest_flowchart = None

        user_question = """
Run the full mechanistic model discovery workflow using run_model_discovery_workflow.

Return the final mechanistic model scaffold.

Important:
- Do not invent parameters.
- Do not invent equations.
- Preserve reported values and units.
- Mark uncertain equations or graph edges for human review.
"""

        with st.spinner("Running model discovery workflow..."):
            response = st.session_state.agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": user_question
                    }
                ]
            })

            answer = response["messages"][-1].content
            st.session_state.model_discovery_result = answer

        st.success("Model discovery completed.")
        render_markdown_with_latex(answer)

        if os.path.exists(REVIEW_PATH):
            st.session_state.reviewed_extraction = load_review_file()
            st.success(f"Review file created: {REVIEW_PATH}")

            diagram = generate_compartment_flowchart(
                st.session_state.reviewed_extraction
            )

            st.session_state.latest_flowchart = diagram

            st.markdown("### Compartment Flowchart")
            render_mermaid(diagram)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# --------------------------------------------------
# Mode 3: Review with chatbot
# --------------------------------------------------

elif mode == "Review with chatbot":

    st.subheader("Review extraction with chatbot")

    if st.session_state.reviewed_extraction is None:
        st.warning("No extraction review found yet. Run model discovery first.")
        st.stop()

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Current reviewed extraction")
        st.text_area(
            "Reviewed extraction state",
            value=st.session_state.reviewed_extraction,
            height=650,
            key="review_state_display"
        )

    with right:
        st.markdown("### Review chat")

        st.info(
            """
You can write commands like:

- `remove Cmax`
- `replace dC/dt = -ka*C with Vd*dC/dt = dose*F*ka*exp(-ka*t) - CL*C`
- `add to Reported ODEs: Vd*dC/dt = dose*F*ka*exp(-ka*t) - CL*C`
- `generate final scaffold`

You can also ask:
- `explain why Ce is listed as derived`
- `mark the effect_dxm-gluca equation as requiring human review`
"""
        )

        for message in st.session_state.review_chat_messages:
            with st.chat_message(message["role"]):
                render_markdown_with_latex(message["content"])

        review_prompt = st.chat_input("Edit, ask, or generate final scaffold...")

        if review_prompt:
            st.session_state.review_chat_messages.append(
                {"role": "user", "content": review_prompt}
            )

            with st.chat_message("user"):
                st.markdown(review_prompt)

            lower_prompt = review_prompt.lower().strip()

            if (
                "generate final scaffold" in lower_prompt
                or "generate final model" in lower_prompt
            ):
                with st.chat_message("assistant"):
                    with st.spinner("Generating final scaffold from reviewed extraction..."):
                        final_answer = generate_final_from_review(
                            st.session_state.reviewed_extraction
                        )

                        st.session_state.final_reviewed_model = final_answer
                        st.session_state.latest_scaffold = final_answer

                        diagram = generate_compartment_flowchart(
                            st.session_state.reviewed_extraction
                        )
                        st.session_state.latest_flowchart = diagram

                        render_markdown_with_latex(final_answer)

                        st.markdown("### Compartment Flowchart")
                        render_mermaid(diagram)

                st.session_state.review_chat_messages.append(
                    {
                        "role": "assistant",
                        "content": final_answer
                    }
                )

            elif lower_prompt.startswith("explain ") or lower_prompt.startswith("why "):
                user_question = f"""
Use only the reviewed extraction below to answer the user's review question.

Reviewed extraction:
{st.session_state.reviewed_extraction}

User question:
{review_prompt}

Rules:
- Do not invent new paper facts.
- Explain based on the current reviewed extraction.
- If the answer is uncertain, say what should be checked.
"""

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.agent.invoke({
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_question
                                }
                            ]
                        })

                        answer = response["messages"][-1].content
                        render_markdown_with_latex(answer)

                st.session_state.review_chat_messages.append(
                    {"role": "assistant", "content": answer}
                )

            else:
                updated_text, edit_message = apply_simple_review_command(
                    review_prompt,
                    st.session_state.reviewed_extraction
                )

                if edit_message.startswith("I did not recognize"):
                    with st.spinner("Applying scientific review edit..."):
                        updated_text = scientific_review_edit(
                            review_prompt,
                            st.session_state.reviewed_extraction
                        )

                    edit_message = "Applied scientific review edit."

                st.session_state.reviewed_extraction = updated_text
                save_review_file(updated_text)

                with st.chat_message("assistant"):
                    st.markdown(edit_message)

                st.session_state.review_chat_messages.append(
                    {"role": "assistant", "content": edit_message}
                )

                st.rerun()


# --------------------------------------------------
# Mode 4: Simulation setup
# --------------------------------------------------

elif mode == "Simulation setup":

    st.subheader("Simulation setup")

    st.markdown("### Equation validation")

    st.info(
        """
    Equation validation is experimental.

    The current lightweight SymPy validator can parse simple Python-like equations,
    but may fail on raw LaTeX/OCR equations.

    A failed parse does not necessarily mean the equation is wrong.
    It means the equation requires review or better formatting before symbolic validation.
    """
    )

    if st.button("Validate equations"):

        parsed_equations = parse_equations(
            st.session_state.reviewed_extraction
        )

        for eq in parsed_equations:

            if eq["valid"]:

                st.success(
                    f"Parsed: {eq['lhs']} = {eq['rhs']}"
                )

            else:

                st.warning(
                    f"Equation could not be symbolically parsed yet:\n\n"
                    f"`{eq['equation']}`\n\n"
                    f"Reason: {eq['error']}\n\n"
                    f"This does not mean the equation is wrong. "
                    f"It means the current lightweight SymPy parser cannot parse raw LaTeX/OCR notation yet."
                )

    if st.session_state.reviewed_extraction is None:
        st.warning("Run model discovery first.")
        st.stop()

    if st.button("Infer simulation requirements"):

        with st.spinner("Inferring simulation requirements..."):
            requirements = infer_simulation_requirements(
                st.session_state.reviewed_extraction
            )

        st.session_state.simulation_requirements = requirements

    if st.session_state.simulation_requirements is not None:

        req = st.session_state.simulation_requirements

        st.markdown("### Model type")
        st.write(req.get("model_type", "unknown"))

        st.markdown("### States / initial conditions")
        for state in req.get("states", []):
            state["user_value"] = st.text_input(
                f"{state['name']} initial value",
                value="" if state.get("suggested_default") is None else str(state["suggested_default"]),
                help=state.get("description", "")
            )

        st.markdown("### Parameters")
        for param in req.get("parameters", []):
            param["user_value"] = st.text_input(
                f"{param['name']} [{param.get('unit')}]",
                value="" if param.get("value") is None else str(param["value"]),
                help=param.get("description", "")
            )

        st.markdown("### Inputs")
        for inp in req.get("inputs", []):
            inp["user_value"] = st.text_input(
                f"{inp['name']} [{inp.get('unit')}]",
                value="" if inp.get("suggested_default") is None else str(inp["suggested_default"]),
                help=inp.get("description", "")
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

            os.makedirs("outputs", exist_ok=True)

            with open("outputs/simulation_requirements.json", "w", encoding="utf-8") as file:
                json.dump(req, file, indent=2)

            st.success("Saved to outputs/simulation_requirements.json")


# --------------------------------------------------
# Mode 5: Generate Python model
# --------------------------------------------------

elif mode == "Generate Python Model":

    st.subheader("Generate executable Python model")

    st.info(
            """
        Workflow:
        1. Run model discovery
        2. Review/correct the extraction
        3. Generate Python model
        4. Choose solver
        5. Run simulation
        """
        )

    if st.session_state.reviewed_extraction is None:
        st.warning(
            "No reviewed extraction found. Run model discovery first."
        )
        st.stop()

    st.markdown(
        """
Provide simulation settings if available.
Leave blank if unknown.
"""
    )

    col1, col2 = st.columns(2)

    with col1:
        simulation_time = st.number_input(
            "Simulation end time",
            value=120.0
        )

        initial_conditions = st.text_area(
            "Initial conditions (JSON format)",
            value='{"C": 0, "Ce": 0}'
        )

    with col2:
        extra_parameters = st.text_area(
            "Extra parameter values (JSON format)",
            value='{"dose": 0.02}'
        )

    if st.button("Generate Python Model"):

        st.session_state.generated_python_model = None

        if os.path.exists("outputs/generated_model.py"):
            os.remove("outputs/generated_model.py")

        simulation_setup = {
            "simulation_end": simulation_time,
            "initial_conditions": initial_conditions,
            "extra_parameters": extra_parameters
        }

        st.session_state.simulation_requirements = simulation_setup

        os.makedirs("outputs", exist_ok=True)

        with open("outputs/simulation_requirements.json", "w", encoding="utf-8") as file:
            json.dump(simulation_setup, file, indent=2)

        with st.spinner("Generating executable model..."):

            code = save_generated_python_model(
                path="outputs/generated_model.py",
                reviewed_extraction=st.session_state.reviewed_extraction,
                simulation_setup=simulation_setup
            )

            st.session_state.generated_python_model = code

        st.success("Python model generated.")

    if st.session_state.generated_python_model:

        st.markdown("### Generated Python model")

        st.download_button(
            "Download generated Python model",
            st.session_state.generated_python_model,
            file_name="generated_model.py",
            mime="text/x-python"
        )

        # -----------------------------------------
        # Simulation controls
        # -----------------------------------------

        st.markdown("### Simulation options")

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

        if st.button("Run simulation"):

            if not os.path.exists("outputs/generated_model.py"):
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
                    [sys.executable, "outputs/generated_model.py"],
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

                if os.path.exists(
                    "outputs/simulation_plot.png"
                ):
                    st.markdown("### Simulation plot")

                    st.image(
                        "outputs/simulation_plot.png"
                    )

                if os.path.exists(
                    "outputs/simulation_results.csv"
                ):

                    with open(
                        "outputs/simulation_results.csv",
                        "rb"
                    ) as file:

                        st.download_button(
                            "Download simulation results CSV",
                            file,
                            file_name="simulation_results.csv",
                            mime="text/csv"
                        )


# --------------------------------------------------
# Mode 6: Manual review editor
# --------------------------------------------------

elif mode == "Manual review editor":

    st.subheader("Manual review editor")

    if st.session_state.reviewed_extraction is None:
        st.warning("No extraction review found yet. Run model discovery first.")
        st.stop()

    edited_review = st.text_area(
        "Edit extracted parameters, equations, mechanisms, and graph edges:",
        value=st.session_state.reviewed_extraction,
        height=650
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save reviewed extraction"):
            st.session_state.reviewed_extraction = edited_review
            save_review_file(edited_review)
            st.success("Reviewed extraction saved.")

    with col2:
        if st.button("Generate final scaffold from reviewed extraction"):
            with st.spinner("Generating final scaffold..."):
                final_answer = generate_final_from_review(
                    edited_review
                )

                st.session_state.final_reviewed_model = final_answer
                st.session_state.latest_scaffold = final_answer

                diagram = generate_compartment_flowchart(edited_review)
                st.session_state.latest_flowchart = diagram

            st.success(f"Final reviewed model saved to {FINAL_REVIEWED_MODEL_PATH}")
            render_markdown_with_latex(final_answer)

            st.markdown("### Compartment Flowchart")
            render_mermaid(diagram)