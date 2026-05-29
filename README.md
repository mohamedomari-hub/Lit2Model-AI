# Lit2Model-AI
### AI-Powered Scientific Assistant for Model Discovery & Simulation

Lit2Model-AI is an AI-assisted scientific paper assistant for evidence-grounded Q&A and mathematical/mechanistic model discovery. It helps organize fragmented scientific evidence from PDFs into searchable, structured model information and can generate draft simulation-ready Python model scaffolds.

The project is designed for scientific exploration and human-in-the-loop model review. It does not replace expert scientific validation.

---

## Demo + Presentation

Watch the demo here:  
https://drive.google.com/file/d/1p0apsd6rBBpI7IOlqAcIa0hv1kJQMgVS/view?usp=drive_link

Project presentation: `Final_Project_Slides.pdf`

---

## Why Lit2Model-AI?

Mechanistic and mathematical modeling papers are difficult to work with because important model information is scattered across:

- text descriptions
- equations
- parameter tables
- figures and diagrams
- assumptions and methods sections

Researchers often need to manually connect mechanisms, variables, parameters, equations, and simulation assumptions. Lit2Model-AI supports this process with paper-grounded Q&A, structured model discovery, OCR/vision-assisted equation recovery, and a human review workflow.

---

## Key Features

- Scientific paper Q&A grounded in retrieved paper evidence
- Retrieval from text, equations, tables, and figures
- Structured extraction of mechanisms, parameters, equations, states, inputs, assumptions, and missing information
- Human review and validation workflow for extracted model JSON
- OCR/vision-assisted equation recovery for difficult PDF equations
- Simulation setup support based on reviewed model information
- LLM-assisted Python model scaffold generation
- Project-specific outputs and cached artifacts

---

## Architecture

```text
Scientific Paper PDF
        ↓
Multimodal Parsing
Docling / PyMuPDF4LLM / PyMuPDF
        ↓
Chunking + Metadata
        ↓
Embeddings
BAAI/bge-small-en-v1.5 or optional OpenAI embeddings
        ↓
ChromaDB Vector Store
        ↓
 ┌──────────────────────────┬────────────────────────────┐
 │ Q&A Agentic RAG          │ Structured Model Discovery │
 │ - retrieval tools        │ - model evidence retrieval │
 │ - evidence-grounded chat │ - structured extraction    │
 └──────────────────────────┴────────────────────────────┘
                                        ↓
                                Human Review
                                        ↓
                                Simulation Setup
                                        ↓
                                LLM-Assisted Python Model Scaffold
```

Architecture notes:

- Q&A mode uses agentic RAG.
- Model discovery uses a structured retrieval/extraction pipeline.
- Python model generation is LLM-assisted code generation.
- Human review is required before simulation or decision-making.

---

## Technology Stack

| Area | Tools |
|---|---|
| App | Streamlit |
| Agent / orchestration | LangChain, LangGraph |
| Vector database | ChromaDB |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5`, optional OpenAI embeddings |
| PDF parsing | Docling, PyMuPDF4LLM, PyMuPDF / `fitz` |
| LLMs | OpenAI API, optional Gemini API |
| OCR / vision | OpenAI vision, optional Gemini vision |
| Structured data | Pydantic |
| Scientific utilities | SymPy, NumPy, SciPy, NetworkX, PyVis |
| Monitoring | LangSmith |

---

## Project Structure

```text
lit2model-ai-v7/
├── chatbot_app.py              # Main Streamlit entry point
├── README.md
├── requirements.txt
├── .env.example
├── Final_Project_Slides.pdf
└── src/
    ├── app/                    # UI helpers, theme, state, sidebar, renderers
    ├── chat/                   # Q&A agent and retrieval tool wrappers
    ├── paper_processing/       # PDF parsing, OCR, crops, chunking, assets
    ├── retrieval/              # Vector store, search, ranking, evidence retrieval
    ├── discovery/              # Model discovery workflow, prompts, formatters
    ├── modeling/               # Validation, equation recovery, simulation/code generation
    └── schemas/                # Pydantic schemas for structured model data
```

Main files:

| File | Purpose |
|---|---|
| `chatbot_app.py` | Streamlit app and workflow routing |
| `src/chat/chat_agent.py` | Q&A agent prompt and tool orchestration |
| `src/chat/chat_tools.py` | Retrieval tools exposed to the Q&A agent |
| `src/retrieval/vector_store.py` | ChromaDB creation/loading and embeddings |
| `src/discovery/run_model_discovery.py` | Controlled model discovery pipeline |
| `src/modeling/generate_model.py` | LLM-assisted Python model scaffold generation |
| `src/modeling/model_validation.py` | Reviewed JSON/model readiness checks |
| `src/paper_processing/pdf_parser.py` | PDF parsing using Docling/PyMuPDF tools |

Generated files such as uploaded PDFs, ChromaDB stores, OCR crops, and model outputs are written to ignored local folders like `uploads/`, `outputs/`, `data/`, and `chroma_db/`.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run chatbot_app.py
```

Then open the Streamlit URL, upload a scientific PDF, process it, and use the workflow pages.

---

## Environment Variables

Create a `.env` file from `.env.example`:

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
GOOGLE_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=lit2model-ai
OPENROUTER_API_KEY=
```

Required:

- `OPENAI_API_KEY`

Optional:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` for Gemini-based OCR/vision support
- `LANGSMITH_API_KEY` and LangSmith settings for tracing
- `OPENROUTER_API_KEY` for optional OpenRouter-compatible models

---

## Example Workflow

1. Upload or select a scientific PDF.
2. Parse the paper and extract text, tables, figures, equations, and metadata.
3. Build or load the ChromaDB vector store.
4. Ask paper questions using evidence-grounded Q&A.
5. Run structured model discovery.
6. Review and validate the extracted JSON model draft.
7. Recover or correct equations when needed using OCR/vision support.
8. Infer simulation requirements.
9. Generate a draft Python model scaffold.
10. Review, edit, and run the generated simulation code.

---

## Current Scope & Limitations

Lit2Model-AI is an MVP and research assistant, not a fully automatic scientific modeling system.

Current limitations:

- Generated models are candidate scaffolds, not validated scientific models.
- OCR/vision can fail on complex equations, dense figures, or low-quality scans.
- Papers may omit assumptions, parameter values, equation details, or model context.
- Retrieved evidence can be incomplete or ambiguous.
- Human scientific review is required before simulation, interpretation, or decision-making.

---

## Long-Term Vision

The long-term goal is to move beyond extraction: from scientific papers to structured, simulation-ready digital models.

Lit2Model-AI is a step toward an AI assistant for mathematical modeling that can:

- understand biological and mechanistic interactions
- identify variables, equations, and assumptions
- propose model structures
- support human model review
- help simulate intervention scenarios

---

## Data & Privacy

Users upload their own PDFs. Copyrighted example papers are not included in the repository.

Generated artifacts, uploaded PDFs, vector stores, and local outputs are ignored by Git through `.gitignore`.

---

## Author

**Mohamed Omari, PhD**  
Quantitative Modelling • Systems Biology • AI Engineering  
GitHub: github.com/mohamedomari-hub

---

## Disclaimer

Lit2Model-AI supports scientific exploration, evidence-grounded Q&A, and model discovery, but does not replace expert scientific validation. Health-related or mechanistic models require human review before simulation or decision-making.
