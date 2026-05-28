# Lit2Model-AI

Lit2Model-AI is a Streamlit-based AI assistant for transforming scientific modelling papers into structured, simulation-ready mechanistic model drafts.

The app combines PDF parsing, retrieval-augmented question answering, mechanistic model discovery, OCR-assisted equation recovery, human-in-the-loop JSON review, simulation setup, and Python model generation.

## Main Features

- Upload and parse scientific PDF papers.
- Ask paper-grounded questions using an agent with retrieval tools.
- Retrieve evidence about equations, parameters, tables, figures, mechanisms, and simulation settings.
- Automatically run mechanistic model discovery.
- Extract states, equations, parameters, inputs, observations, and mechanisms.
- Review and edit the structured JSON model.
- Use OCR/vision to recover missing or corrupted equations.
- Add missing parameters and inputs manually; edits are saved back into JSON.
- Infer simulation requirements.
- Generate simulation-ready Python code.
- Run simulations and download outputs.

## Project Structure

```text
src/
  app/               Streamlit UI support, config, state, theme, rendering
  chat/              Q/A agent and retrieval tools
  paper_processing/  PDF parsing, OCR, crops, chunks, and paper assets
  retrieval/         Chroma vector store, search wrappers, ranking, metadata
  discovery/         Controlled model discovery, prompts, structured extraction
  modeling/          Validation, equation recovery, simulation planning, code generation
  schemas/           Pydantic schemas for structured model data
```

## Folder Overview

### app/

Contains Streamlit app support code such as configuration, file I/O, session state, theme, sidebar navigation, and rendering helpers.

### chat/

Contains the LangChain Q/A agent and tool definitions. The agent routes user questions to specialized retrieval tools such as text, equation, parameter, table, figure, mechanism, and simulation retrieval.

### paper_processing/

Handles scientific PDF processing. It parses PDFs, extracts text, builds chunks, creates equation crops, supports OCR/vision extraction, and builds paper artifact indexes.

### retrieval/

Contains the shared retrieval layer. It builds and loads the Chroma vector database, retrieves relevant paper evidence, ranks scientific context, and provides specialized search modules.

### discovery/

Contains the mechanistic model discovery workflow. It retrieves model-relevant evidence and extracts structured information such as states, ODEs, process modules, parameters, inputs, observations, mechanisms, and missing simulation items.

### modeling/

Contains model validation, equation recovery, simulation planning, graph generation, and Python model generation.

### schemas/

Contains Pydantic schemas for structured scientific model data, including equations, parameters, states, mechanisms, inputs, observations, simulations, and evidence records.

## End-to-End Workflow

```text
PDF
  ↓
Paper parsing and processing
  ↓
Text, equations, tables, figures, crops, and metadata
  ↓
Vector database
  ↓
Retrieval tools
  ↓
Q/A agent or model discovery
  ↓
Structured JSON model draft
  ↓
Human review and validation
  ↓
Simulation setup
  ↓
Generated Python model
  ↓
Simulation outputs
```

## App Workflow

### 1. Upload / Load PDF

The user uploads a scientific paper. The app parses the PDF and creates project-specific outputs.

### 2. Ask Paper Questions

The user can ask questions such as:

- What is the aim of the paper?
- How does dexamethasone affect glucose metabolism?
- How is the drug effect modeled?
- Explain Equation 1.
- What parameters are reported?

### 3. Run Model Discovery

The app extracts a structured mechanistic model draft from retrieved evidence, including equations, states, parameters, inputs, mechanisms, and observations.

### 4. Review & Validate Model

The user reviews the JSON draft, corrects extracted content, uses OCR/vision for equation recovery, and adds missing inputs or parameters. Manual edits are saved back into the JSON model.

### 5. Simulation Setup

The app infers simulation requirements such as states, initial conditions, parameters, inputs, and time settings.

### 6. Generate Python Model

The reviewed model is converted into Python code. The user can edit the code, choose a solver, run the simulation, and download outputs.

## Main Entry Point

Run the app with:

```bash
streamlit run chatbot_app.py
```

## Key Technologies

- Streamlit for the user interface.
- LangChain for agent and tool orchestration.
- OpenAI API for LLM calls, speech, and vision/OCR support.
- Chroma for local vector search.
- HuggingFace embeddings for local document embeddings.
- PyMuPDF, PyMuPDF4LLM, and Docling for PDF parsing.
- Pydantic for structured schemas.
- SymPy for equation parsing support.
- Mermaid, Graphviz, PyVis, and NetworkX for diagrams and graph visualization.

## Outputs

The app creates project-specific outputs under `outputs/projects/`, including:

```text
chroma_db/
extracted_evidence.json
reviewed_model_draft.json
final_reviewed_model.json
missing_equations.json
generated_model.py
simulation_requirements.json
equation_candidates/
equation_pages/
cache/ocr/
```

## Known Limitations

- Extracted models require human review.
- OCR and equation recovery may need manual correction.
- LLM extraction can miss or misclassify scientific details.
- Generated Python models are drafts and require scientific validation.
- Retrieval quality depends on PDF parsing quality.
- Complex tables, scanned PDFs, and corrupted equation text remain challenging.

## Project Vision

Lit2Model-AI explores how retrieval, agents, OCR, structured extraction, and human review can help move from scientific papers to executable mechanistic models.

The long-term goal is to support workflows that transform:

```text
scientific literature
→ structured mechanistic model
→ simulation-ready Python code
```