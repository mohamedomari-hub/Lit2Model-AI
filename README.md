# Lit2Model-AI
### *From Scientific Literature to Mechanistic Models*

**Lit2Model-AI** is a multimodal AI assistant designed to transform **scientific papers into mechanistic model candidates**.

Instead of manually reading dozens of pages to identify **mechanisms, parameters, equations, feedback loops, and model assumptions**, Lit2Model-AI helps researchers extract biologically meaningful information and generate **simulation-ready model scaffolds**.

The system combines:

- **Multimodal PDF understanding** (text, tables, figures)
- **Retrieval-Augmented Generation (RAG)**
- **Agentic AI workflows**
- **Mechanism extraction**
- **Candidate ODE model generation**
- **Scientific paper Q&A**

The long-term vision is to support **Quantitative Systems Pharmacology (QSP)**, **PK/PD**, and **mechanistic systems biology modelling** by accelerating the transition from:

```text
Scientific Paper
        ↓
Biological Understanding
        ↓
Mechanistic Structure
        ↓
Candidate Simulation Model
```

---

## Why this project?

Building mechanistic models (QSP, PK/PD, PBPK, systems biology models) is time-consuming.

Researchers often spend:

- days or weeks reading literature
- manually extracting mechanisms
- identifying parameters
- reconstructing equations
- connecting biological pathways

Scientific papers are often fragmented:

- mechanisms are explained in text
- parameters are hidden in tables
- structures are shown in figures
- assumptions are scattered across sections

Lit2Model-AI aims to assist this process by acting as an:

> **AI Copilot for Mechanistic Model Discovery**

---

## Project Goal

The goal is **not** to automatically replace scientific modelling.

Instead, the system assists researchers by:

### 1. Extracting mechanisms

Example:

```text
Dexamethasone
        ↓ stimulates
Glucagon secretion
        ↓
Blood glucose
        ↓
Insulin
```

---

### 2. Extracting parameters

Example:

```text
ka = 13.4352 1/day
ke = 2.7086 1/day
F = 72%
Vd = 1.105 L/kg
keo = 0.7 1/day
```

---

### 3. Extracting equations

Example:

```math
\frac{dCe}{dt}=keo(C-Ce)
```

or pharmacodynamic coupling equations such as:

```math
Effect(Ce)
```

---

### 4. Generating candidate model scaffolds

The assistant proposes a **draft mechanistic model structure** that researchers can later refine and validate.

Example outputs:

- state variables
- parameters
- equations
- assumptions
- missing information
- human-review notes

---

### 5. Chatting with scientific papers

Example questions:

```text
What feedback loops exist?

What assumptions were made?

Explain the effect compartment.

Which parameters were estimated?

What mechanisms connect metabolism and reproduction?
```

---

## Example Scientific Use Case

This project is currently benchmarked using a **mechanistic endocrine–metabolic dairy cow model (MetRep + Dexamethasone administration)** developed during a PhD project.

The benchmark model contains:

- endocrine feedback loops
- glucose–insulin–glucagon dynamics
- reproductive hormone regulation
- PK/PD dexamethasone effects
- effect compartments
- nonlinear Hill functions
- feedback thresholds

This provides a **known mechanistic ground truth** to evaluate whether the AI successfully reconstructs:

- mechanisms
- state variables
- equations
- parameters
- biological interactions

---

# Features

## Multimodal PDF Parsing

The system extracts:

### Text

Scientific explanations, methods, assumptions.

### Tables

Parameter tables and model coefficients.

### Figures

Mechanistic diagrams, PK/PD schemes, model structures.

Scientific figures can optionally be interpreted using **vision-capable LLMs**.

---

## Retrieval-Augmented Generation (RAG)

Instead of sending the entire paper to an LLM, Lit2Model-AI uses:

### ChromaDB vector retrieval

to search relevant sections of the paper.

The system performs:

### Multi-query retrieval

instead of relying on one broad search.

Example retrieval targets:

```text
PK equations
Effect compartment
Parameter tables
Feedback loops
Pharmacodynamic coupling
Biological mechanisms
Model diagrams
```

This improves scientific recall and reduces missing mechanisms.

---

## Agentic Workflow

Lit2Model-AI uses an **agent-based workflow** to orchestrate model extraction.

Current workflow:

```text
Scientific PDF
        ↓
Multimodal parser
(text + tables + figures)
        ↓
Vector database (ChromaDB)
        ↓
LangChain Agent
        ↓
------------------------------------
| Retrieve scientific evidence     |
| Extract model information        |
| Extract mechanism relationships  |
| Generate candidate model         |
------------------------------------
        ↓
Scientific output
```

The current implementation uses:

### 1 orchestration agent

with specialized tools that simulate:

- literature review
- mechanism extraction
- model extraction
- candidate ODE generation

---

## Architecture

```text
Scientific Paper (PDF)
            ↓
     Multimodal Parser
 ┌─────────┬──────────┬──────────┐
 │ Text    │ Tables   │ Figures  │
 └─────────┴──────────┴──────────┘
            ↓
       Chunking
            ↓
       Embeddings
            ↓
        ChromaDB
            ↓
       LangChain Agent
            ↓
 ┌──────────────────────────┐
 │ retrieve context         │
 │ extract mechanisms       │
 │ extract equations        │
 │ generate model scaffold  │
 └──────────────────────────┘
            ↓
     Candidate Model
```

---

## Example Outputs

### Mechanism extraction

```text
Dexamethasone
        ↓ stimulates
Glucagon secretion
        ↓
Blood glucose
        ↓
Insulin secretion
```

---

### Candidate model extraction

```text
State variables
Parameters
Equations
Assumptions
Missing information
```

---

### Candidate Python model scaffold

Example generated output:

```python
dCdt = (...)
dCedt = keo * (C - Ce)
```

---

## Future Vision

The long-term goal is to evolve Lit2Model-AI toward:

> **AI-assisted Quantitative Systems Pharmacology (QSP)**

Future capabilities may include:

- automatic mechanism graph generation
- interactive biological flowcharts
- parameter inference support
- PubMed integration
- Reactome pathway integration
- DrugBank integration
- SBML export
- simulation-ready model generation
- surrogate modelling support
- human-in-the-loop QSP workflows

# Technology Stack

Lit2Model-AI combines **multimodal document understanding**, **retrieval-augmented generation (RAG)**, and **agentic AI workflows**.

## Core Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM Framework | LangChain |
| Agent Orchestration | LangChain Agents |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace / OpenAI |
| PDF Parsing | PyMuPDF (`fitz`) |
| Vision Understanding | OpenAI Vision |
| Structured Output | Pydantic |
| Monitoring | LangSmith |
| UI (planned) | Streamlit |

---

# Models Used

Different models are used for different tasks.

This allows balancing:

```text
quality
cost
speed
```

## Current Model Architecture

| Script | Task | Current Model | Cost |
|---|---:|---:|---:|
| `rag.py` | document embeddings | `BAAI/bge-small-en-v1.5` | Free |
| `rag.py` (optional) | higher-quality embeddings | `text-embedding-3-small` | Paid |
| `parser.py` | scientific figure understanding | `gpt-4o-mini` | Paid |
| `agent.py` | agent orchestration | `gpt-4o-mini` | Paid |
| `tools.py` | structured model extraction | `gpt-4o-mini` | Paid |
| `tools.py` | mechanism graph extraction | `gpt-4o-mini` | Paid |
| `tools.py` | candidate ODE generation | `gpt-4o-mini` | Paid |

---

## Development vs Final Demo Models

### Development Mode (Cheap)

Recommended while coding/debugging:

| Component | Model |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vision | OFF |
| Agent controller | `gpt-3.5-turbo` |
| Structured extraction | `gpt-4o-mini` |

Goal:

```text
cheap development
minimal API cost
fast iteration
```

---

### Final Demo Mode (Best Quality)

Recommended before presentation:

| Component | Model |
|---|---|
| Embeddings | `text-embedding-3-small` |
| Vision | ON |
| Agent | `gpt-4o-mini` |
| Extraction | `gpt-4o-mini` |

Goal:

```text
best scientific extraction
higher-quality retrieval
better figure understanding
```

---

# Cost Expectations

## Does running `streamlit run chatbot_app.py` always cost money?

### No.

The project was designed with **caching** to avoid repeated costs.

---

## When does it cost money?

| Step | Cost? |
|---|---:|
| Load existing ChromaDB | ❌ No |
| Local HuggingFace embeddings | ❌ No |
| OpenAI embeddings | ✅ Yes |
| Figure understanding (`USE_VISION=True`) | ✅ Yes |
| Agent execution | ✅ Yes |
| Loading cached outputs | ❌ No |

---

## First Run vs Later Runs

### First Run

The first run may cost money because the system needs to:

```text
parse PDF
create embeddings
run the agent
extract mechanisms
generate candidate model
```

---

### Later Runs

Later runs are usually almost free because:

```text
ChromaDB already exists
figure descriptions are cached
agent output already exists
```

The system simply reloads:

```text
chroma_db/
outputs/agent_result.md
```

instead of calling the API again.

---

# Cost-Saving Design

The project includes multiple cost-saving strategies.

## 1. Local Embeddings

Development mode uses:

```text
BAAI/bge-small-en-v1.5
```

instead of OpenAI embeddings.

Benefits:

```text
FREE
runs locally
good retrieval quality
```

---

## 2. Figure Description Caching

Scientific figures are cached.

The first time:

```text
image
        ↓
gpt-4o-mini
        ↓
description saved as .txt
```

Later runs reuse:

```text
saved description
```

No repeated vision cost.

---

## 3. Cached Agent Outputs

After running once:

```text
outputs/agent_result.md
```

is saved.

Later runs:

```text
reuse existing output
```

instead of re-calling the agent.

---

# Project Structure

```text
lit2model-ai/
│
├── src/
│   ├── parser.py
│   ├── rag.py
│   ├── tools.py
│   ├── agent.py
│   ├── schemas.py
│   └── model_generator.py
│
├── data/
│   └── sample_papers/
│
├── chroma_db/
│
├── outputs/
│   ├── agent_result.md
│   ├── candidate_model.py
│   └── extracted_images/
│
│
├── .env
│
├── requirements.txt
│
└── README.md
```

---

# Installation

Clone repository:

```bash
git clone https://github.com/your-username/lit2model-ai.git

cd lit2model-ai
```

Create environment:

```bash
conda create -n lit2model python=3.10

conda activate lit2model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create:

```text
.env
```

Add:

```env
OPENAI_API_KEY=your_openai_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=lit2model-ai
```

---

# Running the System

## Development Mode

Recommended during coding.

### Step 1

In:

```python
src/rag.py
```

set:

```python
USE_OPENAI_EMBEDDINGS = False
```

---

### Step 2

In:

```python
src/parser.py
```

set:

```python
USE_VISION = False
```

---

### Step 3

Run:

```bash
streamlit run chatbot_app.py
```

---

Expected output:

```text
PDF exists: True

Loading existing ChromaDB...

Building agent...

Running agent...

Done.

Agent result saved
Candidate model saved
```

---

## Force Fresh Extraction

Delete caches:

```bash
rm -rf chroma_db

rm outputs/agent_result.md

rm outputs/extracted_images/*.txt
```

Then rerun:

```bash
streamlit run chatbot_app.py
```

Useful when:

```text
changing embedding model
updating prompts
turning vision ON
```

---

# Output Files

After running:

## Scientific extraction summary

```text
outputs/agent_result.md
```

Contains:

```text
mechanisms
parameters
equations
candidate model
missing information
human-review notes
```

---

## Candidate model scaffold

```text
outputs/candidate_model.py
```

Contains:

```text
starter ODE structure
candidate equations
PK/effect-compartment scaffold
```

---

## Extracted figures

```text
outputs/extracted_images/
```

Contains:

```text
cropped scientific figures
cached descriptions
```

# Monitoring & Debugging with LangSmith

Lit2Model-AI uses **LangSmith** for monitoring and debugging agent workflows.

LangSmith helps visualize:

```text
tool calls
retrieval steps
LLM reasoning chains
latency
errors
cost
prompt behavior
```

This is especially useful for **agentic workflows** where multiple tools interact.

---

## Why LangSmith?

Without monitoring:

```text
Why did the model miss a mechanism?
Why was a parameter ignored?
Which retrieval chunks were used?
Did the wrong tool get called?
```

are difficult to debug.

LangSmith makes the hidden workflow visible.

---

## Example Workflow Trace

A typical run may look like:

```text
chatbot_app.py
    ↓
agent.invoke()
    ↓
retrieve_model_building_context()
    ↓
extract_structured_model_info()
    ↓
extract_mechanism_edges()
    ↓
propose_candidate_ode_model()
```

You can inspect:

```text
retrieved chunks
prompt content
tool outputs
LLM responses
runtime
```

---

## LangSmith Setup

Create an account:

 [oai_citation:0‡smith.langchain.com](https://smith.langchain.com/?utm_source=chatgpt.com)

Add to:

```text
.env
```

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=lit2model-ai
```

Then run:

```bash
streamlit run chatbot_app.py
```

Open dashboard:

```text
lit2model-ai
```

You should see traces for each run.

---

# Troubleshooting

## 1. Missing OpenAI credentials

Error:

```text
Missing credentials
Incorrect API key
```

Fix:

Check:

```text
.env
```

contains:

```env
OPENAI_API_KEY=your_key
```

---

## 2. ChromaDB embedding mismatch

Error:

```text
retrieval becomes poor
weird results
missing context
```

Cause:

You switched embedding models:

```text
OpenAI → HuggingFace
or
HuggingFace → OpenAI
```

but reused old embeddings.

Fix:

Delete:

```bash
rm -rf chroma_db
```

Then rerun:

```bash
streamlit run chatbot_app.py
```

---

## 3. Figure extraction missing mechanisms

Cause:

```python
USE_VISION = False
```

This skips scientific figure understanding.

Fix:

In:

```python
src/parser.py
```

change:

```python
USE_VISION = True
```

Then remove cached descriptions:

```bash
rm outputs/extracted_images/*.txt
```

and rerun.

---

## 4. Agent keeps asking for PDF

Cause:

The system prompt is incorrect.

Fix:

Ensure `agent.py` contains:

```text
The PDF has already been uploaded,
parsed, embedded, and stored in the vector database.
Do not ask the user to provide the PDF.
```

---

## 5. Poor scientific extraction

Possible causes:

```text
broad retrieval query
missing figure understanding
insufficient chunk retrieval
paper complexity
```

Fixes:

```text
enable vision
improve multi-query retrieval
increase retrieval k
refine prompts
```

---

## 6. Agent does not rerun

Cause:

Cached output exists:

```text
outputs/agent_result.md
```

Fix:

Delete:

```bash
rm outputs/agent_result.md
```

Then rerun.

---

# Current Limitations

This project is still an **MVP (Minimum Viable Product)**.

Current limitations include:

## No validated model reconstruction

The generated model is:

```text
candidate scaffold
```

and **not a validated mechanistic model**.

Human scientific review is required.

---

## Scientific ambiguity

Papers often:

```text
omit equations
hide assumptions
report incomplete parameters
describe mechanisms informally
```

The assistant may therefore produce:

```text
missing information
assumptions
human-review notes
```

---

## Figure understanding limitations

Scientific figures may contain:

```text
small text
complex arrows
dense diagrams
```

which remain difficult even for vision models.

---

## No automatic equation solver

The system currently:

```text
extracts equations
proposes candidate models
```

but does **not yet simulate full models automatically**.

---

## Limited biological databases

The MVP currently works only on:

```text
uploaded PDF content
```

Future versions may integrate:

```text
PubMed
Reactome
STRING
UniProt
DrugBank
ChEMBL
PathwayCommons
```

for stronger scientific grounding.

---

# Future Roadmap

Planned improvements:

## Mechanism Flowchart Generation

Automatic graph generation:

```text
Dexamethasone
        ↓ stimulates
Glucagon
        ↓ increases
Blood glucose
```

using:

```text
NetworkX
PyVis
Plotly
```

---

## Streamlit Interface

Interactive UI for:

```text
PDF upload
chat with paper
mechanism graph
candidate model
downloads
```

---

## Scientific Paper Chatbot

Ask:

```text
What assumptions were made?

Explain the feedback loops.

Which parameters were estimated?

What equations govern glucose dynamics?
```

---

## Database Integration

Scientific augmentation through:

```text
PubMed
DrugBank
UniProt
Reactome
```

to improve biological grounding.

---

## Candidate ODE Code Generation

Future versions may generate:

```text
Python
MATLAB
SBML
Julia
```

simulation-ready model templates.

---

## QSP Copilot

Long-term vision:

> **AI-assisted Quantitative Systems Pharmacology**

Supporting:

```text
mechanistic modelling
PK/PD
PBPK
systems biology
pharmacometrics
```

through:

```text
literature extraction
mechanism discovery
parameter identification
model generation
human-in-the-loop refinement
```

---

# Why This Project Matters

This project sits at the intersection of:

```text
AI Engineering
Mechanistic Modelling
Scientific Discovery
Quantitative Systems Pharmacology
```

Rather than replacing scientists, the goal is to:

> **augment mechanistic reasoning and accelerate scientific model development**

by reducing manual literature extraction work.

---

# Author

**Mohamed Omari, PhD**

Quantitative Modelling • Systems Biology • AI Engineering

Areas of interest:

```text
Mechanistic modelling
QSP / PK-PD / PBPK
AI for life sciences
LLM agents
Scientific RAG systems
Mechanistic–AI hybrid systems
```

GitHub:

```text
github.com/mohamedomari-hub
```

---

# Citation

If you use this repository, please cite or acknowledge:

```text
Lit2Model-AI:
An AI-assisted multimodal literature-to-model workflow
for mechanistic model discovery.
```

---

# Disclaimer

This project is intended for:

```text
research assistance
scientific exploration
mechanistic hypothesis generation
```

and **must not be considered a substitute for expert scientific modelling or validation**.