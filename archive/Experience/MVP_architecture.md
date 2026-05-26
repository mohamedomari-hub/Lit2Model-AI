# Lit2Model-AI: Current RAG Architecture

The current Lit2Model-AI system follows a **task-specific Retrieval-Augmented Generation (RAG)** architecture designed for scientific papers and mechanistic model extraction.

Rather than using a single generic retriever for all tasks, the platform separates retrieval depending on the scientific objective (general Q&A, parameter extraction, equation extraction, model discovery, and figure explanation).

---

## High-Level Architecture

```text
Scientific PDF
      ↓
Multimodal Parsing (text + OCR + figures + tables)
      ↓
Chunking + Metadata
(page, modality, section index)
      ↓
Embedding Model (BAAI/bge-small-en-v1.5)
      ↓
Chroma Vector Database
      ↓
Task-specific Retrieval
      ↓
LLM Reasoning / Structured Extraction
```

---

## 1. General Paper Q&A RAG

Used in:

```text
Ask paper questions
```

Pipeline:

```text
User question
      ↓
retrieve_pdf_context()
      ↓
multi-query retrieval
      ↓
retrieved scientific evidence
      ↓
LLM answer grounded in paper
```

Purpose:
- explain equations
- explain mechanisms
- explain tables
- answer scientific questions
- retrieve evidence from the paper

The retriever adapts to the query type:

```text
Equation query → equation-focused retrieval
Table query → table-focused retrieval
Parameter query → parameter-focused retrieval
Figure query → figure-aware retrieval
General query → semantic retrieval
```

---

## 2. Model Discovery RAG

Used in:

```text
Run model discovery
```

Pipeline:

```text
run_model_discovery_workflow()
        ↓
retrieve_parameter_context()
retrieve_equation_context()
retrieve_mechanism_context()
        ↓
structured extraction
        ↓
scientific sanity checking
        ↓
candidate mechanistic scaffold
```

Purpose:
- reconstruct mechanistic model structure
- identify parameters and units
- extract ODEs and coupling equations
- infer mechanisms and graph edges
- generate draft model scaffold

Unlike Q&A retrieval, this workflow prioritizes **structured model reconstruction** rather than explanation.

---

## 3. Figure-Specific RAG

Used for:

```text
Explain Figure 9
What does Figure 3 show?
```

Pipeline:

```text
User asks about figure
        ↓
explain_figure_from_pdf() [tool]
        ↓
retrieve_figure_context()
        ↓
OCR + caption + nearby text retrieval
        ↓
extract_figure_explanation()
        ↓
scientific figure explanation
```

Purpose:
- interpret plots and diagrams
- explain retrieved figure evidence
- use OCR-derived labels (axes, variables, units)
- separate direct evidence from interpretation

Important design lesson:

```text
retrieval success ≠ reasoning success
```

A figure may be retrieved successfully through OCR text, but the LLM still needs figure-aware reasoning instructions.

---

## 4. Scientific Review Layer

Used in:

```text
Review with chatbot
Manual review editor
```

Pipeline:

```text
Reviewed extraction
        ↓
scientific_review_edit()
        ↓
sanity_check_reviewed_extraction()
        ↓
final reviewed scaffold
```

Purpose:
- correct extraction mistakes
- fix OCR errors
- validate scientific consistency
- mark uncertain equations for review

This introduces a **human-in-the-loop scientific validation step**, which is essential for mechanistic modelling.

---

## Design Philosophy

The system intentionally uses **task-specific retrieval instead of one generic retriever**.

Why?

Because scientific tasks require different evidence:

```text
Figure explanation → captions + OCR + nearby text
Equation extraction → displayed equations + symbols
Parameter extraction → tables + units + values
Mechanism discovery → biological relationships
General Q&A → flexible semantic retrieval
```

This improves robustness and reduces hallucination during scientific model reconstruction.


PDF
│
├── PyMuPDF4LLM
│      → text / captions / tables / some equation text
│
├── Gemini OCR fallback
│      → equation images only (when parser failed)
│
├── PyMuPDF image extraction
│      → crop figure images from pages
│
├── Hidden PDF text recovery
│      → figure labels / axis text / legends
│      (NOT true OCR)
│
└── OpenAI GPT-4o-mini (optional)
       → actual visual understanding of figures

Gemini OCR is currently used only as a fallback for missing equation images, while figure interpretation relies on embedded PDF text recovery and optional GPT-4o-mini visual understanding rather than OCR.


The pipline becomes like this:
PDF
├── PyMuPDF4LLM → text / captions / tables / equations
├── Gemini → equation OCR fallback
├── PyMuPDF → crop figure images
└── Gemini vision → figure interpretation


# Lit2Model-AI Current Architecture

```text
PDF
│
├── PyMuPDF4LLM
│      → Parse text, captions, tables, equations
│      → Create searchable chunks + metadata
│
├── Chroma Vector Store (RAG)
│      → Retrieve relevant context
│      → figures / equations / parameters / mechanisms
│
├── PyMuPDF
│      → Render target PDF page as image
│      → Only when visual context is needed
│
├── Gemini Fallback (On-demand)
│      → Figure understanding
│      → Missing/omitted equations
│      → Triggered only when retrieval is incomplete
│
└── LLM Reasoning Layer (GPT-4o-mini)
       → Combine retrieved evidence
       → Scientific explanation
       → Cautious interpretation
       → Human-review notes when uncertain
```

### Figure Workflow

```text
User asks: "Explain Figure 9"
        ↓
retrieve_figure_context()
        ↓
RAG retrieves OCR/caption context
        ↓
If figure incomplete/omitted
        ↓
PyMuPDF renders target page
        ↓
Gemini visually interprets page
        ↓
GPT combines:
OCR + retrieved context + Gemini vision
        ↓
Scientific figure explanation
```