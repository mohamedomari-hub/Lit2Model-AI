# Model Discovery Stability Report

## Problem

The output of **Run model discovery** changes across runs, even when using the same paper and the same vector database.

Examples observed:

- Different reported mechanisms
- Different parameter interpretations
- ODEs extracted in one run but missing in another
- Different graph edges
- Different wording of the scaffold
- Different OCR interpretation of equations

This creates instability and reduces reproducibility.

---

# Root Causes of Variability

## 1. Retrieval Variability (Major Cause)

Current implementation uses:

```python
VECTOR_STORE.similarity_search(
    query,
    k=6
)
```

Similarity retrieval ranking is not always deterministic.

Small embedding ranking differences can cause:

Run 1:
- Chunk A retrieved

Run 2:
- Chunk B retrieved

This changes the context passed to downstream extraction.

### Consequences

Different chunks lead to:

- Different equations retrieved
- Different parameters
- Different mechanisms
- Different final scaffold

---

## 2. LLM Non-Determinism

Even with:

```python
temperature=0
```

LLM outputs are still not perfectly deterministic.

Small differences in retrieval order or context lead to different interpretations.

Examples:

Run 1:
> Dexamethasone impairs insulin sensitivity

Run 2:
> Dexamethasone decreases glucose uptake

Scientifically related but inconsistent wording.

---

## 3. OCR Variability

Equation OCR (GPT Vision / pix2tex / Gemini) is probabilistic.

The same equation image may produce:

Run 1:

```text
Cb^7
```

Run 2:

```text
Cb^Cb
```

Small cropping or rendering differences amplify OCR instability.

---

## 4. Context Ordering Sensitivity

Multiple retrieved chunks are concatenated.

Current structure:

```python
"\n---\n".join(results)
```

If retrieval order changes, the LLM receives context in a different sequence.

LLMs are sensitive to context order.

Earlier chunks disproportionately influence outputs.

---

# Recommended Improvements

## 1. Stabilize Retrieval Order

After retrieval, sort documents deterministically.

Recommended:

```python
docs = sorted(
    docs,
    key=lambda d: (
        d.metadata.get("page", 999),
        d.metadata.get("section_index", 999)
    )
)
```

### Benefits

- More reproducible retrieval
- More stable context ordering
- More consistent model discovery output

---

## 2. Reduce LLM Interpretation Freedom

Prompts should emphasize:

> Copy only explicitly supported scientific evidence.

Avoid broad summarization.

Prefer:

```text
Extract only explicitly reported mechanisms.
Do not infer missing biology.
Copy equations exactly.
```

Instead of:

```text
Explain mechanisms broadly.
```

### Benefits

- More reproducible outputs
- Less hallucination
- Better scientific rigor

---

## 3. Cache Discovery Results

Current issue:

Each time the user runs:

```text
Run model discovery
```

the pipeline reruns extraction.

Instead:

### Recommended behavior

If:

- same PDF
- same vector store
- same extraction file exists

then show:

```text
Latest discovery result found.

Use existing result
or
Rerun discovery?
```

### Benefits

- Saves tokens
- Saves API cost
- Improves reproducibility
- Avoids user confusion

---

## 4. Treat OCR as Candidate Evidence Only

OCR equations should never become source-of-truth automatically.

Workflow:

```text
OCR candidate
↓
Human review
↓
Validated reviewed model
↓
Simulation / code generation
```

Important principle:

> OCR-derived equations require human validation before simulation.

---

## 5. Use Reviewed Model as Source of Truth

Never simulate from:

```text
outputs/extraction_review.md
```

Instead:

```text
outputs/reviewed_model.md
```

should drive:

- simulation setup
- Python model generation

This preserves scientific safety.

---

# Recommended Workflow

```text
Upload paper
↓
Build vector store (once)
↓
Run model discovery
↓
outputs/extraction_review.md
(raw AI draft)

↓
Review & Validate Model
(edit + chatbot + corrections)

↓ Save draft
outputs/reviewed_model_draft.md

↓ Validate
outputs/reviewed_model.md
(scientific source of truth)

↓
Simulation setup

↓
Generate Python Model

↓
outputs/generated_model.py
```

---

# Key Design Principle

AI helps discover.

Human validates.

Simulation and code generation should only use validated scientific information.