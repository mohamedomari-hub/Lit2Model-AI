# Improvements for Numbered Equation Retrieval (`Equation X`)

## Problem We Observed

When the user asked:

```text
Explain equation 8
```

the system initially:

- failed to retrieve the exact equation,
- retrieved semantically related chunks,
- and sometimes hallucinated a generic PK/PD equation.

Example failure:

```text
"It typically incorporates..."
```

This was unsafe scientifically.

---

## Key Improvements Implemented

### 1. Added Exact Numbered Equation Routing

Inside `retrieve_pdf_context()`:

We detect equation-specific queries and route them to a dedicated retriever.

Supported formats:

```text
equation 8
Equation 8
Eq 8
Eq. 8
equation (8)
formula 8
ODE 8
model equation 8
```

Routing logic:

```text
retrieve_pdf_context()
        ↓
retrieve_equation_by_number_context()
```

---

### 2. Added Exact Equation Matching

We introduced:

```python
is_exact_equation_match()
```

Purpose:

Prevent semantic retrieval from confusing:

```text
Equation 8
```

with:

```text
Figure 7
Figure 9
random model chunks
```

The retriever now explicitly searches for:

```text
Equation 8
Eq. 8
Eq 8
(8)
```

Only exact matches are prioritized.

---

### 3. Split Retrieval Into Two Buckets

We separated retrieval into:

```python
exact_results
fallback_results
```

Logic:

```text
If exact equation found
    → use exact retrieval only

Else
    → return fallback semantic context
```

This prevents hallucinated explanations.

---

### 4. Added Gemini OCR Fallback

If exact retrieval fails:

```text
Chroma retrieval fails
        ↓
Gemini OCR fallback
        ↓
Search OCR output for Equation X
```

Workflow:

```text
Search Chroma first
        ↓
Exact equation found?
        ↓
No
        ↓
Run Gemini equation OCR
        ↓
Look for:
(8)
Equation 8
Eq. 8
        ↓
Return OCR-derived equation
```

This solved the major retrieval issue.

---

## Important Lesson Learned

Gemini OCR fallback improved retrieval,
but OCR-derived equations still require human validation.

Example issue observed:

Wrong OCR:

```text
C_b^{C_b}
```

Correct equation:

```text
C_b^7
```

Meaning:

```text
Retrieval succeeded
≠
Equation is guaranteed correct
```

OCR can still misread:

- superscripts
- subscripts
- exponents
- symbols
- equation numbering

---

## 5. Added OCR Reliability Rules

Inside:

```python
extract_equations()
```

we added scientific caution:

### OCR Reliability Rule

- OCR-derived equations may contain symbol, exponent, or subscript errors.
- If an exponent looks suspicious (e.g., `Cb^Cb`, `Ce^Ce`, malformed superscripts), do not assume correctness.
- Mark as:

```text
requires human review
```

- Prefer preserving OCR output with warnings rather than silently correcting equations.
- If context strongly suggests a correction, label it:

```text
possible interpretation (requires review)
```

---

## Final Equation Retrieval Architecture

```text
User asks:
"Explain equation 8"

        ↓

retrieve_pdf_context()

        ↓
(numbered equation detected)

retrieve_equation_by_number_context()

        ↓

Chroma exact retrieval
(search: Equation 8, Eq. 8, (8))

        ↓
Exact match found?
        ↓
YES → explain equation

        ↓ NO

Gemini OCR fallback

        ↓

Equation detected in OCR?
        ↓
YES → explain cautiously
       + requires review if suspicious

        ↓ NO

Say:
"No exact Equation 8 was found."
```

## Main Scientific Improvement

Before:

```text
No retrieval
→ LLM guessed equation
```

Now:

```text
No retrieval
→ OCR fallback
→ cautious interpretation
→ human review if suspicious
```

This makes the system significantly more scientifically reliable.





###########################

## Equation Retrieval & OCR Experiment Summary

We evaluated different strategies for extracting scientific equations from mechanistic modelling papers in Lit2Model-AI. Standard PDF text-layer retrieval proved unreliable because mathematical notation (fractions, superscripts, subscripts, and symbols) frequently broke during parsing, leading to incomplete or corrupted equations. Simply switching to a stronger reasoning model (e.g., GPT-4o) did not fully solve the issue because the model sometimes inferred or normalized equations rather than preserving the exact scientific notation.

A targeted OCR fallback using GPT-4o-mini Vision substantially improved equation recovery. Instead of relying solely on text retrieval, the system rendered a crop of the equation from the PDF and asked the model to transcribe it exactly using a constrained scientific OCR prompt. This recovered mathematical structure far more accurately while remaining inexpensive. LangSmith traces confirmed the agent workflow as: **model → retrieval tool → model explanation**, showing that repeated retrieval alone does not guarantee better evidence if the same chunks are returned.

The main lesson learned is that scientific equation extraction requires a specialized pipeline rather than standard RAG. The preferred architecture is now: **PDF text extraction first → targeted OCR fallback only when needed → evidence-grounded explanation with human review flags for uncertain equations**. Using **GPT-4o-mini as both agent brain and OCR engine provided a strong cost-performance tradeoff**, achieving much better equation quality at very low cost while maintaining scientific traceability.