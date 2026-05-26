## Figure 9 Debugging Lesson: Retrieval Worked, Interpretation Failed

- **Initial issue:** When asking “Explain Figure 9,” the answer looked weak and said that little information was available. At first, this seemed like a retrieval failure.

- **Trace finding:** LangSmith showed that `retrieve_pdf_context` did retrieve Figure 9-related chunks. The retrieved text included `picture intentionally omitted` and `Start of picture text`, meaning the parser had extracted OCR text from the figure.

- **Real problem:** The retrieved content was noisy OCR text, such as axis labels, units, and variable names, not a clean figure caption. The LLM treated this fragmented OCR evidence as insufficient context.

- **Fix direction:** The prompt should tell the model to interpret OCR-derived figure text cautiously: extract visible axes, variables, units, and labels; separate direct OCR evidence from interpretation; and clearly mention visual limitations.

- **Key lesson:** Good retrieval does not always produce good answers. In scientific RAG, we must distinguish between retrieval failure, OCR/representation problems, and reasoning failure.


## Latest Improvement: Figure-Specific Retrieval

We improved the paper Q&A system by adding a dedicated **figure-retrieval pipeline**.

Previously, when asking questions such as **“Explain Figure 9”**, the system often responded that figure information was limited, even though OCR-derived figure text had already been retrieved from the PDF. The issue was **not retrieval failure**, but poor interpretation of noisy OCR evidence.

### Before Improvement

The workflow was:

```text
User question
↓
retrieve_pdf_context()
↓
OCR figure text retrieved
↓
LLM treated OCR evidence as insufficient
↓
weak / generic answer
```

Typical behavior:

- overly cautious responses
- generic statements
- “limited information available”
- failure to use OCR labels meaningfully
- poor scientific interpretation of figures

Example behavior:

```text
“There is limited information regarding Figure 9...”
```

Even though the system had actually retrieved:

- axis labels
- variable names
- units
- timing information
- OCR-derived figure text

---

### After Improvement

We added a dedicated architecture:

```text
Figure question
↓
explain_figure_from_pdf()
↓
retrieve_figure_context()
↓
extract_figure_explanation()
↓
scientific figure explanation
```

The system now retrieves and interprets:

- figure captions
- OCR-extracted figure text
- axis labels
- units
- legends
- nearby figure discussion

### Improved LLM Behavior

✅ **Uses OCR-derived context intelligently**

Instead of ignoring OCR figure text, the model now extracts scientific signals such as:

- variables (e.g., IGF-I, LH, E2)
- axis labels
- timing information
- drug administration events
- units and biological quantities

It uses this evidence to build a cautious scientific explanation.

---

✅ **Scientifically cautious tone**

The LLM no longer hallucinates figure content.

Instead, it clearly separates:

```text
Directly Retrieved Evidence
Interpretation
Limitations
```

For example, rather than inventing curve shapes or trends, it explicitly states when the visual figure is unavailable and limits interpretation to retrieved OCR evidence.

This makes the assistant more scientifically reliable and better aligned with mechanistic modelling workflows.

### Key Lesson

```text
Retrieval success ≠ reasoning success
```

Scientific RAG requires:

```text
task-specific retrieval
+
task-specific interpretation
```

A figure may be retrieved successfully, but the LLM still needs specialized reasoning instructions to interpret OCR-derived scientific evidence correctly.