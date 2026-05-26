## Lit2Model-AI Development Experience

During the development of Lit2Model-AI, the project evolved from a simple scientific PDF chatbot into a broader AI-assisted mechanistic model reconstruction platform. The original idea was to upload a modelling paper, retrieve relevant content, extract equations and parameters, and generate a candidate model scaffold. Early experiments showed that this was much harder than expected, especially because scientific PDFs often contain equations as images, fragmented symbols, table-based parameters, figure captions, and incomplete model descriptions. Standard PDF parsing frequently returned outputs such as “picture intentionally omitted,” which meant that important ODEs, coupling functions, and pharmacodynamic equations were missing. This led to the introduction of a hybrid extraction strategy combining text retrieval with Gemini OCR as a fallback for visually embedded equations.

A major learning point was that OCR alone is not enough. OCR can read symbols from a paper, but it does not guarantee that the mathematical meaning is correct. For example, superscripts, subscripts, and Hill-function exponents can be misread, producing expressions such as `Ce^Ca` instead of a biologically meaningful exponent. This showed that full automation is unsafe for mechanistic modelling. As a result, the platform was redesigned around a human-in-the-loop workflow: extraction first, then modeller review, then final scaffold generation. The review interface allows the user to correct equations, mark uncertain OCR-derived expressions, remove wrong graph edges, and regenerate a cleaner final scaffold.

Several engineering problems also appeared during development. Streamlit reruns caused outputs to disappear when switching between modes, so persistent `session_state` variables were added for the latest scaffold, flowchart, discovery result, reviewed extraction, and generated Python model. Another issue was that equations were sometimes displayed incorrectly because Markdown interpreted underscores as italics or the LLM returned broken multiline mathematical text. This led to improvements in LaTeX rendering, stricter prompt rules for equation formatting, and safer handling of OCR-derived equations. The review chatbot also initially rewrote equations too aggressively when the user gave vague feedback such as “this equation is weird.” This was corrected by instructing the system to preserve equations unless the user provides an explicit correction, and otherwise mark them as requiring human review.

The mechanism graph and flowchart components also went through several iterations. Initially, the system generated incorrect or over-inferred graph edges, such as reversing compartment directions or converting vague biological statements into causal relationships. The graph extraction prompts were therefore made more conservative, requiring explicit evidence from equations, model descriptions, or figure captions. The flowchart started as a simple text diagram, but was later upgraded to a Mermaid-based visual diagram rendered in Streamlit. A key design decision was to avoid hardcoding PK/PD-specific diagrams and instead generate model-agnostic flowcharts from the reviewed extraction, so the platform can eventually support PK/PD, PBPK, QSP, epidemiological, metabolic, ecological, and signalling models.

The platform was then extended toward executable modelling. A model generator was added to convert reviewed equations and user-provided simulation settings into Python code using `numpy`, `scipy.integrate.solve_ivp`, and `matplotlib`. This introduced new challenges: the LLM sometimes returned English explanations such as “Here is the code” inside the generated `.py` file, causing Python syntax errors. The code-generation pipeline was therefore hardened by asking for raw executable Python only and cleaning markdown fences or explanatory text before saving the file. The next layer added solver selection and simulation execution from the Streamlit UI, allowing the user to choose solvers such as LSODA, BDF, Radau, RK45, and others, then run the generated model and view plots or download CSV results.

Overall, the main lesson was that building an AI system for mechanistic modelling is not just a prompt-engineering task. It requires careful architecture: retrieval, OCR fallback, conservative extraction, human review, scientific sanity checking, state management, visualisation, code generation, and simulation execution. The project showed that the most realistic and scientifically credible direction is not “paper to perfect model automatically,” but rather “paper to candidate mechanistic model with transparent uncertainty and modeller-in-the-loop correction.” This makes Lit2Model-AI a practical AI-assisted modelling copilot rather than a black-box model generator. Future improvements could include symbolic equation parsing with SymPy, automatic equation validation, sensitivity analysis, uncertainty propagation, identifiability diagnostics, and calibration against experimental data.



####################################################################


# Lit2Model-AI: Lessons Learned from Building an AI-Assisted Mechanistic Model Reconstruction Platform

## Motivation and Initial Vision

Lit2Model-AI emerged from a practical scientific problem: **mechanistic modelling papers are difficult to reproduce**. In fields such as PK/PD, QSP, systems biology, and epidemiology, mathematical models are rarely reported in a clean, structured format. Equations may be fragmented across methods sections, tables, figures, supplementary files, or embedded as images in PDFs. Parameters may be partially reported, variable names inconsistent, and assumptions hidden in narrative descriptions.

The original goal was ambitious:

```text
Scientific paper → automatic mechanistic model reconstruction
```

The idea was to upload a scientific modelling paper and automatically extract:

- state variables
- parameters
- ODEs
- algebraic equations
- mechanisms
- interaction graphs
- executable simulation code

Initially, the assumption was that a strong LLM combined with OCR would largely solve the problem. However, development quickly showed that **scientific model extraction is fundamentally different from ordinary document question answering**.

The challenge was not obvious hallucinations, but rather **scientifically plausible mistakes** that looked convincing while being wrong.

The architecture gradually evolved into:

```text
PDF
↓
Parser + OCR fallback
↓
RAG retrieval
↓
LLM scientific extraction
↓
Human review
↓
Scientific sanity checking
↓
Final mechanistic scaffold
↓
Python model generation
↓
Simulation
```

A major realization during development was:

> In scientific AI systems, plausible ≠ correct.

---

## Major Technical Challenges and How We Solved Them

### 1. Hallucinated Mechanism Direction (Reverse Causality)

One of the earliest failures involved **incorrect biological directionality**.

For example, the paper reported the effect-compartment equation:

$$
\frac{dC_e}{dt}
=
k_{eo}(C-C_e)
$$

Scientifically, this means:

```text
Central compartment → Effect compartment
```

However, the LLM initially generated:

> “The effect compartment transports to the central compartment.”

At first glance, this looked scientifically reasonable. However, mechanistically it was wrong and would fundamentally alter the simulation structure.

#### Why this happened

The LLM inferred causal direction from surrounding scientific language instead of respecting the mathematical structure of the equation.

In other words:

```text
LLM optimized for plausibility,
not mechanistic causality.
```

#### Solution

The extraction prompt was redesigned to become much stricter.

**Before**

```text
Extract graph-ready biological mechanisms.
```

**After**

```text
Do not infer reverse directions.

If an equation supports:
Central → Effect

Do not rewrite it as:
Effect → Central
```

Uncertain graph edges were flagged as:

```text
(requires human review)
```

instead of being accepted automatically.

#### Lesson learned

> Scientific extraction systems must be conservative. It is safer to preserve uncertainty than invent confidence.

---

### 2. Mixing Biological Background with Model Mechanisms

Another major issue was that the LLM often mixed **background biology** with **mechanistic assumptions explicitly implemented in the model**.

For example, scientific papers often contain statements such as:

> “Glucagon contributes to glucose metabolism.”

The LLM sometimes converted these statements into mechanistic graph edges even when they were **not part of the mathematical model**.

This produced outputs like:

```text
Glucagon stimulates glucose uptake
```

even though the actual model equations described:

```text
Dexamethasone impairs insulin sensitivity
```

#### Why this happened

Scientific papers contain multiple information layers:

1. biological explanation  
2. experimental context  
3. implemented mathematical mechanisms

The model initially treated all scientific text equally.

#### Solution

Mechanism extraction was restricted to only include relationships explicitly supported by:

- equations
- coupling hypotheses
- model diagrams
- simulation assumptions

A new rule was introduced:

```text
Do NOT extract general biological discussion
unless explicitly implemented in the model.
```

#### Lesson learned

> Biological knowledge is not the same thing as model structure.

---

### 3. Parameters Misclassified as Equations

Another subtle failure involved **scientific notation ambiguity**.

The model frequently interpreted parameter descriptions as equations.

For example:

```text
Emax = maximum stimulatory effect
Ca = concentration for half-maximal stimulation
Cb = concentration for half-maximal inhibition
```

were incorrectly classified as:

```text
Reported algebraic equations
```

instead of parameter definitions.

This generated strange outputs such as:

```text
Emax = maximum stimulatory effect of drug
```

inside the mathematical equation section.

#### Why this happened

The LLM treated:

```text
X = something
```

as:

```text
mathematical equation
```

instead of recognizing:

```text
scientific parameter definition
```

#### Solution

The extraction schema was redesigned to explicitly separate:

```text
reported_parameters
```

from:

```text
reported_algebraic_equations
```

Additional guardrails were added:

```text
These are NOT equations:
- Emax = Maximum stimulatory effect
- Ca = Half-maximal concentration
- Cb = Half-maximal inhibition parameter
```

#### Lesson learned

> Scientific notation can be syntactically mathematical while semantically descriptive.

---

### 4. OCR Corrupting Equations

One of the hardest technical problems was **equation corruption caused by OCR**.

A Hill-function equation was occasionally extracted incorrectly.

Correct equation:

$$
1+
\frac{E_{max} C_e^{10}}
{C_e^{10}+C_a^{10}}
$$

OCR occasionally produced:

$$
1+
\frac{E_{max} C_e^{C_a}}
{C_e^{C_a}+C_a^{C_a}}
$$

This mistake was especially dangerous because:

```text
the equation still looked mathematically valid
```

The generated Python model would run successfully, but the biological meaning would be wrong.

#### Why this happened

OCR systems struggle with:

- superscripts
- subscripts
- Greek symbols
- scanned PDFs
- embedded equations

The LLM trusted corrupted OCR too easily.

#### Solution

A **human review layer** became mandatory.

Instead of automatically trusting extracted equations, suspicious equations were flagged:

```text
(requires human review)
```

Users could correct equations through:

```text
Review with chatbot
```

For example:

User feedback:

```text
This equation looks weird
```

Previously:

```text
LLM rewrote equation aggressively
```

After redesign:

```text
Keep original equation
+
(requires human review / OCR uncertain)
```

#### Lesson learned

> OCR reads symbols. It does not understand mathematics.

This led to an important conceptual distinction:

```text
OCR = eyes
LLM = scientific assistant
Human = scientific validator
```

---

## Engineering Challenges During Development

Several practical software engineering problems also emerged.

### Streamlit synchronization problems

Switching between modes such as:

```text
Run model discovery
→ Ask paper questions
→ Review with chatbot
```

sometimes caused outputs to disappear.

Examples included:

- flowcharts disappearing
- scaffolds resetting
- reviewed extraction becoming blank
- state inconsistencies after reruns

#### Solution

Persistent synchronization was added using:

```python
st.session_state
```

including variables such as:

```python
latest_scaffold
latest_flowchart
reviewed_extraction
model_discovery_result
generated_python_model
```

This allowed outputs to persist across interface changes and prevented users from losing work.

#### Lesson learned

> LLM systems are not only about prompts — state management is equally important.

---

## From Extraction to Executable Simulation

A major evolution of the platform was moving from:

```text
paper understanding
```

to:

```text
model execution
```

The platform gradually evolved toward:

```text
Reviewed extraction
↓
Simulation requirements inference
↓
Human correction of missing inputs
↓
Python model generation
↓
solve_ivp simulation
↓
Plots + outputs
```

The idea became:

> Let the agent reconstruct the model, but allow the user to intervene whenever simulation assumptions are missing.

For example, users could manually provide:

- missing parameter values
- initial conditions
- dosing values
- simulation horizon
- solver choice

This made the platform more generalizable beyond a single PK/PD case study.

---

## Why Human-in-the-loop Became Essential

Initially, the vision was:

```text
Fully automated reconstruction
```

However, repeated failures demonstrated that mechanistic modelling requires expert oversight.

The platform evolved toward:

```text
AI-assisted reconstruction
```

instead of:

```text
AI replacement of scientific judgement
```

The final workflow became:

```text
Run model discovery
↓
Review extracted equations/mechanisms
↓
Apply scientific corrections
↓
Run sanity check
↓
Generate scaffold
↓
Generate Python model
↓
Simulate
```

This proved much more reliable.

---

## Final Lessons Learned

Several important lessons emerged during development:

- **Plausible does not mean scientifically correct**
- **Biological background is not model structure**
- **OCR can silently corrupt equations**
- **LLMs tend to over-infer scientific meaning**
- **Uncertainty should be preserved, not hidden**
- **Human review is essential for mechanistic modelling**
- **Scientific AI should assist modellers, not replace them**

The biggest conceptual shift was moving from:

```text
Can AI automatically reconstruct a model?
```

to:

```text
How can AI accelerate model reconstruction
while preserving scientific reliability?
```

This became the central design philosophy of Lit2Model-AI.


#####################################

## Why We Used Multiple Retrieval Functions

In Lit2Model-AI, we separated retrieval into **general Q&A retrieval** and **specialized model-extraction retrieval**.

`retrieve_pdf_context()` is the general retriever used when the user asks open questions such as “Explain Figure 9”, “What is in Table 1?”, or “What assumptions does the model make?”. It adapts dynamically: if the query mentions a figure, table, parameter, or equation, it adds extra search queries to retrieve the most relevant context.

By contrast, functions like `retrieve_parameter_context()` and `retrieve_equation_context()` are specialized retrievers used during the model discovery workflow. `retrieve_parameter_context()` focuses on parameter tables, values, units, and sources, while `retrieve_equation_context()` focuses on ODEs, algebraic equations, Hill functions, coupling equations, and effect-compartment equations.

This design avoids using one generic retriever for everything. Instead, each scientific task gets the type of evidence it needs:

```text
General user question → retrieve_pdf_context()
Parameter extraction → retrieve_parameter_context()
Equation extraction → retrieve_equation_context()
Model discovery → specialized retrieval pipeline