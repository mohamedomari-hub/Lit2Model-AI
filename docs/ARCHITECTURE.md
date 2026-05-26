# Architecture

This project is a Streamlit + LangChain system for scientific PDF question answering and mechanistic model discovery.

## High-Level Flow

PDF files are parsed into text, scientific chunks, figures, tables, and equation candidates. The chunks are embedded into a Chroma vector store. Q/A tools and model discovery both retrieve evidence from that shared store.

```text
PDF -> ingestion/extraction -> chunks -> Chroma vector store -> retrieval wrappers -> Q/A + discovery
```

## Main Folders

`src/chat`

LangChain chat agent and chat tool definitions.

`src/retrieval`

Shared retrieval layer. It contains wrappers for paper text, equations, parameters, mechanisms, tables, figures, simulations, metadata formatting, ranking, and vector store setup.

`src/discovery`

Model discovery workflow. It builds discovery context, calls the LLM, normalizes JSON output, and saves reviewed model evidence.

`src/ingestion`

PDF parsing, OCR, crop rendering, metadata helpers, and scientific artifact indexing.

`src/extraction`

Lower-level PDF text extraction and scientific chunk building.

`src/modelling`

Model generation, validation, simulation planning, graph generation, equation recovery, table extraction, and figure extraction helpers.

`src/llm`

Structured LLM extraction utilities.

`src/schemas`

Pydantic/data schemas for extracted model components.

`src/ui`

Streamlit UI rendering and sidebar helpers.

## Shared Retrieval Layer

The retrieval layer is the common bridge between the PDF vector store and downstream workflows.

Q/A tools call retrieval wrappers such as:

- `search_paper`
- `search_equations`
- `search_parameters`
- `search_mechanisms`
- `search_tables`
- `search_figures`
- `search_simulations`

Model discovery also uses shared retrieval helpers for raw documents, equation candidates, table evidence, and discovery context.

## Compatibility Wrappers

Some older files still exist as compatibility wrappers. They import from newer clearer module names so existing app imports keep working.

Examples:

- `src/agent.py` imports from `src/chat/chat_agent.py`
- `src/tools.py` imports from `src/chat/chat_tools.py`
- `src/rag.py` imports from `src/retrieval/vector_store.py`
- old `src/document/*` files import from `src/ingestion/*`
- old discovery/modeling files import from clearer new modules

## Current Stable Status

Q/A and model discovery now share retrieval capabilities through `src/retrieval`. Existing behavior is preserved through compatibility wrappers.

## Future Cleanup

Later, after full app testing:

- update imports to use the new clearer paths directly
- remove old compatibility wrapper files
- finish moving remaining legacy retrieval helpers into `src/retrieval`
- keep changes small and compile/test after each step
