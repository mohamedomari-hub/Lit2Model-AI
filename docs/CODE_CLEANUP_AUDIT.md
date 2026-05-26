# Code Cleanup Audit

Status values: `ACTIVE`, `UNUSED_CANDIDATE`, `REVIEW_REQUIRED`

## Current File Status

| File | Status | Why | Safe to remove? |
|---|---|---|---|
| `src/app/config.py` | ACTIVE | Imported by app IO, state, sidebar, and `chatbot_app.py`. | NO |
| `src/app/io.py` | ACTIVE | Imported by `chatbot_app.py`. | NO |
| `src/app/state.py` | ACTIVE | Imported by `chatbot_app.py`. | NO |
| `src/app/theme.py` | ACTIVE | Imported and called by `chatbot_app.py`. | NO |
| `src/chat/chat_agent.py` | ACTIVE | Imported by `chatbot_app.py`; builds Q/A agent. | NO |
| `src/chat/chat_tools.py` | ACTIVE | Imported by `chatbot_app.py` and chat agent. | NO |
| `src/discovery/discovery_prompts.py` | ACTIVE | Imported by `run_model_discovery.py`. | NO |
| `src/discovery/run_model_discovery.py` | ACTIVE | Imported by `chatbot_app.py`; controlled discovery entry point. | NO |
| `src/discovery/discovery_pipeline.py` | REVIEW_REQUIRED | Imported by `chatbot_app.py`; no call found by grep. | MAYBE |
| `src/document/candidate_classifier.py` | ACTIVE | Imported by `src.ingestion.scientific_assets`. | NO |
| `src/extraction/chunk_builder.py` | ACTIVE | Used by `src.discovery.discovery_pipeline`. | NO |
| `src/extraction/pdf_text.py` | ACTIVE | Used by `src.discovery.discovery_pipeline`. | NO |
| `src/ingestion/crops.py` | ACTIVE | Used by equation/figure extraction. | NO |
| `src/ingestion/ocr.py` | ACTIVE | Used by app, retrieval, discovery, and modelling. | NO |
| `src/ingestion/pdf_parser.py` | ACTIVE | Imported by `chatbot_app.py`; main PDF parser. | NO |
| `src/ingestion/scientific_assets.py` | ACTIVE | Used by model generation and candidate classifier. | NO |
| `src/ingestion/metadata.py` | UNUSED_CANDIDATE | No imports or references found in current source grep. | YES |
| `src/llm/structured_extractor.py` | ACTIVE | Used by `src.discovery.discovery_pipeline`. | NO |
| `src/modelling/consistency_checks.py` | ACTIVE | Re-exported by `validate_model.py`. | NO |
| `src/modelling/equation_recovery.py` | ACTIVE | Used by retrieval/model generation for equation recovery. | NO |
| `src/modelling/figure_extraction.py` | ACTIVE | Used by `src.retrieval.figure_search`. | NO |
| `src/modelling/generate_model.py` | ACTIVE | Used by chat tools and `chatbot_app.py`. | NO |
| `src/modelling/graph_generation.py` | ACTIVE | Used by `src.modelling.generate_model`. | NO |
| `src/modelling/plan_simulations.py` | ACTIVE | Imported by `chatbot_app.py`. | NO |
| `src/modelling/table_extraction.py` | ACTIVE | Used by `src.retrieval.table_search`. | NO |
| `src/modelling/validate_model.py` | ACTIVE | Imported by `chatbot_app.py`. | NO |
| `src/retrieval/__init__.py` | ACTIVE | Re-exports retrieval helpers used by project modules. | NO |
| `src/retrieval/context.py` | ACTIVE | Core semantic retrieval service. | NO |
| `src/retrieval/equation_search.py` | ACTIVE | Shared equation retrieval/candidate helpers. | NO |
| `src/retrieval/figure_search.py` | ACTIVE | Shared figure retrieval wrapper. | NO |
| `src/retrieval/mechanism_search.py` | ACTIVE | Shared mechanism retrieval helpers. | NO |
| `src/retrieval/metadata.py` | ACTIVE | Used by retrieval context and retrieval exports. | NO |
| `src/retrieval/paper_search.py` | ACTIVE | Shared paper/discovery retrieval helpers. | NO |
| `src/retrieval/parameter_search.py` | ACTIVE | Shared parameter retrieval helpers. | NO |
| `src/retrieval/ranking.py` | ACTIVE | Used by retrieval context and exports. | NO |
| `src/retrieval/simulation_search.py` | ACTIVE | Shared simulation retrieval wrapper. | NO |
| `src/retrieval/table_search.py` | ACTIVE | Shared table retrieval/evidence helpers. | NO |
| `src/retrieval/vector_store.py` | ACTIVE | Imported by `chatbot_app.py`; Chroma/vector store setup. | NO |
| `src/review/compact_formatter.py` | ACTIVE | Used by controlled discovery output. | NO |
| `src/review/evidence_formatter.py` | ACTIVE | Used by `src.discovery.discovery_pipeline`. | NO |
| `src/schemas/evidence_schema.py` | ACTIVE | Used by LLM extractor and discovery pipeline. | NO |
| `src/schemas/__init__.py` | REVIEW_REQUIRED | Re-exports schema classes; no direct current import found. | MAYBE |
| `src/schemas/model_card.py` | REVIEW_REQUIRED | No direct current workflow import found. | MAYBE |
| `src/schemas/equations.py` | REVIEW_REQUIRED | Referenced by model card/schema package only. | MAYBE |
| `src/schemas/inputs.py` | REVIEW_REQUIRED | Referenced by schema package only. | MAYBE |
| `src/schemas/mechanisms.py` | REVIEW_REQUIRED | Referenced by model card/schema package only. | MAYBE |
| `src/schemas/observations.py` | REVIEW_REQUIRED | Referenced by schema package only. | MAYBE |
| `src/schemas/parameters.py` | REVIEW_REQUIRED | Referenced by model card/schema package only. | MAYBE |
| `src/schemas/simulation.py` | REVIEW_REQUIRED | Referenced by schema package only. | MAYBE |
| `src/schemas/states.py` | REVIEW_REQUIRED | Referenced by model card/schema package only. | MAYBE |
| `src/ui/renderers.py` | ACTIVE | Imported and used by `chatbot_app.py`. | NO |
| `src/ui/sidebar.py` | ACTIVE | Imported and used by `chatbot_app.py`. | NO |

## High Confidence Unused Candidates

| File | Why | Safe to remove? |
|---|---|---|
| `src/ingestion/metadata.py` | No imports or references found in current source grep. | YES |

## Review Required

| File | Why | Safe to remove? |
|---|---|---|
| `src/discovery/discovery_pipeline.py` | Imported by `chatbot_app.py`, but no call found by grep. | MAYBE |
| `src/schemas/__init__.py` | Schema package export; no direct current import found. | MAYBE |
| `src/schemas/model_card.py` | No direct current workflow import found. | MAYBE |
| `src/schemas/equations.py` | Only schema references found. | MAYBE |
| `src/schemas/inputs.py` | Only schema package reference found. | MAYBE |
| `src/schemas/mechanisms.py` | Only schema references found. | MAYBE |
| `src/schemas/observations.py` | Only schema package reference found. | MAYBE |
| `src/schemas/parameters.py` | Only schema references found. | MAYBE |
| `src/schemas/simulation.py` | Only schema package reference found. | MAYBE |
| `src/schemas/states.py` | Only schema references found. | MAYBE |

## Notes

- `src/document/extraction.py` no longer exists in `src`; it was archived earlier.
- `src/document/` currently contains `candidate_classifier.py` plus cache files.
- Old compatibility wrapper files have been removed from `src`.
- Some old wrapper paths remain mentioned in docs/README only, not active Python imports.
