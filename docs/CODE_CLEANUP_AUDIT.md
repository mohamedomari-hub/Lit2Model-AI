# Code Cleanup Audit

Status values: `ACTIVE`, `WRAPPER`, `UNUSED_CANDIDATE`, `REVIEW_REQUIRED`

| File | Status | Why | Safe to remove? |
|---|---|---|---|
| `src/agent.py` | WRAPPER | Compatibility import for `src.chat.chat_agent`. Imported by `chatbot_app.py`. | NO |
| `src/tools.py` | WRAPPER | Compatibility import for `src.chat.chat_tools`. Imported by `chatbot_app.py` and `src.chat.chat_agent`. | NO |
| `src/rag.py` | WRAPPER | Compatibility import for `src.retrieval.vector_store`. Imported by `chatbot_app.py`. | NO |
| `src/app/config.py` | ACTIVE | Imported by app state, IO, sidebar, and `chatbot_app.py`. | NO |
| `src/app/io.py` | ACTIVE | Imported by `chatbot_app.py` for app file/path helpers. | NO |
| `src/app/state.py` | ACTIVE | Imported by `chatbot_app.py` for workflow reset. | NO |
| `src/app/theme.py` | ACTIVE | Imported and called by `chatbot_app.py`. | NO |
| `src/chat/chat_agent.py` | ACTIVE | Real agent implementation used through `src/agent.py`. | NO |
| `src/chat/chat_tools.py` | ACTIVE | Real LangChain tools implementation used through `src/tools.py`. | NO |
| `src/discovery/rag_controlled_discovery.py` | WRAPPER | Compatibility import for `src.discovery.run_model_discovery`. Imported by `chatbot_app.py`. | NO |
| `src/discovery/run_model_discovery.py` | ACTIVE | Real controlled discovery workflow used through wrapper. | NO |
| `src/discovery/discovery_prompts.py` | ACTIVE | Provides `SYSTEM_PROMPT` for controlled discovery. | NO |
| `src/discovery/discovery_pipeline.py` | REVIEW_REQUIRED | Imported by `chatbot_app.py`, but no call found in current grep results. | MAYBE |
| `src/document/artifact_index.py` | WRAPPER | Compatibility import for `src.ingestion.scientific_assets`. | NO |
| `src/document/crops.py` | WRAPPER | Compatibility import for `src.ingestion.crops`. Used by modelling imports. | NO |
| `src/document/ingestion.py` | WRAPPER | Compatibility import for `src.ingestion.pdf_parser`. Imported by `chatbot_app.py`. | NO |
| `src/document/metadata.py` | WRAPPER | Compatibility import for `src.ingestion.metadata`. | NO |
| `src/document/ocr.py` | WRAPPER | Compatibility import for `src.ingestion.ocr`. Imported by app/retrieval/modelling. | NO |
| `src/document/candidate_classifier.py` | ACTIVE | Imported by `src.ingestion.scientific_assets`. | NO |
| `src/document/extraction.py` | UNUSED_CANDIDATE | No imports or references found. Contains only `text_layer_is_weak`. | YES |
| `src/extraction/pdf_text.py` | ACTIVE | Used by `src.discovery.discovery_pipeline`. | NO |
| `src/extraction/chunk_builder.py` | ACTIVE | Used by `src.discovery.discovery_pipeline`. | NO |
| `src/ingestion/pdf_parser.py` | ACTIVE | Real PDF parser used through document wrapper. | NO |
| `src/ingestion/scientific_assets.py` | ACTIVE | Real artifact index implementation used through document wrapper. | NO |
| `src/ingestion/ocr.py` | ACTIVE | Real OCR implementation used through document wrapper. | NO |
| `src/ingestion/crops.py` | ACTIVE | Real crop rendering implementation used through document wrapper. | NO |
| `src/ingestion/metadata.py` | ACTIVE | Real metadata helpers used through document wrapper. | NO |
| `src/llm/structured_extractor.py` | ACTIVE | Used by `src.discovery.discovery_pipeline`. | NO |
| `src/modelling/model_generator.py` | WRAPPER | Compatibility import for `src.modelling.generate_model`. Imported by app/chat tools. | NO |
| `src/modelling/generate_model.py` | ACTIVE | Real model generation/discovery pipeline implementation. | NO |
| `src/modelling/equation_validation.py` | WRAPPER | Compatibility import for `src.modelling.validate_model`. Imported by `chatbot_app.py`. | NO |
| `src/modelling/validate_model.py` | ACTIVE | Real equation/model validation implementation. | NO |
| `src/modelling/simulation_planner.py` | WRAPPER | Compatibility import for `src.modelling.plan_simulations`. Imported by `chatbot_app.py`. | NO |
| `src/modelling/plan_simulations.py` | ACTIVE | Real simulation requirement inference implementation. | NO |
| `src/modelling/consistency_checks.py` | ACTIVE | Re-exported by `validate_model.py`. | NO |
| `src/modelling/equation_recovery.py` | ACTIVE | Used by retrieval and model generation for numbered equation recovery/OCR. | NO |
| `src/modelling/table_extraction.py` | ACTIVE | Used by `src.retrieval.table_search`. | NO |
| `src/modelling/figure_extraction.py` | ACTIVE | Used by `src.retrieval.figure_search`. | NO |
| `src/modelling/graph_generation.py` | ACTIVE | Used by `src.modelling.generate_model`. | NO |
| `src/retrieval/__init__.py` | ACTIVE | Re-exports retrieval helpers used across project. | NO |
| `src/retrieval/context.py` | ACTIVE | Core semantic retrieval service used by wrappers/tools. | NO |
| `src/retrieval/metadata.py` | ACTIVE | Used by retrieval context and retrieval package exports. | NO |
| `src/retrieval/ranking.py` | ACTIVE | Used by retrieval context and retrieval package exports. | NO |
| `src/retrieval/vector_store.py` | ACTIVE | Real vector store implementation used through `src/rag.py`. | NO |
| `src/retrieval/paper_search.py` | ACTIVE | Shared paper/discovery retrieval helpers. | NO |
| `src/retrieval/equation_search.py` | ACTIVE | Shared equation retrieval and candidate helpers. | NO |
| `src/retrieval/parameter_search.py` | ACTIVE | Shared parameter retrieval wrappers; evidence helper prepared. | NO |
| `src/retrieval/mechanism_search.py` | ACTIVE | Shared mechanism retrieval wrappers; evidence helper prepared. | NO |
| `src/retrieval/table_search.py` | ACTIVE | Shared table retrieval and evidence helpers. | NO |
| `src/retrieval/figure_search.py` | ACTIVE | Shared figure retrieval wrapper. | NO |
| `src/retrieval/simulation_search.py` | ACTIVE | Shared simulation retrieval wrapper. | NO |
| `src/review/compact_formatter.py` | ACTIVE | Used by controlled discovery output. | NO |
| `src/review/evidence_formatter.py` | ACTIVE | Used by `src.discovery.discovery_pipeline`. | NO |
| `src/schemas/__init__.py` | REVIEW_REQUIRED | Re-exports schema classes; no direct app import found. | MAYBE |
| `src/schemas/evidence_schema.py` | ACTIVE | Used by LLM extractor and discovery pipeline. | NO |
| `src/schemas/model_card.py` | REVIEW_REQUIRED | Referenced by `src.schemas.__init__`; no direct workflow import found. | MAYBE |
| `src/schemas/equations.py` | REVIEW_REQUIRED | Referenced by `model_card.py` and schema package exports. | MAYBE |
| `src/schemas/inputs.py` | REVIEW_REQUIRED | Referenced by schema package exports. | MAYBE |
| `src/schemas/mechanisms.py` | REVIEW_REQUIRED | Referenced by `model_card.py` and schema package exports. | MAYBE |
| `src/schemas/observations.py` | REVIEW_REQUIRED | Referenced by schema package exports. | MAYBE |
| `src/schemas/parameters.py` | REVIEW_REQUIRED | Referenced by `model_card.py` and schema package exports. | MAYBE |
| `src/schemas/simulation.py` | REVIEW_REQUIRED | Referenced by schema package exports. | MAYBE |
| `src/schemas/states.py` | REVIEW_REQUIRED | Referenced by `model_card.py` and schema package exports. | MAYBE |
| `src/ui/renderers.py` | ACTIVE | Imported and used by `chatbot_app.py`. | NO |
| `src/ui/sidebar.py` | ACTIVE | Imported and used by `chatbot_app.py`. | NO |

## High Confidence Unused Candidates

| File | Why | Safe to remove? |
|---|---|---|
| `src/document/extraction.py` | No imports or references found in project search. | YES |

## Manual Review Candidates

| File | Why | Safe to remove? |
|---|---|---|
| `src/discovery/discovery_pipeline.py` | Imported by `chatbot_app.py`, but no call found. Removing would require import cleanup first. | MAYBE |
| `src/schemas/model_card.py` | No direct workflow import found; schema may be reserved for future structured output. | MAYBE |
| `src/schemas/equations.py` | Only schema references found. | MAYBE |
| `src/schemas/inputs.py` | Only schema package export found. | MAYBE |
| `src/schemas/mechanisms.py` | Only schema references found. | MAYBE |
| `src/schemas/observations.py` | Only schema package export found. | MAYBE |
| `src/schemas/parameters.py` | Only schema references found. | MAYBE |
| `src/schemas/simulation.py` | Only schema package export found. | MAYBE |
| `src/schemas/states.py` | Only schema references found. | MAYBE |
