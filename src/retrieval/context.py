from src.retrieval.metadata import format_doc_metadata
from src.retrieval.ranking import make_chunk_key, sort_docs_for_discovery


SEMANTIC_CONTEXTS = {
    "text": {
        "label": "Text Query",
        "templates": [],
        "k": 6,
    },
    "state": {
        "label": "State Query",
        "templates": [
            "state variables compartments dynamic variables initial conditions",
            "ordinary differential equations state variables compartments",
            "derived variables algebraic variables model quantities",
        ],
        "entity": "{entity} state variable initial condition",
        "k": 5,
    },
    "equation": {
        "label": "Equation Query",
        "templates": [
            "ordinary differential equations ODE mathematical model",
            "algebraic equations coupling functions forcing terms",
            "equation symbols state variables derived variables",
        ],
        "k": 6,
    },
    "parameter": {
        "label": "Parameter Query",
        "templates": [
            "model parameters values units estimated fixed reported table",
            "parameter estimates initial conditions values units",
            "rate constants coefficients EC50 IC50 Emax Hill units",
        ],
        "entity": "{entity} parameter value unit",
        "k": 5,
    },
    "input": {
        "label": "Input Query",
        "templates": [
            "dose administration intervention input forcing function timing units",
            "time varying covariate external input target state equation",
            "drug administration feeding nutrition scenario simulation input",
        ],
        "entity": "{entity} input dose timing units",
        "k": 5,
    },
    "observation": {
        "label": "Observation Query",
        "templates": [
            "observed measured biomarker fitted variable output data units",
            "observation equation mapping model output measurement",
            "data used for fitting validation measurements",
        ],
        "entity": "{entity} observation measured output units",
        "k": 5,
    },
    "mechanism": {
        "label": "Mechanism Query",
        "templates": [
            "biological mechanism stimulation inhibition feedback regulation",
            "compartment interactions causal relationship model mechanism",
            "Hill threshold Emax feedback effect mechanism",
        ],
        "entity": "{entity} mechanism stimulation inhibition feedback",
        "k": 5,
    },
    "table": {
        "label": "Table Query",
        "templates": [
            "table caption rows columns values units parameters",
            "supplementary table initial conditions experimental settings",
            "parameter table estimated fixed reported values units",
        ],
        "entity": "Table {entity} caption values units",
        "k": 6,
    },
    "simulation": {
        "label": "Simulation Query",
        "templates": [
            "simulation solver time grid duration initial conditions scenario",
            "numerical tolerances integration solver time units",
            "model simulation settings dose scenario initial condition",
        ],
        "k": 5,
    },
    "assumption": {
        "label": "Assumption Query",
        "templates": [
            "model assumptions simplifications limitations ignored pathways",
            "steady state assumption fixed values assumed equal",
            "missing information uncertainty requires review model",
        ],
        "k": 5,
    }
}


def multi_query_retrieve(vector_store, queries, label: str, k: int = 4) -> str:
    """
    Run multiple retrieval queries and deduplicate chunks across all results.
    """

    if vector_store is None:
        return "Vector store is not initialized."

    all_results = []
    seen = set()

    for query in queries:
        docs = vector_store.similarity_search(query, k=k)
        docs = sort_docs_for_discovery(docs)

        for doc in docs:
            chunk_key = make_chunk_key(doc)

            if chunk_key in seen:
                continue

            seen.add(chunk_key)

            content = doc.page_content

            if "Start of picture text" in content:
                content = (
                    "[FIGURE OCR TEXT: visual figure was omitted by parser, "
                    "but OCR extracted visible labels, axes, legends, or units.]\n"
                    + content
                )

            header = format_doc_metadata(
                doc=doc,
                search_query=query,
                label=label,
            )
            all_results.append(f"{header}\n{content}")

    if not all_results:
        return "No relevant context was retrieved from the PDF."

    return "\n\n---\n\n".join(all_results)


def retrieve_context_with_templates(
    vector_store,
    query: str,
    label: str,
    templates: list[str],
    entity_query: str | None = None,
    k: int = 5,
) -> str:
    queries = [query, *templates]

    if entity_query:
        queries.insert(0, entity_query)

    return multi_query_retrieve(
        vector_store=vector_store,
        queries=list(dict.fromkeys(queries)),
        label=label,
        k=k,
    )


def retrieve_semantic_context(
    vector_store,
    query: str,
    context_type: str,
    entity: str | None = None,
    k: int | None = None,
) -> str:
    config = SEMANTIC_CONTEXTS[context_type]
    entity_query = None

    if entity and config.get("entity"):
        entity_query = config["entity"].format(entity=entity)

    if not config["templates"] and not entity_query:
        return multi_query_retrieve(
            vector_store=vector_store,
            queries=[query],
            label=config["label"],
            k=k or config["k"],
        )

    return retrieve_context_with_templates(
        vector_store=vector_store,
        query=query,
        label=config["label"],
        templates=config["templates"],
        entity_query=entity_query,
        k=k or config["k"],
    )


def retrieve_equation_context_service(
    vector_store,
    pdf_path: str | None,
    query: str,
    equation_number: str | None = None,
    ocr_mode: str = "auto",
) -> str:
    import re

    from src.modelling.equation_recovery import retrieve_numbered_equation_context

    if equation_number:
        return retrieve_numbered_equation_context(
            vector_store=vector_store,
            pdf_path=pdf_path,
            query=f"{query} equation {equation_number} ocr_mode={ocr_mode}",
        )


    equation_match = re.search(
        r"\b(?:equation|equa|eqn|eq\.?)\s*\(?\s*(\d+)\s*\)?",
        query.lower(),
    )

    if equation_match:
        return retrieve_numbered_equation_context(
            vector_store=vector_store,
            pdf_path=pdf_path,
            query=query,
        )

    context = retrieve_semantic_context(
        vector_store=vector_store,
        query=query,
        context_type="equation",
    )

    return (
        context
        + "\n\nReview metadata: exact equations from OCR or weak text-layer "
        "extraction require human review before simulation."
    )
