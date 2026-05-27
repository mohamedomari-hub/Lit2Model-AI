def make_chunk_key(doc):
    metadata = doc.metadata or {}

    return (
        metadata.get("source_pdf", "unknown_pdf"),
        metadata.get("page", "unknown"),
        metadata.get("modality", "unknown"),
        metadata.get("content_type", "unknown_type"),
        metadata.get("section_index", "unknown"),
        doc.page_content[:300],
    )


def metadata_order_value(value, default=999):
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sort_docs_for_discovery(docs):
    return sorted(
        docs,
        key=lambda doc: (
            metadata_order_value((doc.metadata or {}).get("page")),
            metadata_order_value((doc.metadata or {}).get("section_index")),
            (doc.metadata or {}).get("chunk_id", ""),
            doc.page_content[:80],
        ),
    )


def score_scientific_evidence(text: str, evidence_type: str = "general") -> int:
    """
    Lightweight rule-based score for scientific modelling evidence.
    Higher score means the chunk is more likely to contain useful model information.
    """
    text_lower = (text or "").lower()

    general_keywords = [
        "equation",
        "parameter",
        "table",
        "symbol",
        "value",
        "unit",
        "model",
        "state",
        "variable",
        "defined as",
        "where",
        "estimated",
        "fitted",
        "fixed",
        "calibrated",
        "assumed",
    ]

    parameter_table_keywords = [
        "parameter",
        "symbol",
        "value",
        "unit",
        "table",
        "estimated",
        "fitted",
        "fixed",
        "calibrated",
        "initial value",
        "threshold",
        "maximum effect",
        "half maximal",
        "rate constant",
        "clearance",
        "volume",
        "absorption",
        "elimination",
    ]

    mechanism_keywords = [
        "mechanism",
        "interaction",
        "feedback",
        "inhibition",
        "stimulation",
        "activation",
        "suppression",
        "regulation",
        "nonlinear",
        "threshold",
        "saturation",
        "hill",
    ]

    keywords = list(general_keywords)

    if evidence_type in {"parameter", "table"}:
        keywords.extend(parameter_table_keywords)
    elif evidence_type == "mechanism":
        keywords.extend(mechanism_keywords)

    score = 0

    for keyword in keywords:
        if keyword in text_lower:
            score += 1

    return score


def sort_docs_by_evidence_score(docs, evidence_type: str = "general"):
    """
    Sort documents by scientific evidence score, then by original discovery order.
    """
    scored_docs = []

    for index, doc in enumerate(docs):
        score = score_scientific_evidence(
            text=getattr(doc, "page_content", ""),
            evidence_type=evidence_type,
        )
        scored_docs.append((score, index, doc))

    scored_docs.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    return [doc for score, index, doc in scored_docs]
