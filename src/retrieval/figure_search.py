from src.modeling.figure_extraction import retrieve_figure_context_service


def search_figures(
    vector_store,
    query: str,
    k: int = 6,
    figure_number: str | None = None,
    pdf_path: str | None = None,
) -> str:
    """Shared wrapper used by Q/A and model discovery."""
    return retrieve_figure_context_service(
        vector_store=vector_store,
        pdf_path=pdf_path,
        query=query,
        figure_number=figure_number,
    )
