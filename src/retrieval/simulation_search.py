from src.retrieval.context import retrieve_semantic_context


def search_simulations(vector_store, query: str, k: int = 6) -> str:
    """Shared wrapper used by Q/A and model discovery."""
    return retrieve_semantic_context(
        vector_store=vector_store,
        query=query,
        context_type="simulation",
        k=k,
    )
