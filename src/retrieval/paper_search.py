from src.retrieval.context import retrieve_semantic_context


def search_paper(vector_store, query: str, k: int = 6) -> str:
    """Shared wrapper used by Q/A and model discovery."""
    return retrieve_semantic_context(
        vector_store=vector_store,
        query=query,
        context_type="text",
        k=k,
    )


def search_paper_docs(vector_store, query: str, k: int = 6):
    """
    Return raw retrieved documents from the vector store.
    Used by model discovery when metadata/page info is needed.
    """
    return vector_store.similarity_search(query, k=k)
