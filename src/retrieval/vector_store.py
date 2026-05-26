import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==========================================================
# Embedding configuration
# ==========================================================

CHROMA_DIR = "chroma_db"

# False = free local embeddings
# True = OpenAI embeddings (better retrieval, costs money)
USE_OPENAI_EMBEDDINGS = False


# ==========================================================
# Helper: Create unique Chroma path per PDF
# ==========================================================

def get_chroma_dir(
    pdf_path: str,
    parser_mode: str = "default",
) -> str:
    import os
    import re

    pdf_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    safe_pdf_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        pdf_name,
    )

    safe_parser_mode = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        parser_mode or "default",
    )

    return os.path.join(
        "chroma_db",
        safe_pdf_name,
        safe_parser_mode,
    )

# ==========================================================
# Embedding model selection
# ==========================================================

def get_embeddings():
    """
    Select embedding model.

    Local HuggingFace:
        - free
        - runs locally
        - good for development

    OpenAI:
        - better retrieval quality
        - costs money
    """

    if USE_OPENAI_EMBEDDINGS:
        print(
            "Using OpenAI embeddings: "
            "text-embedding-3-small"
        )

        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    print(
        "Using local HuggingFace embeddings: "
        "BAAI/bge-small-en-v1.5"
    )

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ==========================================================
# Build vector database
# ==========================================================

def build_vector_store(
    documents,
    pdf_path: str,
    reset_db: bool = True,
    parser_mode: str = "docling",
    chroma_dir: str | None = None,
):
    if chroma_dir is None:
        chroma_dir = get_chroma_dir(
            pdf_path,
            parser_mode=parser_mode,
        )

    if reset_db and os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )

    return vector_store
# ==========================================================
# Load existing vector database
# ==========================================================

def load_vector_store(
    pdf_path: str,
    parser_mode: str = "default",
    chroma_dir: str | None = None,
):
    """
    Load existing Chroma database.

    If chroma_dir is provided, use it directly.
    Otherwise, fall back to the old PDF/parser-based path.
    """

    if chroma_dir is None:
        chroma_dir = get_chroma_dir(
            pdf_path,
            parser_mode=parser_mode,
        )

    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(
            f"No Chroma DB found at: {chroma_dir}"
        )

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )

    print(f"Loaded Chroma DB from: {chroma_dir}")

    return vector_store