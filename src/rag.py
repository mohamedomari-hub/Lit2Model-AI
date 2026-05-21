import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


CHROMA_DIR = "chroma_db"

# False = free local embeddings
# True = OpenAI embeddings, better quality but costs money
USE_OPENAI_EMBEDDINGS = False


def get_embeddings():
    """
    Select the embedding model.

    Local HuggingFace embeddings:
        - free
        - runs locally
        - good for development

    OpenAI embeddings:
        - better retrieval quality
        - costs money
    """

    if USE_OPENAI_EMBEDDINGS:
        print("Using OpenAI embeddings: text-embedding-3-small")

        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    print("Using local HuggingFace embeddings: BAAI/bge-small-en-v1.5")

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vector_store(documents, reset_db: bool = True):
    """
    Build a Chroma vector store from parsed PDF documents.

    documents:
        List of LangChain Document objects coming from parser.py

    reset_db:
        If True, delete old vector database and rebuild from the current PDF.
    """

    if reset_db and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks for RAG.")

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store


def load_vector_store():
    """
    Load an existing Chroma vector store from disk.

    Use this when you already embedded the PDF before and do not want
    to create embeddings again.
    """

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    return vector_store