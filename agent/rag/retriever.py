from pathlib import Path
from typing import Optional
from agent.rag.loader import load_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

VECTOR_DB_PATH = Path(__file__).parent.parent / "data" / "vectorstore"


def build_retriever(embedding_model: Optional[OllamaEmbeddings] = None):
    print("Building RAG vector database...")

    documents = load_documents()
    if not documents:
        print("No available documents (PDF/TXT) found in agent/data/docs/.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    splits = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=str(VECTOR_DB_PATH),
        collection_name="ctk_rag_collection",
        collection_metadata={"source": "pdf_txt_docs"},
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "score_threshold": 0.7},
    )
    print("Vector database successfully built.")
    return retriever


def load_existing_retriever(embedding_model: Optional[OllamaEmbeddings] = None):
    if not (VECTOR_DB_PATH / "chroma.sqlite3").exists():
        return None

    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_DB_PATH),
        collection_name="ctk_rag_collection",
        collection_metadata={"source": "pdf_txt_docs"},
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "score_threshold": 0.7},
    )


def get_retriever(embedding_model: Optional[OllamaEmbeddings] = None):
    return load_existing_retriever(embedding_model) or build_retriever(embedding_model)
