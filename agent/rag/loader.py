from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader

DOCS_PATH = Path(__file__).parent.parent / "data" / "docs"
SUPPORTED_EXTENSIONS = [".pdf", ".txt"]

def load_documents() -> List:
    all_docs = []
    for file in DOCS_PATH.glob("*"):
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        else:
            continue
        all_docs.extend(loader.load())
    return all_docs