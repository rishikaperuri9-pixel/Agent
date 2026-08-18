"""
Ingests local documents (PDF, DOCX, TXT) into the SAME Chroma collection
used by the website crawler, so retrieval covers both sources.
"""
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_chroma import Chroma

from ingest import CHROMA_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, get_embeddings
from cost_tracker import CostTracker, count_tokens

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def load_documents_from_folder(folder_path: str):
    all_docs = []
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return all_docs

    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        loader_cls = LOADERS.get(ext)
        if not loader_cls:
            print(f"  [skip] {fname} (unsupported type)")
            continue
        path = os.path.join(folder_path, fname)
        loaded = loader_cls(path).load()
        for d in loaded:
            d.metadata["source"] = fname          # shown in citations
            d.metadata["source_type"] = "document"
        all_docs.extend(loaded)
        print(f"  [loaded] {fname} ({len(loaded)} page(s))")
    return all_docs


def ingest_documents(folder_path: str, tracker: CostTracker = None) -> Chroma:
    tracker = tracker or CostTracker()

    print(f"Loading documents from {folder_path} ...")
    raw_docs = load_documents_from_folder(folder_path)
    if not raw_docs:
        print("No documents were loaded.")
        return None

    print(f"Loaded {len(raw_docs)} document page(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} chunks.")

    for c in chunks:
        tracker.log_embedding(count_tokens(c.page_content))

    embeddings = get_embeddings()

    # Adds to the SAME persisted collection the website crawler writes to
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectordb.add_documents(chunks)
    print(f"Added {len(chunks)} document chunks to existing Chroma collection.")
    return vectordb


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "./docs"
    t = CostTracker()
    ingest_documents(folder, tracker=t)
    t.print_report()

