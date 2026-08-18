import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from crawler import crawl
from cost_tracker import CostTracker, count_tokens

load_dotenv()

CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
COLLECTION_NAME = "website_rag"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk-your_"):
            raise ValueError("EMBEDDING_PROVIDER is set to 'openai' but OPENAI_API_KEY is invalid or missing.")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # Default: HuggingFace local CPU embeddings (100% free, no OpenAI key required)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_index(start_url: str, max_pages: int = 20, tracker: CostTracker = None) -> Chroma:
    tracker = tracker or CostTracker()

    print(f"Crawling {start_url} ...")
    pages = crawl(start_url, max_pages=max_pages)
    print(f"Crawled {len(pages)} pages.")

    if not pages:
        raise RuntimeError(f"No pages could be crawled from {start_url}. Please check the URL or connection.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": page["url"], "title": page["title"], "chunk_index": i},
            ))

    print(f"Created {len(docs)} chunks.")

    for d in docs:
        tracker.log_embedding(count_tokens(d.page_content))

    embeddings = get_embeddings()

    # Reset old Chroma index to maintain clean state for target site
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Indexed {len(docs)} chunks into Chroma at {CHROMA_DIR}")
    return vectordb


def load_index() -> Chroma:
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"\n[ERROR] Vector index not found at '{CHROMA_DIR}'.\n"
            f"Please run website ingestion first:\n"
            f"  python main.py ingest https://fastapi.tiangolo.com/ 20\n"
        )
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://fastapi.tiangolo.com/"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    t = CostTracker()
    build_index(url, max_pages=n, tracker=t)
    t.print_report()

