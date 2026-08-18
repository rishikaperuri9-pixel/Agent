# Website-Grounded RAG Agent

An AI agent built with **LangChain**, **LangGraph**, **ChromaDB**, and **Groq** (`openai/gpt-oss-120b`) that crawls a public website, indexes its content into a vector database, and answers natural-language questions **grounded strictly in that website's content**.

Includes:
- Source URL citations for all generated answers.
- Explicit "not enough information" fallback when answers are absent.
- Detailed token and cost analysis (ingestion + query projections).
- 11-question evaluation suite covering 5 distinct test categories.
- **Zero OpenAI Dependency**: Uses local HuggingFace CPU embeddings (`all-MiniLM-L6-v2`) by default, requiring **only a free Groq API key**.

---

## 🛠️ Stack & Architecture

| Component | Technology | Why |
|---|---|---|
| **Orchestration** | **LangGraph** | Explicit, inspectable stateful graph pipeline (`retrieve -> generate`). |
| **LLM** | **Groq** (`openai/gpt-oss-120b`) | Ultra-fast, high-quality grounded generation. |
| **Embeddings** | **HuggingFace** (`all-MiniLM-L6-v2`) | Local CPU embeddings — zero API key required, 100% free. (Optional OpenAI support available). |
| **Vector DB** | **ChromaDB** | Local persistent vector storage with URL & page metadata. |
| **Crawler** | `requests` + `BeautifulSoup` | Clean BFS crawler with domain locking, polite delay, and HTML boilerplate stripping. |

### Architecture Diagram

```
                 ┌─────────────┐
  start_url ───► │   Crawler   │  (BFS, same-domain, strips nav/footer/script)
                 └──────┬──────┘
                        │ pages [{url, title, text}]
                        ▼
                 ┌─────────────┐
                 │  Chunker    │  RecursiveCharacterTextSplitter (800 chars / 100 overlap)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  Embedder   │  HuggingFace all-MiniLM-L6-v2 (Local CPU)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  ChromaDB   │  Persisted vector store (metadata: url, title)
                 └──────┬──────┘
                        │
       ┌────────────────┴──────────────────┐
       │            LangGraph                │
       │  ┌──────────┐      ┌─────────────┐  │
       │  │ retrieve │ ───► │  generate   │  │
       │  │ (top-4)  │      │ (Groq LLM)  │  │
       │  └──────────┘      └─────────────┘  │
       └──────────────────────┬──────────────┘
                              ▼
                Answer + cited source URL(s)
```
