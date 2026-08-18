# Website-Grounded RAG Agent

A retrieval-augmented Q&A agent that crawls a public website, indexes its content into a vector store, and answers natural-language questions **strictly grounded** in that site's content — with source URLs on every answer, explicit "not enough information" fallbacks, and full token/cost tracking.

**Target website used for this assessment:** (https://fastapi.tiangolo.com/) (FastAPI documentation — 20+ content-rich pages).

---

## 1. Architecture

```
                ┌─────────────┐
                │   crawler.py│  BFS crawl, same-domain, strip boilerplate
                └──────┬──────┘
                       │ pages [{url, title, text}]
                       ▼
                ┌─────────────┐
                │  ingest.py  │  chunk (RecursiveCharacterTextSplitter)
                │             │  embed (HuggingFace / OpenAI)
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │   ChromaDB  │  persisted vector store (website_rag collection)
                └──────┬──────┘
                       │ similarity_search (top-k)
                       ▼
                ┌─────────────┐
                │  graph.py   │  LangGraph: retrieve → generate
                │  (Groq LLM) │  answer grounded only in retrieved chunks
                └──────┬──────┘
                       ▼
          answer + cited source URL(s) + cost report
```

Entry points into this pipeline:
- **`main.py`** — CLI (`ingest`, `ingest-docs`, `ask`, `eval`, `api`)
- **`api.py`** — FastAPI REST wrapper with Swagger UI (`/ingest`, `/ask`, `/eval`)

A separate diagram image is included as `architecture_diagram.png` for a visual version of the above.

### Pipeline stages

| Stage | File | What it does |
|---|---|---|
| Crawl | `crawler.py` | BFS crawl of the target domain (depth ≤ 3, polite delay), strips `nav/footer/header/script/style/svg/form`, keeps `<main>`/`<article>`/body text, skips near-empty pages and non-HTML assets |
| Chunk & Embed | `ingest.py` | Splits page text with `RecursiveCharacterTextSplitter` (800 chars, 100 overlap), embeds with HuggingFace `all-MiniLM-L6-v2` (default, free/local) or OpenAI `text-embedding-3-small`, stores in a persisted Chroma collection (`website_rag`) |
| Optional local docs | `docs_ingest.py` | Lets you additionally ingest local PDF/DOCX/TXT files into the *same* Chroma collection, so retrieval can span both website and local documents |
| Retrieve + Generate | `graph.py` | A 2-node LangGraph (`retrieve` → `generate`). Retrieval does `similarity_search(k=4)`. Generation sends retrieved chunks (tagged with `[Source: url]`) to a Groq-hosted LLM with a strict system prompt forcing grounding, source citation, and refusal when unsupported |
| Cost tracking | `cost_tracker.py` | Counts tokens with `tiktoken` for both embedding calls (ingestion) and LLM calls (query), applies a per-model price table, and reports totals, per-query average, and projections at 100/1,000/10,000 queries |
| Evaluation | `eval_runner.py` + `eval_questions.json` | Runs a fixed 11-question benchmark (straightforward / paraphrased / multi-page / misleading / unanswerable) through the graph and writes `eval_results.json` |

---

## 2. Key Technical Decisions

- **LangGraph over a plain chain** — the retrieve/generate flow is modeled as an explicit 2-node graph so state (question → retrieved chunks → answer/sources) is typed (`TypedDict`) and each stage is independently testable/extensible (e.g., adding a re-ranking or query-rewriting node later is a one-node change).
- **Chroma as the vector store** — zero external infrastructure, persists to disk (`chroma_db/`), and is trivial to reset between ingestion runs (each `ingest` call wipes and rebuilds the collection to avoid stale/duplicate content from a previous site).
- **HuggingFace local embeddings by default** — `all-MiniLM-L6-v2` runs on CPU with no API key and no cost, which keeps the assessment reproducible for anyone cloning the repo. OpenAI embeddings are available as a drop-in swap via `EMBEDDING_PROVIDER=openai`.
- **Groq as the LLM provider** — fast, low-cost inference (`openai/gpt-oss-120b` by default) suited to an assessment/demo context; swappable via `GROQ_MODEL`.
- **Strict grounding via system prompt + post-processing** — the system prompt forbids outside knowledge and requires a verbatim `Sources:` list. `graph.py` then cross-checks the model's cited sources against what was *actually retrieved*, so a vague or paraphrased citation from the LLM can never silently replace a real URL — it falls back to the full set of retrieved sources instead.
- **Partial-answer handling** — rather than an all-or-nothing refusal, the prompt instructs the model to answer whatever part of the question the context supports and explicitly flag the part(s) it can't, only using the full refusal message when *none* of the question is answerable.
- **Token/cost tracking is provider-aware** — `CostTracker` reads actual `token_usage` metadata from the LLM response when available and only falls back to a `tiktoken` estimate if the provider doesn't return it, so reported costs reflect real API billing wherever possible.
- **Single shared vector collection** — website and (optionally) local documents are ingested into the same Chroma collection with a `source_type` metadata flag, so the retrieval/generation code doesn't need to branch by content origin.

---

## 3. Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### Install
```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure environment
Copy `.env.example` to `.env` and fill in your key:
```bash
cp .env.example .env
```
```env
# .env.example
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b

# Embeddings: "huggingface" (default, free/local) or "openai"
EMBEDDING_PROVIDER=huggingface
OPENAI_API_KEY=sk-your_key_here   # only needed if EMBEDDING_PROVIDER=openai

TARGET_WEBSITE=https://fastapi.tiangolo.com/
```

---

## 4. Running the Solution

### CLI (`main.py`)

```bash
# 1. Crawl + index the target website (max 20 pages)
python main.py ingest https://fastapi.tiangolo.com/ 20

# 2. (Optional) ingest local documents into the same index
python main.py ingest-docs ./docs

# 3. Ask a question
python main.py ask "What is FastAPI?"

# 4. Run the full evaluation suite
python main.py eval
```

Each command prints a cost/token report at the end (see `cost_tracker.print_report()`).

### REST API / Swagger UI (`api.py`)

```bash
python main.py api
# or: python api.py
```
Then open **http://127.0.0.1:8000/docs** for interactive Swagger UI, or call directly:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://fastapi.tiangolo.com/", "max_pages": 20}'

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'

curl http://127.0.0.1:8000/eval
```

### Evaluation

`eval_questions.json` contains 11 questions spanning five categories: straightforward, paraphrased, multi-page, misleading, and unanswerable. Running `python main.py eval` (or `GET /eval`) executes all of them through the live graph and writes results to `eval_results.json`, alongside a printed cost report.

**Result summary** (see `eval_results.json` for full transcripts):
- **Straightforward (3/3 correct)** — direct factual answers with accurate source citations (e.g., FastAPI's key features, Swagger/ReDoc docs).
- **Paraphrased (2/2 correct)** — differently-worded versions of the same underlying facts were still answered correctly and grounded in the same source pages, showing retrieval isn't brittle to phrasing.
- **Multi-page (1/1 correct)** — the async/await + dependency-injection question correctly pulled and synthesized content from two different pages (`/async` and `/tutorial/dependencies`).
- **Misleading (2/2 correctly refused/corrected)** — false-premise questions (Django ORM requirement, Apache Foundation ownership) were correctly identified as false rather than answered as if true.
- **Unanswerable (3/3 correctly flagged)** — out-of-scope questions (Docker/Uvicorn deployment specifics, a fictional "FastAPI Inc" stock price, "FastAPI Airlines" flights) were correctly identified as not supported by the indexed content, rather than hallucinated.

Overall: **11/11** behaved as expected — correct grounded answers where the site supports it, correct refusals/corrections otherwise, with source URLs attached to every answer.

---

## 5. Cost Tracking

`cost_tracker.py` counts tokens via `tiktoken` for:
- **Ingestion**: total tokens across all embedded chunks
- **Queries**: prompt + completion tokens per LLM call (from provider metadata when available)

It applies a per-model price table (`PRICES`, $ per 1M tokens) and reports:
- Total ingestion cost (one-time)
- Total query cost so far
- Average cost per query
- Projected cost at 100 / 1,000 / 10,000 queries

See the separate **Cost Analysis** document for a worked example with real numbers from this project's ingestion + eval run.

---

## 6. Project Structure

```
.
├── main.py            # CLI entry point (ingest / ingest-docs / ask / eval / api)
├── api.py             # FastAPI REST wrapper + Swagger UI
├── crawler.py         # Website crawler
├── ingest.py          # Chunking, embedding, Chroma indexing (website)
├── docs_ingest.py      # Chunking, embedding, Chroma indexing (local files)
├── graph.py           # LangGraph retrieve → generate pipeline
├── cost_tracker.py    # Token counting + cost estimation
├── eval_questions.json
├── eval_runner.py
├── eval_results.json  # Generated by eval_runner.py
├── .env.example
└── README.md
```

---

## 7. Known Limitations

- **Crawl scope**: BFS crawl with depth ≤ 3 and a configurable page cap; very large or JS-rendered (client-side-only) sites may not be fully captured since the crawler uses `requests` + BeautifulSoup rather than a headless browser.
- **No re-ranking**: retrieval is plain vector similarity (top-4); there's no cross-encoder re-ranking step, so borderline-relevant chunks can occasionally be included or a genuinely relevant chunk just outside top-k can be missed.
- **No conversational memory**: each question is handled independently; there's no multi-turn follow-up/context carry-over (e.g., "what about its performance?" after a prior question won't resolve "its").
- **Chunking is generic**: fixed-size recursive character splitting doesn't exploit document structure (headings, code blocks), which can occasionally split a code sample or table awkwardly.
- **Full reindex on ingest**: each `ingest` run wipes and rebuilds the Chroma collection rather than incrementally updating it — fine for a single-site assessment scope, but not ideal for frequently-changing production sites.
- **Cost estimates are approximate**: local HuggingFace embeddings are free but their "cost" entry is just for consistency in the report; LLM costs depend on Groq's published pricing at the time and may drift if pricing changes.
- **No authentication/rate-limiting** on the FastAPI endpoints — fine for a local assessment demo, not production-hardened.
