"""
FastAPI REST Server with Swagger UI (/docs).
Exposes /ingest, /ask, and /eval endpoints for interactive web/Swagger usage.
"""
import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

app = FastAPI(
    title="Website-Grounded RAG Agent API",
    description="Interactive REST API for crawling websites, vector indexing with ChromaDB, and grounded Q&A with Groq.",
    version="1.0.0",
)


class IngestRequest(BaseModel):
    url: str = Field(default="https://fastapi.tiangolo.com/", description="Website URL to crawl")
    max_pages: int = Field(default=20, ge=1, le=100, description="Maximum pages to crawl")


class AskRequest(BaseModel):
    question: str = Field(..., example="What is FastAPI?")


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    cost_report: dict


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/ingest", summary="Crawl website and build vector index")
def ingest_website(req: IngestRequest):
    try:
        from cost_tracker import CostTracker
        from ingest import build_index
        from graph import reset_cache
        tracker = CostTracker()
        build_index(req.url, max_pages=req.max_pages, tracker=tracker)
        reset_cache()  # index was deleted & rebuilt on disk; drop the cached connection
        return {
            "status": "success",
            "message": f"Successfully crawled and indexed {req.url} ({req.max_pages} max pages).",
            "cost_report": tracker.summary(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse, summary="Ask question to grounded RAG agent")
def query_agent(req: AskRequest):
    try:
        from cost_tracker import CostTracker
        from graph import make_graph, ask
        tracker = CostTracker()
        g, tracker = make_graph(tracker)
        res = ask(req.question, g)
        return AskResponse(
            question=req.question,
            answer=res["answer"],
            sources=res["sources"],
            cost_report=tracker.summary(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval", summary="Run full 11-question evaluation benchmark")
def execute_eval():
    try:
        from eval_runner import run_eval
        results = run_eval()
        return {
            "status": "success",
            "total_questions": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
