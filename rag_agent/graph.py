"""
LangGraph pipeline: retrieve -> generate.
Answers are grounded ONLY in retrieved website chunks; model must cite source URLs
and explicitly say when the site doesn't have the answer.
"""
import os
import re
import tiktoken
from typing import TypedDict, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from ingest import load_index
from cost_tracker import CostTracker

load_dotenv()

TOP_K = 4  # keep retrieval tight to minimize tokens/cost, but wide enough for multi-part questions

_encoder = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = """Answer ONLY from the CONTEXT below (retrieved from the target website). No outside knowledge.

Rules:
1. Answer every part of the question the context supports. For any part it doesn't, say so specifically (e.g. "the site doesn't state X") instead of refusing the whole answer.
2. Only reply "I don't have enough information in the indexed sources to answer that." if NONE of the question is answerable from context.
3. No inline citations/brackets/URLs mid-answer. At the end, under "Sources:", list the exact identifier from each [Source: ...] tag you used — copy it verbatim, never paraphrase or describe it (e.g. not "the documentation page", the literal URL/filename itself).
4. Be clear, accurate, concise.
"""


def _parse_answer_and_sources(raw_text: str, fallback_sources: list):
    """
    Splits the model's raw response into clean answer text + the sources it
    actually cited (parsed from its own "Sources:" section), instead of
    blindly reporting every chunk that was retrieved. Also strips any inline
    citation artifacts that slip past the prompt instructions.
    """
    match = re.search(r'\*{0,2}Sources:\*{0,2}', raw_text, re.IGNORECASE)
    if match:
        answer_text = raw_text[:match.start()].strip()
        sources_block = raw_text[match.end():]
        cited = [ln.strip(" -*•\t") for ln in sources_block.splitlines()]
        cited = sorted(set(c for c in cited if c))
    else:
        answer_text = raw_text.strip()
        cited = []

    # Strip any leftover inline citation brackets (【url】, [url], {url}) wherever they appear
    answer_text = re.sub(r'[【\[\{]\s*https?://\S+?\s*[】\]\}]', '', answer_text).strip()
    answer_text = re.sub(r'\s{2,}', ' ', answer_text)

    return answer_text, cited or fallback_sources


class RAGState(TypedDict):
    question: str
    retrieved: List[dict]
    answer: str
    sources: List[str]

_vectordb = None
_llm = None
_llm_model_name = None


def _get_resources():
    global _vectordb, _llm, _llm_model_name
    if _vectordb is None:
        _vectordb = load_index()
    if _llm is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or groq_api_key.startswith("gsk_your_"):
            raise ValueError(
                "\n[ERROR] GROQ_API_KEY is missing or invalid in your .env file.\n"
                "Please get a free Groq API key from https://console.groq.com/keys and set:\n"
                "  GROQ_API_KEY=gsk_...\n"
                "in your .env file."
            )
        _llm_model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        _llm = ChatGroq(model=_llm_model_name, temperature=0)
    return _vectordb, _llm, _llm_model_name


def reset_cache():
    """Clears the cached vector store/LLM so the next make_graph() call
    re-initializes them. Call this after re-ingesting (the index was deleted
    and rebuilt on disk), so /ask doesn't keep querying a stale connection."""
    global _vectordb, _llm, _llm_model_name
    _vectordb, _llm, _llm_model_name = None, None, None


def make_graph(tracker: CostTracker = None):
    tracker = tracker or CostTracker()
    vectordb, llm, model_name = _get_resources()
    tracker.llm_model = model_name  # keep cost report in sync with the model actually used

    def retrieve_node(state: RAGState) -> RAGState:
        results = vectordb.similarity_search(state["question"], k=TOP_K)
        retrieved = [{
            "text": r.page_content,
            "source": r.metadata.get("source", "Unknown"),
            "title": r.metadata.get("title", "")
        } for r in results]
        return {**state, "retrieved": retrieved}

    def generate_node(state: RAGState) -> RAGState:
        if not state["retrieved"]:
            return {
                **state,
                "answer": "I don't have enough information in the indexed sources to answer that.",
                "sources": []
            }

        context = "\n\n---\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}" for r in state["retrieved"]
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {state['question']}"),
        ]

        try:
            response = llm.invoke(messages)
        except Exception as err:
            err_msg = str(err).lower()
            if "model_not_found" in err_msg or "404" in err_msg or "invalid_api_key" in err_msg or "401" in err_msg:
                raise ValueError(
                    "\n[ERROR] Groq API Request Failed (404/401).\n"
                    "Your GROQ_API_KEY in .env appears to be invalid or unauthorized.\n"
                    "Please generate a free API key at https://console.groq.com/keys and set:\n"
                    "  GROQ_API_KEY=gsk_your_real_key_here\n"
                    "in your .env file."
                )
            raise err

        # Log token usage from response metadata, falling back to a tiktoken estimate
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        if not isinstance(in_tok, int):
            in_tok = len(_encoder.encode(SYSTEM_PROMPT + context + state["question"]))
        if not isinstance(out_tok, int):
            out_tok = len(_encoder.encode(response.content))
        tracker.log_llm_call(in_tok, out_tok, label=state["question"][:40])

        known_sources = set(r["source"] for r in state["retrieved"] if r.get("source"))
        all_retrieved = sorted(known_sources)
        clean_answer, cited_sources = _parse_answer_and_sources(response.content, fallback_sources=all_retrieved)

        # Trust the model's cited list only where it names a source that was
        # actually retrieved. A vague/paraphrased claim (e.g. "the docs page"
        # instead of the real URL) never overrides the real retrieved sources.
        valid_cited = [c for c in cited_sources if c in known_sources]
        final_sources = valid_cited if valid_cited else all_retrieved

        return {**state, "answer": clean_answer, "sources": final_sources}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(), tracker


def ask(question: str, compiled_graph):
    """Run one question through an already-compiled graph (its tracker is already
    wired in via closure from make_graph — no separate tracker needed here)."""
    return compiled_graph.invoke({"question": question, "retrieved": [], "answer": "", "sources": []})


if __name__ == "__main__":
    import sys
    t = CostTracker()
    try:
        g, t = make_graph(t)
        q = sys.argv[1] if len(sys.argv) > 1 else "What is FastAPI?"
        result = ask(q, g)
        print("\nANSWER:\n", result["answer"])
        print("\nSOURCES:", result["sources"])
        t.print_report()
    except Exception as e:
        print(e)
