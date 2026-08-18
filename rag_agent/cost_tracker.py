"""
Tracks token usage across ingestion (embeddings) and query (LLM) calls,
and estimates $ cost. Prices are per 1M tokens — update PRICES if models change.
"""
import os
from dataclasses import dataclass, field
import tiktoken
from dotenv import load_dotenv

load_dotenv()

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Shared token-counting helper used by ingestion and query paths."""
    return len(_encoder.encode(text))


# $ per 1M tokens.
PRICES = {
    "all-MiniLM-L6-v2": {"input": 0.00, "output": 0.00},          # Local HuggingFace CPU (Free)
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},    # OpenAI embeddings
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},       # Groq LLM pricing 
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def _get_default_embedding_model() -> str:
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    return "text-embedding-3-small" if provider == "openai" else "all-MiniLM-L6-v2"


@dataclass
class CostTracker:
    embedding_tokens: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embedding_model: str = field(default_factory=_get_default_embedding_model)
    llm_model: str = "openai/gpt-oss-120b"
    calls: list = field(default_factory=list)

    def log_embedding(self, num_tokens: int):
        self.embedding_tokens += num_tokens

    def log_llm_call(self, input_tokens: int, output_tokens: int, label: str = ""):
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        self.calls.append({"label": label, "in": input_tokens, "out": output_tokens})

    def _cost(self, tokens: int, rate_per_million: float) -> float:
        return (tokens / 1_000_000) * rate_per_million

    def summary(self) -> dict:
        emb_price = PRICES.get(self.embedding_model, {"input": 0.0, "output": 0.0})
        llm_price = PRICES.get(self.llm_model, {"input": 0.15, "output": 0.60})

        embedding_cost = self._cost(self.embedding_tokens, emb_price["input"])
        llm_input_cost = self._cost(self.llm_input_tokens, llm_price["input"])
        llm_output_cost = self._cost(self.llm_output_tokens, llm_price["output"])
        total = embedding_cost + llm_input_cost + llm_output_cost

        return {
            "embedding_tokens": self.embedding_tokens,
            "embedding_cost_usd": round(embedding_cost, 6),
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_cost_usd": round(llm_input_cost + llm_output_cost, 6),
            "total_cost_usd": round(total, 6),
        }

    def per_query_cost(self) -> float:
        """Average $ cost per query, based on calls logged so far (excludes one-time ingestion)."""
        if not self.calls:
            return 0.0
        llm_price = PRICES.get(self.llm_model, {"input": 0.15, "output": 0.60})
        total = sum(
            self._cost(c["in"], llm_price["input"]) + self._cost(c["out"], llm_price["output"])
            for c in self.calls
        )
        return total / len(self.calls)

    def projected_costs(self) -> dict:
        """Projects query-time cost at 100 / 1,000 / 10,000 queries (ingestion cost is separate & one-time)."""
        per_query = self.per_query_cost()
        return {n: round(per_query * n, 4) for n in (100, 1_000, 10_000)}

    def print_report(self):
        s = self.summary()
        print("\n--- Cost & Token Usage Report ---")
        print(f"Embedding Provider/Model: {self.embedding_model}")
        print(f"LLM Provider/Model: {self.llm_model}")
        print(f"Ingestion (embeddings): {s['embedding_tokens']} tokens -> ${s['embedding_cost_usd']}")
        print(f"Queries logged: {len(self.calls)} call(s), "
              f"{s['llm_input_tokens']} in / {s['llm_output_tokens']} out tokens -> ${s['llm_cost_usd']}")
        print(f"Total spent so far: ${s['total_cost_usd']}")
        proj = self.projected_costs()
        print(f"Avg cost/query: ${round(self.per_query_cost(), 6)}")
        print("Projected query-time costs:")
        for n, cost in proj.items():
            print(f"  {n:>6} queries -> ${cost}")
        print(f"(+ one-time ingestion cost: ${s['embedding_cost_usd']})")

