"""
Runs eval_questions.json through the RAG graph and prints a results summary.
Outputs detailed test results to eval_results.json and logs cost projections.
"""
import os
import json

from graph import make_graph, ask
from cost_tracker import CostTracker


def run_eval(questions_filename: str = "eval_questions.json"):
    base_dir = os.path.dirname(__file__)
    questions_path = os.path.join(base_dir, questions_filename)
    results_path = os.path.join(base_dir, "eval_results.json")

    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found at {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    tracker = CostTracker()
    compiled_graph, tracker = make_graph(tracker)

    results = []
    print(f"\n=======================================================")
    print(f"       RUNNING BENCHMARK EVALUATION ({len(questions)} QUESTIONS)")
    print(f"=======================================================\n")

    for q in questions:
        r = ask(q["question"], compiled_graph)
        item = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": r["answer"],
            "sources": r["sources"],
        }
        results.append(item)
        print(f"[{q['id']:02d}] [{q['category'].upper()}]")
        print(f"  Q: {q['question']}")
        print(f"  A: {r['answer'].strip()}")
        print(f"  Sources: {r['sources']}")
        print("-" * 60)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation results to: {results_path}")
    tracker.print_report()
    return results


if __name__ == "__main__":
    run_eval()
