import os
import sys

# Ensure rag_agent folder is on python path for clean imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cost_tracker import CostTracker


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "ingest":
        from ingest import build_index
        target_url = sys.argv[2] if len(sys.argv) > 2 else os.getenv("TARGET_WEBSITE", "https://fastapi.tiangolo.com/")
        max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        tracker = CostTracker()
        build_index(target_url, max_pages=max_pages, tracker=tracker)
        tracker.print_report()

    elif cmd == "ingest-docs":
        from docs_ingest import ingest_documents
        folder = sys.argv[2] if len(sys.argv) > 2 else "./docs"
        tracker = CostTracker()
        ingest_documents(folder, tracker=tracker)
        tracker.print_report()

    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Please provide a question. Example: python main.py ask \"What is FastAPI?\"")
            return
        from graph import make_graph, ask
        question = sys.argv[2]
        tracker = CostTracker()
        g, tracker = make_graph(tracker)
        result = ask(question, g)
        print("\n=================== ANSWER ===================")
        print(result["answer"])
        print("\n=================== SOURCES ===================")
        for s in result["sources"]:
            print(f"- {s}")
        tracker.print_report()

    elif cmd == "eval":
        from eval_runner import run_eval
        run_eval()

    elif cmd == "api" or cmd == "serve":
        import uvicorn
        from api import app
        print("\nStarting Swagger UI server at: http://127.0.0.1:8000/docs\n")
        uvicorn.run(app, host="127.0.0.1", port=8000)

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
