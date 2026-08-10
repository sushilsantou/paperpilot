"""
Retrieval-only evaluation — the half of run_eval.py that needs no LLM.

run_eval.py drives the full agent graph (LLM query rewriting -> synthesis ->
critic), so it needs GOOGLE_API_KEY and costs API calls. But the retrieval
question — does blending keyword overlap into vector search retrieve better
passages, and at what weight — is answerable without any of that:
`hybrid_search` is pure embedding + arithmetic, and `keyword_recall` scores
the retrieved chunks alone.

So this harness sweeps the same four strategies as run_eval.py and reports
keyword recall and retrieval latency per strategy, with zero API cost.

The deliberate difference from run_eval.py: there is no LLM query rewriting,
so the raw question text is the search query. That makes this a measurement of
the *retrieval blend* in isolation rather than of the full pipeline, and its
numbers are not interchangeable with run_eval.py's.

Usage:
    QDRANT_PATH=./qdrant_local python -m src.eval.run_retrieval_eval
"""
import json
import time
from pathlib import Path
from statistics import mean
from typing import List

from src.config import settings
from src.agents.retriever_agent import hybrid_search
from src.eval.metrics import keyword_recall
from src.vectorstore import get_qdrant_client

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"

STRATEGIES = {
    "vector_only": 0.0,
    "hybrid_light": 0.15,
    "hybrid": 0.3,
    "hybrid_heavy": 0.5,
}


def run(mlflow_tracking: bool = True) -> dict:
    eval_set: List[dict] = json.loads(EVAL_SET_PATH.read_text())
    client = get_qdrant_client()

    # Warm up before timing anything: the first hybrid_search call pays for
    # lazily loading the sentence-transformers model, which otherwise lands
    # entirely on whichever strategy runs first and makes its latency column
    # meaningless (it read ~6x the others before this was added).
    hybrid_search(client, search_query=eval_set[0]["question"], top_k=settings.top_k, keyword_weight=0.0)

    summary = {}
    for name, keyword_weight in STRATEGIES.items():
        recalls, latencies = [], []
        for item in eval_set:
            start = time.perf_counter()
            retrieved = hybrid_search(
                client,
                search_query=item["question"],
                top_k=settings.top_k,
                keyword_weight=keyword_weight,
            )
            latencies.append((time.perf_counter() - start) * 1000.0)
            recalls.append(keyword_recall(retrieved, item["expected_keywords"]))

        summary[name] = {
            "keyword_weight": keyword_weight,
            "avg_keyword_recall": round(mean(recalls), 4),
            "perfect_recall_questions": sum(1 for r in recalls if r == 1.0),
            "avg_retrieval_latency_ms": round(mean(latencies), 2),
        }
        print(f"{name:<14} weight={keyword_weight:<5} "
              f"recall={summary[name]['avg_keyword_recall']:.4f} "
              f"perfect={summary[name]['perfect_recall_questions']}/{len(eval_set)} "
              f"latency={summary[name]['avg_retrieval_latency_ms']:.1f}ms")

    if mlflow_tracking:
        try:
            import mlflow

            mlflow.set_experiment("paperpilot-retrieval-only")
            for name, metrics in summary.items():
                with mlflow.start_run(run_name=name):
                    mlflow.log_params({"strategy": name, "keyword_weight": metrics["keyword_weight"],
                                       "top_k": settings.top_k, "llm_query_rewriting": False})
                    mlflow.log_metrics({k: v for k, v in metrics.items() if k != "keyword_weight"})
        except Exception as e:  # noqa: BLE001 — tracking is a convenience, not the result
            print(f"(MLflow logging skipped: {e})")

    return summary


if __name__ == "__main__":
    print(f"Retrieval-only eval — {len(STRATEGIES)} strategies, no LLM calls\n")
    result = run()
    best = max(result.items(), key=lambda kv: kv[1]["avg_keyword_recall"])
    print(f"\nBest by keyword recall: {best[0]} ({best[1]['avg_keyword_recall']:.4f})")
