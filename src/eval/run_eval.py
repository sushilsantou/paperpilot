"""
Runs the eval set against N retrieval strategies (varying the hybrid
keyword-weight) end-to-end through the full agent graph, and logs
per-strategy aggregate metrics to MLflow so strategies can be compared
run-over-run in the MLflow UI.

Usage:
    mlflow ui   # in another terminal, to view results at localhost:5000
    python -m src.eval.run_eval

Requires: a running Qdrant instance with data already ingested
(see src/ingestion/ingest.py) and GOOGLE_API_KEY set.
"""
import json
import time
from pathlib import Path
from statistics import mean
from typing import List

import mlflow
from qdrant_client import QdrantClient

from src.agents.graph import answer_question
from src.config import settings
from src.eval.metrics import citation_validity_rate, has_citations, keyword_recall
from src.vectorstore import get_qdrant_client

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"

STRATEGIES = {
    "vector_only": {"keyword_weight": 0.0},
    "hybrid_light": {"keyword_weight": 0.15},
    "hybrid": {"keyword_weight": 0.3},
    "hybrid_heavy": {"keyword_weight": 0.5},
}


def load_eval_set() -> List[dict]:
    return json.loads(EVAL_SET_PATH.read_text())


def run_strategy(strategy_name: str, params: dict, eval_set: List[dict], client: QdrantClient) -> dict:
    recalls, citation_rates, latencies, retries, cited_flags = [], [], [], [], []

    for item in eval_set:
        start = time.time()
        result = answer_question(item["question"], client=client, keyword_weight=params["keyword_weight"])
        elapsed = time.time() - start

        retrieved = result.get("retrieved", [])
        cited_ids = result.get("source_ids_cited", [])
        final_answer = result.get("final_answer", "")

        recalls.append(keyword_recall(retrieved, item["expected_keywords"]))
        citation_rates.append(citation_validity_rate(cited_ids, retrieved))
        latencies.append(elapsed)
        retries.append(result.get("retry_count", 0))
        cited_flags.append(1.0 if has_citations(final_answer) else 0.0)

        print(f"  [{strategy_name}] {item['id']}: recall={recalls[-1]:.2f} "
              f"citation_validity={citation_rates[-1]:.2f} retries={retries[-1]} latency={elapsed:.1f}s")

    return {
        "avg_keyword_recall": mean(recalls),
        "avg_citation_validity": mean(citation_rates),
        "avg_latency_sec": mean(latencies),
        "avg_retries": mean(retries),
        "citation_presence_rate": mean(cited_flags),
    }


def main():
    eval_set = load_eval_set()
    client = get_qdrant_client()

    mlflow.set_experiment("paperpilot-retrieval-strategies")

    summary = {}
    for name, params in STRATEGIES.items():
        print(f"\n=== Strategy: {name} (keyword_weight={params['keyword_weight']}) ===")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"strategy": name, **params, "top_k": settings.top_k, "chunk_size": settings.chunk_size})
            metrics = run_strategy(name, params, eval_set, client)
            mlflow.log_metrics(metrics)
            summary[name] = metrics

    print("\n=== Summary ===")
    for name, metrics in summary.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
