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
import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import List

import mlflow
from qdrant_client import QdrantClient

from src.agents.graph import answer_question
from src.agents.llm import QuotaExhausted
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


class PartialRun(Exception):
    """Carries the metrics scored before a quota wall cut the run short."""

    def __init__(self, summary: dict):
        super().__init__("run stopped early")
        self.summary = summary


def _summarise(recalls, citation_rates, latencies, retries, cited_flags, completed: int) -> dict:
    if not recalls:
        return {"questions_completed": 0}
    return {
        "questions_completed": completed,
        "avg_keyword_recall": mean(recalls),
        "avg_citation_validity": mean(citation_rates),
        "avg_latency_sec": mean(latencies),
        "avg_retries": mean(retries),
        "citation_presence_rate": mean(cited_flags),
    }


def run_strategy(strategy_name: str, params: dict, eval_set: List[dict], client: QdrantClient) -> dict:
    recalls, citation_rates, latencies, retries, cited_flags = [], [], [], [], []
    completed = 0

    for item in eval_set:
        start = time.time()
        try:
            result = answer_question(item["question"], client=client, keyword_weight=params["keyword_weight"])
        except QuotaExhausted as e:
            # A daily cap mid-run must not throw away the questions already
            # scored. Report what completed and let main() stop cleanly.
            print(f"  [{strategy_name}] stopped at {item['id']}: {e}")
            raise PartialRun(_summarise(recalls, citation_rates, latencies, retries, cited_flags, completed)) from e
        elapsed = time.time() - start

        retrieved = result.get("retrieved", [])
        cited_ids = result.get("source_ids_cited", [])
        final_answer = result.get("final_answer", "")

        recalls.append(keyword_recall(retrieved, item["expected_keywords"]))
        citation_rates.append(citation_validity_rate(cited_ids, retrieved))
        latencies.append(elapsed)
        retries.append(result.get("retry_count", 0))
        cited_flags.append(1.0 if has_citations(final_answer) else 0.0)

        completed += 1
        print(f"  [{strategy_name}] {item['id']}: recall={recalls[-1]:.2f} "
              f"citation_validity={citation_rates[-1]:.2f} retries={retries[-1]} latency={elapsed:.1f}s")

    return _summarise(recalls, citation_rates, latencies, retries, cited_flags, completed)


def main():
    parser = argparse.ArgumentParser(description="Full agent eval across retrieval strategies")
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N questions (the free Gemini tier caps requests per day)")
    parser.add_argument("--strategies", default=None,
                        help="comma-separated subset, e.g. vector_only,hybrid")
    parser.add_argument("--out", default="eval_results.json", help="where to write results")
    args = parser.parse_args()

    eval_set = load_eval_set()
    if args.limit:
        eval_set = eval_set[: args.limit]

    chosen = STRATEGIES
    if args.strategies:
        wanted = [s.strip() for s in args.strategies.split(",")]
        unknown = [s for s in wanted if s not in STRATEGIES]
        if unknown:
            parser.error(f"unknown strategies: {unknown}. Available: {list(STRATEGIES)}")
        chosen = {k: STRATEGIES[k] for k in wanted}

    client = get_qdrant_client()
    mlflow.set_experiment("paperpilot-retrieval-strategies")

    summary, stopped_early = {}, False
    for name, params in chosen.items():
        print(f"\n=== Strategy: {name} (keyword_weight={params['keyword_weight']}) ===")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"strategy": name, **params, "top_k": settings.top_k,
                               "chunk_size": settings.chunk_size, "model": settings.gemini_model})
            try:
                metrics = run_strategy(name, params, eval_set, client)
            except PartialRun as partial:
                metrics = partial.summary
                stopped_early = True
            mlflow.log_metrics(metrics)
            summary[name] = metrics
        if stopped_early:
            break

    print("\n=== Summary ===")
    for name, metrics in summary.items():
        print(f"{name}: {metrics}")
    if stopped_early:
        print("\nNOTE: stopped early on API quota. Numbers above cover only the "
              "questions listed as questions_completed — do not read them as a full run.")

    Path(args.out).write_text(json.dumps(
        {"model": settings.gemini_model, "questions_per_strategy": len(eval_set),
         "stopped_early": stopped_early, "results": summary}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
