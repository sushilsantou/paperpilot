# PaperPilot — Measured Results

First run against a real corpus, 2026-08-10. Every number below came from a
command in this repo; nothing is estimated.

**Scope, stated up front:** this covers the *retrieval* half of the system.
The full agent pipeline (LLM query rewriting → synthesis → critic) needs a
Gemini API key, which was not available for this run, so citation validity,
critic-retry counts, and end-to-end answer latency are **not measured yet**.
What is measured is the retrieval blend, which is the part that needs no LLM.

## Corpus

| | |
|---|---|
| Source | arXiv, query `retrieval augmented generation`, top 40 by relevance |
| Papers ingested | 40 / 40 PDFs downloaded |
| Chunks indexed | **487** (800 chars, 120 overlap) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, local |
| Vector store | Qdrant, embedded mode (`QDRANT_PATH`), cosine distance |

## Retrieval strategy sweep

`QDRANT_PATH=./qdrant_local python -m src.eval.run_retrieval_eval` — 12
labeled questions, `top_k=5`, candidate pool 20. Keyword recall is the
fraction of expected keywords appearing anywhere in the retrieved chunks.

| strategy | keyword weight | avg keyword recall | perfect-recall questions | avg retrieval latency |
|---|---|---|---|---|
| vector_only | 0.0 | 0.8833 | 6 / 12 | **25.8 ms** |
| hybrid_light | 0.15 | 0.8833 | 7 / 12 | 34.3 ms |
| **hybrid** | **0.3** | **0.9000** | **7 / 12** | 40.8 ms |
| hybrid_heavy | 0.5 | 0.9000 | 7 / 12 | 33.9 ms |

**Blending keyword overlap into vector search improved keyword recall from
0.8833 to 0.9000 — +1.7 points, +1.9% relative — and took perfect-recall
questions from 6/12 to 7/12, at roughly 1.3-1.6x the retrieval latency.**

Reading this honestly: it is a **small** improvement on a **small** eval set.
One additional question reaching perfect recall out of twelve is a single-item
change; it is directionally consistent with the recall average moving the same
way, but 12 questions cannot separate a 1.7-point difference from noise. The
defensible claim is "hybrid did not hurt and slightly helped at this corpus
size," not "hybrid is 1.9% better."

Also worth noting: `hybrid_heavy` (0.5) matched `hybrid` (0.3) exactly rather
than continuing to improve, so the benefit saturates well before keyword
overlap dominates the blend.

## Caveats

- **No LLM query rewriting.** `run_eval.py` rewrites each question into a
  focused search query before retrieving; this harness feeds the raw question
  text. Those numbers are therefore *not* interchangeable with `run_eval.py`'s.
- **12 questions, 40 papers.** Too small to establish significance. Treat the
  ranking as a smoke test of the mechanism, not a benchmark result.
- **Keyword recall is a proxy.** It rewards retrieving chunks that contain the
  expected terms, not chunks that actually answer the question. It was chosen
  because it needs no gold-passage labels; it is not a relevance ground truth.
- Latency is CPU-only, embedded Qdrant, single process, after a warmup call
  (the first `hybrid_search` pays for lazily loading the embedding model — it
  inflated the first strategy's latency ~6x before warmup was added).

## Still to run

Requires `GOOGLE_API_KEY`:

```bash
QDRANT_PATH=./qdrant_local python -m src.eval.run_eval
```

That produces citation-validity rate, citation-presence rate, average critic
retries, and end-to-end latency per strategy — the metrics that actually
exercise the two-stage hallucination gate, which is the part of this project
worth talking about.

## Fixes this run required

Neither was a design problem; both were upstream API drift that only shows up
when you actually execute the code:

- `arxiv` 4.x removed `Result.download_pdf`, so every download failed and the
  fetch reported "0 papers" while exiting 0. Replaced with a direct HTTP GET
  against `result.pdf_url`, which is stable across package versions.
- `qdrant-client` removed `client.search()` in 1.13. Switched to
  `query_points()`, which exists from 1.12 onward.
- Added embedded-Qdrant support (`QDRANT_PATH`) so the project runs with no
  Docker daemon — see `src/vectorstore.py`.
