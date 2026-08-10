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

## The full agent eval: attempted, quota-blocked at n=1

`run_eval.py` was run against a real Gemini key on 2026-08-10. It completed
**one question of forty-eight** before hitting the free tier's daily cap:

```
[vector_only] q1: recall=0.80  citation_validity=1.00  retries=2  latency=38.0s
```

```
429 RESOURCE_EXHAUSTED — GenerateRequestsPerDayPerProjectPerModel-FreeTier
quotaValue: 20, model: gemini-3.5-flash
```

The free tier allows **20 requests per day per model**. This eval needs roughly
430: each question costs ~3 calls (query rewrite → synthesis → critic), and the
critic's retry loop re-runs all three per retry, with 2 retries observed.

**That single data point is not a result and is not quoted as one anywhere.**
Citation validity of 1.00 on one question is consistent with the deterministic
gate working, and it is also exactly what you would see by chance on an easy
question. n=1 measures nothing. The pipeline is *verified to run end-to-end* —
real query rewriting, real hybrid retrieval, real synthesis with inline
`[arxiv:...]` citations, a real critic loop routing work back for
re-retrieval — but citation-validity rate, citation-presence rate, average
retries, and latency remain **unmeasured** at any meaningful sample size.

Completing it needs a billing-enabled key (the whole eval is roughly $0.05–0.20
of `flash-lite` tokens) or roughly a week of daily-quota slices:

```bash
python -m src.eval.run_eval                              # full 4 x 12
python -m src.eval.run_eval --limit 2 --strategies hybrid  # a quota-sized slice
```

### One thing the attempt did surface

`retry_count` came back as **2–3 against `max_critic_retries=2`**, on both the
smoke test and q1. The critic is rejecting drafts aggressively, and the retry
bound may be off by one — worth investigating before reading anything into a
future retries-per-question average, since it triples the LLM cost per
question.

## Compatibility fixes this attempt required

None of these were visible until real API calls were made:

- **`gemini-1.5-flash` is retired** — returns 404 `NOT_FOUND` from the API.
  Default is now `gemini-3.5-flash`, chosen over the newer `gemini-3.6-flash`
  because 3.6 uses fixed sampling defaults and **silently ignores
  `temperature`**, which would make a `temperature=0` eval non-reproducible.
- **`langchain-google-genai` changed `.content` from `str` to a list of content
  blocks** (`[{'type': 'text', 'text': ...}]`), so `.content.strip()` raised
  `AttributeError: 'list' object has no attribute 'strip'` in all three agents.
  `src/agents/llm.py::message_text()` now normalises this in one place and
  still accepts the old string form.
- **No rate-limit handling at all.** `invoke_with_retry()` now backs off on 429
  honouring the server's `retryDelay`, but fails fast on a *per-day* cap rather
  than sleeping for hours, and `run_eval.py` keeps partial results with an
  explicit `questions_completed` count instead of losing the whole run.

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
