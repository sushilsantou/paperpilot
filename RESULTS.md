# PaperPilot — Measured Results

First run against a real corpus, 2026-08-10. Every number below came from a
command in this repo; nothing is estimated.

**Scope, stated up front:** what is measured here is the *retrieval* half of
the system — the part that needs no LLM. The full agent pipeline (LLM query
rewriting → synthesis → critic) has been **run end-to-end against a live Gemini
key but not measured**: the eval hit the free tier's daily cap after 1 of 48
questions. Citation validity, critic-retry counts and end-to-end latency
therefore remain unmeasured at any meaningful sample size. Details in
[The full agent eval](#the-full-agent-eval-attempted-quota-blocked-at-n1).

Two results are reported below that did not come out the way the design
intended — graph-expanded retrieval measured as no improvement, and the critic's
retry bound turned out to be an unbounded loop rather than the off-by-one it
first looked like. Both are here rather than quietly dropped.

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

| strategy | kw weight | graph weight | avg keyword recall | perfect-recall questions | avg retrieval latency |
|---|---|---|---|---|---|
| vector_only | 0.0 | — | 0.8833 | 6 / 12 | **17.7 ms** |
| hybrid_light | 0.15 | — | 0.8833 | 7 / 12 | 28.5 ms |
| **hybrid** | **0.3** | **—** | **0.9000** | **7 / 12** | 27.4 ms |
| hybrid_heavy | 0.5 | — | 0.9000 | 7 / 12 | 26.1 ms |
| graph_light | 0.3 | 0.2 | 0.9000 | 7 / 12 | 64.3 ms |
| graph | 0.3 | 0.4 | 0.9000 | 7 / 12 | 63.4 ms |
| graph_heavy | 0.3 | 0.6 | 0.9000 | 7 / 12 | 67.7 ms |

**Blending keyword overlap into vector search improved keyword recall from
0.8833 to 0.9000 — +1.7 points, +1.9% relative — and took perfect-recall
questions from 6/12 to 7/12, at roughly 1.5x the retrieval latency.**

**Reproducibility.** The recall and perfect-recall columns reproduced *exactly*
across three separate runs of this sweep on the same corpus. The latency column
did not — for the four hybrid rows:

| run | vector_only | hybrid_light | hybrid | hybrid_heavy |
|---|---|---|---|---|
| 1 (4 strategies) | 25.8 | 34.3 | 40.8 | 33.9 |
| 2 (7 strategies, table above) | 17.7 | 28.5 | 27.4 | 26.1 |
| 3 (7 strategies) | 10.1 | 15.4 | 14.1 | 12.6 |

Retrieval is deterministic, so recall is stable by construction; latency is
wall-clock on a shared CPU and moves by more than 2x across runs. **Do not read
latency differences under ~2x as signal.** Notably, run 1 ranks `hybrid` as the
*slowest* of the four and run 3 ranks it second-fastest — the within-run
ordering of those rows is noise.

What did survive all three runs is the graph/hybrid gap: 63.4 vs 27.4 ms in run
2 and 30.6 vs 14.1 ms in run 3, i.e. **2.2-2.3x in both**. That ratio is stable
even though the absolute numbers are not, which is why it is quoted as a ratio
below and not as a millisecond figure.

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

## Graph-expanded retrieval: a measured null result

`src/graph/` builds a knowledge graph over the corpus from metadata alone —
**558 nodes** (40 papers, 165 authors, 12 categories, 341 TF-IDF concepts) and
**745 edges** — and expands retrieval one hop along it. The motivating failure
is specific: vector search ranks chunks by how much they *read* like the query,
so a paper solving the same problem in different vocabulary loses to one that
merely echoes the query's wording. Following shared concepts and categories
should reach those papers by relation rather than phrasing.

**It did not help.** All three graph weights (0.2 / 0.4 / 0.6) match plain
`hybrid` exactly — same 0.9000 recall, same 7/12 perfect, same 3.5 distinct
papers per answer — at 63-68 ms against hybrid's 27 ms. The graph costs ~2.3x
latency and moves no measured metric.

`graph_weight=0.0` degrades exactly to `hybrid_search`, so these rows are an
A/B on identical seeds, not a comparison against a different pipeline. The
implementation is committed anyway, as a measured negative rather than deleted.

**One caveat cuts in the graph's favour, and it is a real one.** Keyword recall
rewards retrieving chunks that contain the expected terms. A retrieval strategy
whose entire purpose is reaching papers that use *different* vocabulary is the
one strategy this metric is structurally least able to credit. The honest
statement is "no gain on this metric at this corpus size", not "graph expansion
is worthless" — and distinguishing those needs a relevance measure this eval
does not have. Two other things likely matter as much: 40 topically-scoped
papers is a small neighbourhood to expand into, and of 165 authors only **4**
appear on more than one paper, so co-authorship connects almost nothing.
Categories carry nearly all the connectivity, and they are coarse — `cs.IR`
alone covers 29 of the 40 papers, so it barely discriminates.

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

**What the quota does not block.** The *behaviour* of the agent loop is verified
without a key at all: `python -m src.selfcheck` stubs only the LLM and the
embedding model — the two things that need credentials or a network — and runs
everything else as real production code. It confirms the hallucination gate
rejects a fabricated citation without wasting a judge call, that insufficient
context routes back to the retriever rather than to synthesis, and that the
retry loop is bounded. That is how the unbounded-loop bug below was reproduced
and fixed despite the eval itself being quota-blocked. What remains unmeasured
is the *quality* of real model output, not whether the machinery works.

### One thing the attempt did surface — and it was not an off-by-one

`retry_count` came back as **2–3 against `max_critic_retries=2`**, on both the
smoke test and q1. That was originally recorded here as a suspected off-by-one
in the retry bound. It was not. It was an **unbounded loop**.

The cap was checked only *after* the LLM judge call, and the critic's
structural citation check returns before ever reaching it:

```python
structural_issue = _structural_check(state)
if structural_issue:
    return {... "retry_count": retry_count + 1}   # never consults the cap
...
response = invoke_with_retry(llm, prompt)
if retry_count >= settings.max_critic_retries:    # cap lived only here
```

So a synthesis agent that kept citing a source which was never retrieved
incremented `retry_count` forever. Reproduced offline with
`max_critic_retries=2`: **5003 synthesis calls** before LangGraph's recursion
limit stopped it. Against a live key that is 5003 billed calls from a single
question — on a project already blocked by a 20-requests/day quota.

The observed 2–3 came from *mixed* runs, which is what disguised it: a
structural failure on the final pass pushed the counter past the cap, and the
next pass approved before the loop could run away. A run that failed
structurally every time would have hung instead.

**Fixed.** Every rejection now routes through one `_reject()` that owns the
bound; the same repro produces 3 synthesis calls and stops at
`retry_count=2`. The give-up path also no longer reports
`critic_verdict="approved"` — see the next section. Regression test:
`test_repeated_structural_failures_are_bounded`.

The lesson worth keeping: the existing bounded-retry test passed throughout,
because it only ever drove the *semantic* rejection path, where the cap did
apply. A second rejection path was added later and silently escaped the bound
the first one was tested against.

## Metrics added before the real eval run

Two metrics exist in `run_eval.py` that have never been reported, because the
run that would report them is quota-blocked. They are noted here so the eventual
numbers are read correctly:

- **`unverified_rate`** — the fraction of answers returned by hitting the retry
  cap rather than passing the critic. Previously unmeasurable: the give-up path
  reported `critic_verdict="approved"`, so an answer the critic never accepted
  was indistinguishable from one it did.
- **`avg_citation_validity_verified`** — citation validity on answers that
  actually passed the gate. `avg_citation_validity` keeps its existing pooled
  meaning.

The distinction matters for the headline number. In a synthetic 3-question case
where one answer hit the cap, pooled validity reads 0.667 while validity on
answers that passed reads 1.000 — the same run tells two different stories, and
only one of them is a claim about whether the citation gate works. Any future
citation-validity figure quoted from this project should be accompanied by
`unverified_rate`, or it is not interpretable.

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

None of these were design problems; the first two were upstream API drift that
only shows up when you actually execute the code, and the third was a missing
affordance:

- `arxiv` 4.x removed `Result.download_pdf`, so every download failed and the
  fetch reported "0 papers" while exiting 0. Replaced with a direct HTTP GET
  against `result.pdf_url`, which is stable across package versions.
- `qdrant-client` removed `client.search()` in 1.13. Switched to
  `query_points()`, which exists from 1.12 onward.
- Added embedded-Qdrant support (`QDRANT_PATH`) so the project runs with no
  Docker daemon — see `src/vectorstore.py`.
