# PaperPilot — Measured Results

First run against a real corpus, 2026-08-10. Every number below came from a
command in this repo; nothing is estimated.

**Scope, stated up front:** what is fully measured here is the *retrieval* half
of the system — the part that needs no LLM. The full agent pipeline (LLM query
rewriting → synthesis → critic) has been **run end-to-end against a live Gemini
key but not measured at scale**: the free tier's 20-requests/day cap against an
eval needing ~430 means it has only ever been sampled a few questions at a time.
Citation-validity *rates*, approval rates and latency distributions remain
unmeasured. Details in
[The full agent eval](#the-full-agent-eval-sampled-never-completed).

Those few live questions were not wasted, though. They are what surfaced the
critic/synthesis context mismatch below — a bug that made the citation gate
reject every answer it was given. The offline suite could not see it: the stub
judge returns a scripted verdict regardless of what context it is handed, so a
judge starved of evidence behaves identically to a well-fed one. The fix is now
guarded offline by asserting the two agents' context formatters agree, which is
a check that could have been written in advance — but only by someone who
already suspected the mismatch. A live run is what raised the suspicion.

**Four results below did not come out the way the design intended**, and all
four are reported rather than quietly dropped:

1. Graph-expanded retrieval measured as **no improvement** at 2.3x the latency.
2. The critic's retry bound was not the off-by-one it first looked like — it was
   an **unbounded loop**, 5003 model calls from one question.
3. The critic was **rejecting every answer**, because it was shown about a
   tenth of the evidence the draft was written from.
4. The citation gate could **approve answers it never checked**: an
   unparseable judge reply counted as a pass, and a draft with no citations
   skipped the structural check.

## Corpus

| | |
|---|---|
| Source | arXiv, query `retrieval augmented generation`, top 40 by relevance |
| Papers ingested | 40 / 40 PDFs downloaded |
| Chunks indexed | **487** (800 words, 120-word overlap; median 5,496 characters) |
| Second index | 1,916 chunks (200 words, 30-word overlap), `arxiv_papers_200`, for the [chunk-size comparison](#chunk-size-200-words-vs-800-at-equal-text-budget) only |
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
across four separate runs of this sweep on the same corpus — every run returned
0.8833 / 0.8833 / 0.9000 / 0.9000 and 6 / 7 / 7 / 7. The latency column did not:

| run | vector_only | hybrid_light | hybrid | hybrid_heavy |
|---|---|---|---|---|
| 1 (4 strategies) | 25.8 | 34.3 | 40.8 | 33.9 |
| 2 (7 strategies, table above) | 17.7 | 28.5 | 27.4 | 26.1 |
| 3 (7 strategies) | 10.1 | 15.4 | 14.1 | 12.6 |
| 4 (7 strategies) | 8.8 | 13.9 | 13.8 | 14.8 |

Retrieval is deterministic, so recall is stable by construction; latency is
wall-clock on a shared CPU and moves by more than 4x across runs (25.8 ms to
8.8 ms on the same row). **Do not read latency differences under ~2x as
signal.** Notably, run 1 ranks `hybrid` as the *slowest* of the four, run 3
ranks it second-fastest, and run 4 ranks it second-slowest — the within-run
ordering of those rows is noise.

What survived every run is the graph/hybrid gap: 63.4 vs 27.4 ms in run 2,
30.6 vs 14.1 in run 3, 30.7 vs 13.8 in run 4 — **2.2-2.3x each time**. That
ratio is stable even though the absolute numbers are not, which is why it is
quoted as a ratio below and not as a millisecond figure.

This sweep has no LLM in it — it feeds raw question text to `hybrid_search` and
never constructs an agent — so none of the agent-side fixes (the retry bound,
the critic context mismatch) can move these numbers, and re-running after them
confirms that they do not. The caveat below about missing query rewriting is the
same point from the other direction.

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

## Chunk size: 200 words vs 800, at equal text budget

Run 2026-10-25. A second index over the same 40 PDFs with the same embedder,
in its own collection so the 800-word index above is untouched:

```
CHUNK_SIZE=200 CHUNK_OVERLAP=30 QDRANT_COLLECTION=arxiv_papers_200 python -m src.ingestion.ingest
QDRANT_COLLECTION=arxiv_papers_200 python -m src.eval.run_retrieval_eval            # top_k=5
QDRANT_COLLECTION=arxiv_papers_200 TOP_K=20 python -m src.eval.run_retrieval_eval   # top_k=20
```

That produced **1,916 chunks** (vs 487). Overlap is 15% of the chunk in both
indexes, so the two differ only in granularity. Same 12 questions, raw
question text, candidate pool 20.

| strategy | 800w, top_k=5 (above) | 200w, top_k=5 | **200w, top_k=20** |
|---|---|---|---|
| vector_only | 0.8833 · 6/12 | 0.7500 · 2/12 | **0.9667 · 10/12** |
| hybrid_light | 0.8833 · 7/12 | 0.8000 · 3/12 | 0.9667 · 10/12 |
| hybrid | 0.9000 · 7/12 | 0.7833 · 3/12 | 0.9667 · 10/12 |
| hybrid_heavy | 0.9000 · 7/12 | 0.7833 · 3/12 | 0.9667 · 10/12 |
| graph_light | 0.9000 · 7/12 | 0.7833 · 3/12 | 0.9500 · 9/12 |
| graph | 0.9000 · 7/12 | 0.7833 · 3/12 | 0.9500 · 9/12 |
| graph_heavy | 0.9000 · 7/12 | 0.7833 · 3/12 | 0.9500 · 9/12 |

Cells are avg keyword recall · perfect-recall questions. Latency is omitted:
see the reproducibility note above.

**At the same top_k, small chunks lose on every row, and that comparison is
rigged.** Five 200-word chunks are about 1,000 words of retrieved text against
about 4,000 for five 800-word chunks, and keyword recall counts expected terms
*anywhere* in the retrieved text. A quarter of the text finds fewer terms more
or less automatically. The 200w/top_k=5 column is reported because it was run,
not because it measures chunk size.

**At equal text budget, small chunks win: 0.9667 vs 0.9000 recall, and 10/12
questions at perfect recall vs 7/12.** Twenty 200-word chunks are the same
~4,000 words as five 800-word chunks. The likely explanation, not tested here,
is that a budget spent in smaller pieces covers more distinct passages rather
than the unrelated stretches inside a few long ones. Three more questions at perfect recall is a larger move than hybrid's
single-question gain, though on 12 questions it is still a small one.

Three things limit what this shows:

- **The top_k=20 column says nothing about keyword blending.** `hybrid_search`
  re-ranks a candidate pool of 20 down to `top_k`. At `top_k=20` it returns the
  whole pool, so the keyword weight can only reorder chunks, not change which
  ones come back. The four identical hybrid rows are true by construction, not
  a finding. Testing the blend at this granularity needs a pool larger than
  `top_k`.
- **The graph lost one question** (0.9500, 9/12, against vector_only's
  0.9667, 10/12). That is a single-item change on 12 questions. It fits the
  null result above and does not show the graph causing harm.
- **The metric probably favours this result.** More, smaller passages means
  more varied vocabulary in the retrieved text, which is exactly what keyword
  recall counts. Whether the extra passages are *relevant* needs the relevance
  labels this eval does not have, plus a live agent run to see whether
  synthesis does better with 20 short excerpts than with 5 long ones.

**Nothing in the system was changed.** The agent still retrieves from the
800-word `arxiv_papers` at `TOP_K=5`. Switching would mean indexing at 200
words *and* raising `TOP_K` to 20: the prompt stays about the same size, but
the critic judges four times as many separate excerpts. That is worth a
measured agent run before it becomes the default. Each configuration was run
once. Recall reproduced exactly across four runs of the 800-word sweep, so it
is expected to be stable here too, but that has not been checked for this index.

## The full agent eval: sampled, never completed

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
(It is also now known to have been an *unverified* answer — `retries=2` is the
retry cap, so the critic never approved it. See the 2026-09-12 section above.)
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

### 2026-09-12: the critic was rejecting everything, and it was right to

Two questions were run live against `hybrid` on the fixed retry bound. **Both
hit `max_critic_retries` and came back unverified:**

```
[hybrid] q1: recall=1.00 citation_validity=1.00 retries=2 latency=78.4s  [UNVERIFIED]
[hybrid] q2: recall=1.00 citation_validity=0.67 retries=2 latency=34.1s  [UNVERIFIED]
```

`unverified_rate` was **1.00** — the critic never approved anything. Note q1:
citation validity of 1.00 on an answer that was never approved. Under the
pre-`unverified` code this would have been logged as `approved` with perfect
citation validity, which is precisely the misreading the metric was added to
prevent. It earned its place on the first live question.

**The cause was a context mismatch, not a strict judge.** The critic truncated
every chunk to `text[:500]` while synthesis received the full text. Chunks are
800 *words*, a median of 5,496 characters across the 487 indexed chunks, so the
judge saw on average about **10%** of each chunk the draft was written from:
any claim drawn from past a chunk's first few sentences looked unsupported
because the supporting text was never in the judge's prompt. The critic was
being shown less than the writer and rejecting the difference.

*Correction, 2026-10-25:* this section originally said chunks were 800
characters and the judge saw 62% of the evidence. `chunk.py` windows on words,
not characters; the 10% figure is measured over the indexed collection
(the mean of `min(500, len(chunk)) / len(chunk)`). The fix and the
before/after below are unaffected; the mismatch was larger than first reported.

Re-running q1 after the fix — same question, same strategy, same corpus:

| | retries | latency | verdict | citation validity |
|---|---|---|---|---|
| before | 2 | 78.4 s | **unverified** | 1.00 (never approved) |
| after | **0** | **14.7 s** | **approved, first pass** | 1.00 (verified) |

Approved on the first pass instead of never, and **5.3x faster** because the
retry loop is gone. The retries were not buying quality; they were re-running
the whole pipeline against a judge that could not see the evidence.

**What this is and is not.** It is a controlled before/after on one fixed
question, and it points one way. It is **not** a measured approval rate: n=1
after the fix, and q2's re-run was blocked when the daily quota ran out
mid-question — the harness stopped cleanly with `questions_completed: 0` rather
than losing the run, which is the partial-results handling working as intended.
An `unverified_rate` across the full 12 questions is still unmeasured.

One further caveat, found later: at the time, an unparseable judge reply was
also recorded as `approved` and left no trace of which branch produced the
verdict, so the "after" row cannot be shown, after the fact, to be a judge
ruling rather than a fail-open. See
[2026-10-25](#2026-10-25-two-ways-past-the-citation-gate-without-being-checked).

**Re-run 2026-10-25, on code with both gate fixes in.** Same question (q1),
same strategy (`hybrid`), same 800-word index. On this code an approval can
only come from a judge that returned a parseable `faithful: true` on a draft
that cites retrieved sources:

```
python -m src.eval.run_eval --offset 0 --limit 1 --strategies hybrid --out eval_slice5.json
[hybrid] q1: recall=0.80 citation_validity=1.00 retries=1 latency=32.8s
```

| | retries | latency | verdict | citation validity |
|---|---|---|---|---|
| before the fix (2026-09-12) | 2 | 78.4 s | **unverified** | 1.00 (never approved) |
| after, current code (2026-10-25) | 1 | 32.8 s | **approved, after one retry** | 1.00 (verified) |

That answers the open question above: q1 does pass the gate when a fail-open
is impossible. It **did not pass on the first attempt**, though. One draft was
rejected before the second was approved, so the speed-up is **2.4x**, not the
5.3x reported in September. Why that first draft was rejected cannot be read
back from this run: `run_eval.py` records only the final verdict, not each
rejection's reason, so a judge ruling, a malformed judge reply and the new
uncited-draft check all look the same. Query rewriting and synthesis are also
nondeterministic, so September's 0 retries and today's 1 are two single
samples of the same question, not a trend. What both runs agree on is the
verdict: approved, where both live questions before the fix hit the retry cap.

One retrospective correction this forces: the historical data point quoted
below (`retries=2`) hit the retry cap, which means it was almost certainly an
unverified answer reported without that caveat. Its `citation_validity=1.00`
described an answer the critic never accepted.

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

### 2026-10-25: two ways past the citation gate without being checked

Found in code review, not in a live run. Both let an answer come back
`approved` without the gate ever verifying it:

- **An unparseable judge reply was an approval.** If the judge returned prose
  instead of JSON, the critic substituted `faithful: true` and approved the
  draft, with `critic_feedback` left empty, so the final state could not
  distinguish "the judge ruled it faithful" from "the judge said nothing
  usable". Reproduced offline with a stub judge returning
  `"I think this looks fine, honestly"`: verdict `approved`.
- **A draft with no citations skipped the structural check.** The check
  compared cited IDs against retrieved ones; an empty set has nothing to
  compare, so it passed, and the judge asks whether claims are *supported*,
  not whether they are *cited*. A citation gate could approve an answer with
  no citations.

**Fixed.** A reply the parser cannot read (no JSON object, or no boolean
`faithful`) is now a rejection through `_reject()`, so it stays inside the
retry bound and ends `unverified` at the cap. The parser was loosened at the
same time to accept a JSON object wrapped in fences or a sentence of prose,
so harmless formatting does not become a rejection. An uncited draft is now a
structural rejection routed to the **retriever**, not synthesis: a writer that
cited nothing most likely found nothing it could attribute, and a fresh search
addresses that, where re-drafting against the same chunks only helps if it
forgot the format. Regression tests:
`test_unparseable_judge_is_a_failed_check_not_an_approval`,
`test_uncited_draft_is_rejected_structurally_and_re_retrieves`, and the parser
cases in `tests/test_critic_parsing.py`.

**What this changes about earlier numbers.** Nothing measured, because nothing
at scale has been measured: three live questions total. But it does weaken one
claim: the 2026-09-12 "after" row (q1 approved on the first pass) was produced
by code where a fail-open looked identical to a real approval, so it is
consistent with the context fix working and is not proof of it. q1 has since
been re-run on this version and was approved after one retry (see the
[re-run](#2026-09-12-the-critic-was-rejecting-everything-and-it-was-right-to)).
The full eval should run on this version, and its `unverified_rate` now counts
malformed judge replies and uncited drafts as unverified rather than approved.

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
