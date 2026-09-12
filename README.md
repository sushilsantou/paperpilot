# PaperPilot — Multi-Agent Research Assistant with a Verified Citation Loop

A multi-agent Retrieval-Augmented Generation (RAG) system over arXiv ML/AI papers. Unlike a
single-shot "retrieve then generate" pipeline, answers here pass through a **critic agent** that
checks every citation against what was actually retrieved and can send the pipeline back for
another retrieval pass or another draft — before anything is returned to the user.

Built as a portfolio project to demonstrate multi-agent orchestration, vector search, and
measured (not vibes-based) retrieval quality — evaluated across strategies and tracked in MLflow.

## Architecture

```
                ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   question ──▶ │  Retriever  │ ─────▶ │  Synthesis  │ ─────▶ │   Critic    │
                │   Agent     │        │    Agent    │        │   Agent     │
                └─────────────┘        └─────────────┘        └──────┬──────┘
                       ▲                      ▲                      │
                       │                      │                      │
                       │   needs_retrieval    │    needs_rewrite      │
                       └──────────────────────┴───────────┬──────────┘
                                                            │ approved
                                                            ▼
                                                      final_answer
```

- **Retriever agent** — rewrites the incoming question into a focused search query (via Gemini),
  then runs hybrid retrieval against Qdrant: vector similarity (embeddings from a local
  `sentence-transformers` model, no API key needed) blended with a keyword-overlap re-rank.
- **Synthesis agent** — drafts an answer using *only* the retrieved excerpts, with inline
  `[arxiv:ID]` citations required on every claim.
- **Critic agent** — two-stage check: (1) a structural pass that catches citations to sources that
  were never retrieved (a cheap, deterministic hallucination check), then (2) an LLM-as-judge pass
  that verifies the draft's claims are actually supported by the excerpts. Routes back to the
  retriever (if context was insufficient) or back to synthesis (if the draft misused good context).

Every rejection is bounded by `max_critic_retries`. If the critic never approves, the loop stops
and returns the best draft with verdict `unverified` and a caveat naming the last issue — never
`approved`, so the one field recording whether the citation gate passed cannot say the opposite of
what happened. That bound is load-bearing: an earlier version checked it on only one of the two
rejection paths, and a draft that kept citing a nonexistent source produced 5003 model calls from a
single question. See [RESULTS.md](RESULTS.md#one-thing-the-attempt-did-surface--and-it-was-not-an-off-by-one).

## Stack

| Layer | Tech |
|---|---|
| Agent orchestration | LangGraph + LangChain |
| LLM | Google Gemini (`gemini-3.5-flash`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free, no key) |
| Vector DB | Qdrant (server, or embedded via `QDRANT_PATH`) |
| Knowledge graph | hand-rolled, built from paper metadata + TF-IDF concepts |
| Serving | FastAPI + Uvicorn |
| Experiment tracking | MLflow |
| Containerization | Docker / docker-compose |

`gemini-1.5-flash` was the original default and is now retired (404 from the API). `gemini-3.5-flash`
is chosen over the newer `gemini-3.6-flash` because 3.6 uses fixed sampling defaults and silently
ignores `temperature`, which would make a `temperature=0` eval non-reproducible.

## Project layout

```
src/
  config.py                 # all tunables, loaded from .env
  vectorstore.py            # Qdrant client factory (server or embedded)
  selfcheck.py              # runs the whole agent loop offline — no key, no network
  stubs.py                  # the only two test doubles: the LLM and the embedding model
  ingestion/
    fetch_arxiv.py          # pull papers from the arXiv API, download PDFs
    chunk.py                # PDF text extraction + overlapping word-window chunking
    embed.py                # local embedding model wrapper
    ingest.py               # orchestrates fetch -> chunk -> embed -> upsert into Qdrant
  metadata/
    schema.py               # typed chunk payload + which fields are Qdrant-indexed
  agents/
    state.py                # shared LangGraph state schema
    llm.py                  # Gemini client factory, 429 backoff, content-block normalising
    retriever_agent.py      # query rewrite + hybrid search
    synthesis_agent.py      # cited-answer drafting
    critic_agent.py         # structural + semantic faithfulness checks, bounded retries
    graph.py                # wires the three agents into a LangGraph state machine
  graph/
    knowledge_graph.py      # Paper/Author/Category/Concept graph built from metadata
    graph_retrieval.py      # vector search + one hop along the graph (measured: no gain)
  eval/
    eval_set.json           # 12 labeled questions with expected-keyword sets
    metrics.py              # keyword recall, citation validity — pure functions, unit tested
    run_eval.py             # full agent eval across strategies, logs to MLflow (needs a key)
    run_retrieval_eval.py   # retrieval-only sweep — no LLM, no key required
  api/
    main.py                 # FastAPI app (/health, /query)
    schemas.py              # request/response models
tests/                      # pytest — 27 tests, none needing a key, a server, or network
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GOOGLE_API_KEY (get one free at https://aistudio.google.com/apikey)
#
# Version note: requirements.txt pins the versions this was written against.
# RESULTS.md was produced on Python 3.13.14 with unpinned installs resolving to
# langchain 1.3.14, langgraph 1.2.10, qdrant-client 1.19.0,
# sentence-transformers 5.7.0, arxiv 4.0.1, mlflow 3.15.1. Install unpinned if
# the pinned set fails on your Python.

docker compose up -d qdrant     # or: docker run -p 6333:6333 qdrant/qdrant
```

**No Docker?** Set `QDRANT_PATH` instead and qdrant-client runs the store
embedded, in-process, against local files — same client API, no server:

```bash
export QDRANT_PATH=./qdrant_local    # Windows: set QDRANT_PATH=.\qdrant_local
```

Embedded mode holds an exclusive lock on that directory, so one process may
use it at a time (ingest, *then* eval, *then* serve — not concurrently). When
`QDRANT_PATH` is unset, `QDRANT_URL` is used and nothing changes. See
`src/vectorstore.py`.

## Running it

```bash
# 0. Check the pipeline works before spending a single API call. No key, no
#    network, no Qdrant server — stubs only the LLM and the embedder and runs
#    everything else as real code. Start here.
python -m src.selfcheck

# 1. Pull papers + build the index (takes a few minutes the first time)
python -m src.ingestion.fetch_arxiv --query "retrieval augmented generation" --max-results 40
python -m src.ingestion.ingest
python -m src.graph.knowledge_graph            # optional: build the knowledge graph

# 2. Ask a question from the CLI
python -m src.agents.graph "How does RAG reduce hallucination compared to a standalone LLM?"

# 3. Or run it as a service
uvicorn src.api.main:app --reload
# then: curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"question":"..."}'

# 4a. Retrieval-only sweep — no LLM, no API key, runs in seconds
python -m src.eval.run_retrieval_eval

# 4b. Full agent eval — needs a key, and ~430 requests against a 20/day free tier
python -m src.eval.run_eval
python -m src.eval.run_eval --limit 2 --strategies hybrid   # a quota-sized slice

mlflow ui   # open http://localhost:5000 to compare strategies
```

**Step 0 is the one worth knowing about.** The Gemini free tier allows 20
requests/day/model and the full eval needs roughly 430, so "did this fail
because the wiring is broken, or because of my key?" is a question you will
otherwise be asking with no way to answer it. `src.selfcheck` answers it
offline: if it passes, everything except the model calls themselves works.

## Testing

```bash
pytest tests/ -v
```

**27 tests, none of which need a Gemini key, a Qdrant server, or network access** — so they run the
same in CI as they do locally. They split into two kinds:

- **19 pure-function tests** — chunking windows/overlap, keyword-recall scoring, citation validity,
  keyword extraction, eval aggregation.
- **8 tests that drive the real pipeline** — the actual LangGraph state machine, the critic's
  structural check, the retry loop and Qdrant search, against an in-memory Qdrant. Only the LLM and
  the embedding model are stubbed, because they are the only two things here that need credentials
  or a network (`src/stubs.py`).

That second group is the point: it means "the critic rejects a fabricated citation without wasting
a judge call" and "the retry loop is bounded" are *tested claims* rather than described ones. The
unbounded-loop bug in RESULTS.md was found and fixed entirely through this path, with the eval
itself still quota-blocked.

## A note on how this was validated

**Updated 2026-09-12 — the retrieval half has been run for real:** 40 arXiv papers fetched,
487 chunks embedded and indexed into Qdrant, and seven retrieval strategies swept over the
12-question eval set. Numbers, caveats, and the upstream API breakages this surfaced are in
[RESULTS.md](RESULTS.md). Headline: hybrid vector/keyword retrieval reached 0.90 average keyword
recall against 0.88 for pure vector search — a small gain on a small eval set, and RESULTS.md
argues explicitly against over-reading it.

**Graph-expanded retrieval was built, measured, and did not work.** A knowledge graph over the
corpus (558 nodes, 745 edges, built from metadata with TF-IDF concepts) expands retrieval one hop
along shared concepts and categories. At every weight tried it matched plain hybrid *exactly* on
every metric, at ~2.3x the latency. It is committed as a measured negative rather than deleted,
with the caveat that keyword recall is structurally poor at crediting a strategy whose purpose is
reaching differently-worded papers — see
[RESULTS.md](RESULTS.md#graph-expanded-retrieval-a-measured-null-result).

The LLM half has been **run but not measured**. With a real Gemini key the full pipeline executes
end-to-end — query rewriting, hybrid retrieval, synthesis with inline `[arxiv:...]` citations, and
the critic loop routing work back for re-retrieval — but the eval hit the free tier's cap of
20 requests/day/model after **1 of 48 questions** (it needs ~430, since each critic retry re-runs
rewrite + synthesis + critic). So citation-validity rate, citation-presence rate, average retries,
and latency remain unmeasured at any real sample size. One question is not a result and is not
quoted as one. Details in [RESULTS.md](RESULTS.md).

This project was originally scaffolded in a sandboxed environment without outbound network access
to Hugging Face, Docker Hub, or arXiv.org, so the embedding-model download, live Qdrant container,
and real Gemini calls couldn't be exercised end-to-end there. What was verified in that
environment, with real (not mocked) libraries:

- The unit tests passing (chunking, metrics, keyword scoring)
- Real PDF generation + `pypdf` text extraction + chunking on actual PDF bytes
- A full ingestion → embedding → Qdrant upsert → hybrid-search smoke test against a real
  (embedded/in-memory) Qdrant instance — with a deterministic stand-in for the embedding model,
  confirming the topically-correct document ranked first under both vector-only and hybrid scoring
- The LangGraph state machine compiling and wiring correctly against the real `langgraph`/`langchain` APIs
- FastAPI `/health` and `/query` error-handling paths (missing Qdrant, missing API key) via `TestClient`
- Caught and fixed a deprecated dependency (`langchain-google-genai` was pinned to a version built on
  Google's now-sunset `google.generativeai` SDK; upgraded the whole LangChain/LangGraph stack to the
  current major versions built on `google.genai`)

Everything around the LLM path has since been exercised for real (see above). The LLM-dependent
path itself — query rewriting, synthesis, and the critic's faithfulness judging — still needs a
live run with your own Gemini API key to be *measured*. Its **behaviour** no longer depends on one:
`python -m src.selfcheck` and the 8 pipeline tests exercise the real graph, routing, structural
check and retry bound offline. That is how the unbounded-retry bug was caught and fixed while the
eval itself remained quota-blocked.

## Roadmap / stretch goals

- [x] ~~Graph/relational retrieval over paper metadata~~ — built and measured; **no gain at 2.3x
      latency**, see [RESULTS.md](RESULTS.md#graph-expanded-retrieval-a-measured-null-result)
- [ ] Finish the full agent eval on a billing-enabled key (~$0.05–0.20) and report
      `unverified_rate` alongside citation validity
- [ ] Replace keyword recall with a relevance measure that can actually credit semantic retrieval —
      the current metric cannot distinguish "the graph didn't help" from "the metric can't see it"
- [ ] Add a second corpus type (tables/figures) for a legitimate "multimodal" claim
- [ ] Small grid-search over synthesis prompt variants, scored against the eval set
- [ ] Deploy the FastAPI service to Cloud Run or AWS App Runner
- [ ] Swap the keyword-overlap re-rank for a proper reranker model (e.g. `bge-reranker`)
