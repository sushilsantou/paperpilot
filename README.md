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

## Stack

| Layer | Tech |
|---|---|
| Agent orchestration | LangGraph + LangChain |
| LLM | Google Gemini (`gemini-1.5-flash`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free, no key) |
| Vector DB | Qdrant |
| Serving | FastAPI + Uvicorn |
| Experiment tracking | MLflow |
| Containerization | Docker / docker-compose |

## Project layout

```
src/
  config.py               # all tunables, loaded from .env
  ingestion/
    fetch_arxiv.py         # pull papers from the arXiv API, download PDFs
    chunk.py                # PDF text extraction + overlapping word-window chunking
    embed.py                 # local embedding model wrapper
    ingest.py                 # orchestrates fetch -> chunk -> embed -> upsert into Qdrant
  agents/
    state.py                # shared LangGraph state schema
    llm.py                    # Gemini client factory
    retriever_agent.py        # query rewrite + hybrid search
    synthesis_agent.py        # cited-answer drafting
    critic_agent.py            # structural + semantic faithfulness checks
    graph.py                    # wires the three agents into a LangGraph state machine
  eval/
    eval_set.json              # 12 labeled questions with expected-keyword sets
    metrics.py                  # keyword recall, citation validity — pure functions, unit tested
    run_eval.py                  # runs eval set across retrieval strategies, logs to MLflow
  api/
    main.py                    # FastAPI app (/health, /query)
    schemas.py                  # request/response models
tests/                          # pytest — chunking, metrics, keyword scoring (all pure, no API/network needed)
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
# 1. Pull papers + build the index (takes a few minutes the first time)
python -m src.ingestion.fetch_arxiv --query "retrieval augmented generation" --max-results 40
python -m src.ingestion.ingest

# 2. Ask a question from the CLI
python -m src.agents.graph "How does RAG reduce hallucination compared to a standalone LLM?"

# 3. Or run it as a service
uvicorn src.api.main:app --reload
# then: curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"question":"..."}'

# 4. Run the evaluation harness (compares vector-only vs. hybrid retrieval)
python -m src.eval.run_eval
mlflow ui   # open http://localhost:5000 to compare strategies
```

## Testing

```bash
pytest tests/ -v
```

All 15 tests are pure-function tests (chunking windows/overlap, keyword-recall scoring, citation
validity, keyword extraction) — they don't need Qdrant, a Gemini key, or network access, so they
run the same in CI as they do locally.

## A note on how this was validated

**Updated 2026-08-10 — the retrieval half has now been run for real:** 40 arXiv papers fetched,
487 chunks embedded and indexed into Qdrant, and all four retrieval strategies swept over the
12-question eval set. Numbers, caveats, and the two upstream API breakages this surfaced are in
[RESULTS.md](RESULTS.md). Headline: hybrid vector/keyword retrieval reached 0.90 average keyword
recall against 0.88 for pure vector search — a small gain on a small eval set, and RESULTS.md
argues explicitly against over-reading it.

Still unmeasured: everything requiring `GOOGLE_API_KEY` — query rewriting, synthesis, and the
critic's faithfulness judging, and therefore citation validity and critic-retry counts.

This project was originally scaffolded in a sandboxed environment without outbound network access
to Hugging Face, Docker Hub, or arXiv.org, so the embedding-model download, live Qdrant container,
and real Gemini calls couldn't be exercised end-to-end there. What was verified in that
environment, with real (not mocked) libraries:

- All 15 unit tests passing (chunking, metrics, keyword scoring)
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
live run with your own Gemini API key.

## Roadmap / stretch goals

- [ ] Add a second corpus type (tables/figures) for a legitimate "multimodal" claim
- [ ] Small grid-search over synthesis prompt variants, scored against the eval set
- [ ] Deploy the FastAPI service to Cloud Run or AWS App Runner
- [ ] Swap the keyword-overlap re-rank for a proper reranker model (e.g. `bge-reranker`)
