"""
Graph-expanded retrieval: vector search, then one hop through the knowledge
graph to pull in chunks the embedding never would have surfaced.

The failure this targets is specific. Vector search ranks chunks by how much
they *read* like the query, so a paper that solves the same problem in
different vocabulary loses to one that merely echoes the query's wording.
Following graph edges — shared concepts, categories, authors — reaches those
papers by relation rather than by phrasing.

How it works:

1. Run normal hybrid search for a seed set.
2. Take the papers those seeds came from.
3. Ask the graph for related papers (two hops: paper -> concept/category/author
   -> paper), scored by relation weight.
4. Pull chunks from those neighbours with a *filtered* vector search — this is
   where the payload index earns its place, since `source_id` filtering happens
   server-side rather than by fetching everything and discarding most of it.
5. Blend: a neighbour chunk keeps its own similarity, discounted by how far it
   sits from the seed set.

`graph_weight=0.0` degrades exactly to `hybrid_search`, so the eval can A/B the
graph's contribution on identical seeds rather than against a different
pipeline.
"""
from typing import Dict, List, Optional

from qdrant_client import QdrantClient, models

from src.agents.retriever_agent import _keyword_overlap_score, _keywords, hybrid_search
from src.agents.state import RetrievedChunk
from src.config import settings
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.embed import embed_texts


def _chunks_from_papers(
    client: QdrantClient,
    query_vector: List[float],
    paper_ids: List[str],
    limit: int,
) -> List[tuple]:
    """Vector search restricted to a set of papers, using the source_id index."""
    if not paper_ids:
        return []
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=limit,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="source_id", match=models.MatchAny(any=paper_ids))]
        ),
    ).points
    return [(h.score, h.payload or {}) for h in hits]


def graph_expanded_search(
    client: QdrantClient,
    search_query: str,
    graph: KnowledgeGraph,
    top_k: int = 5,
    candidate_pool: int = 20,
    keyword_weight: float = 0.3,
    graph_weight: float = 0.3,
    seed_papers_considered: int = 3,
    neighbours_per_seed: int = 3,
) -> List[RetrievedChunk]:
    seeds = hybrid_search(
        client,
        search_query=search_query,
        top_k=candidate_pool,
        candidate_pool=candidate_pool,
        keyword_weight=keyword_weight,
    )
    if graph_weight <= 0.0 or not seeds:
        return seeds[:top_k]

    # Seed papers, in the order the retriever ranked them.
    seed_paper_ids: List[str] = []
    for chunk in seeds:
        if chunk["source_id"] not in seed_paper_ids:
            seed_paper_ids.append(chunk["source_id"])
    considered = seed_paper_ids[:seed_papers_considered]

    # Neighbours, keeping the best relatedness score seen for each.
    neighbour_scores: Dict[str, float] = {}
    for pid in considered:
        for neighbour, score in graph.related_papers(pid, limit=neighbours_per_seed):
            if neighbour in seed_paper_ids:
                continue  # already retrieved directly; no need to re-rank it in
            neighbour_scores[neighbour] = max(neighbour_scores.get(neighbour, 0.0), score)

    scored: List[tuple] = [(c["score"], c, 1.0) for c in seeds]

    if neighbour_scores:
        query_vector = embed_texts([search_query])[0].tolist()
        query_kw = _keywords(search_query)
        max_relatedness = max(neighbour_scores.values()) or 1.0

        for sim, payload in _chunks_from_papers(
            client, query_vector, list(neighbour_scores), limit=candidate_pool
        ):
            pid = payload.get("source_id", "")
            relatedness = neighbour_scores.get(pid, 0.0) / max_relatedness
            kw = _keyword_overlap_score(query_kw, payload.get("text", "")) if keyword_weight > 0 else 0.0
            base = (1.0 - keyword_weight) * sim + keyword_weight * kw
            # Graph-reached chunks are discounted by distance from the seeds, so
            # a weakly related paper cannot outrank a direct hit on similarity
            # alone. graph_weight controls how much the graph can promote.
            boost = (1.0 - graph_weight) + graph_weight * relatedness
            scored.append((base * boost, _to_chunk(payload, base * boost), relatedness))

    scored.sort(key=lambda row: row[0], reverse=True)

    out: List[RetrievedChunk] = []
    seen = set()
    for score, chunk, _ in scored:
        key = (chunk["source_id"], chunk["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        chunk["score"] = round(float(score), 4)
        out.append(chunk)
        if len(out) >= top_k:
            break
    return out


def _to_chunk(payload: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=payload.get("text", ""),
        source_id=payload.get("source_id", ""),
        source_title=payload.get("source_title", ""),
        chunk_index=payload.get("chunk_index", -1),
        score=round(float(score), 4),
    )


_GRAPH: Optional[KnowledgeGraph] = None


def get_graph(path: Optional[str] = None) -> KnowledgeGraph:
    """Lazily load the graph once per process — it is a few hundred KB of JSON."""
    global _GRAPH
    if _GRAPH is None:
        from src.graph.knowledge_graph import DEFAULT_GRAPH_PATH

        _GRAPH = KnowledgeGraph.load(path or DEFAULT_GRAPH_PATH)
    return _GRAPH
