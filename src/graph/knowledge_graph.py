"""
A knowledge graph over the paper corpus, built from metadata alone.

Purpose: retrieval by embedding similarity finds chunks that *read* like the
query. It has no notion that two papers share an author, sit in the same
arXiv category, or argue about the same concept. This graph adds those
relations so retrieval can expand from a hit to its neighbourhood — the
"related work you'd have found by following citations" that pure vector search
structurally cannot reach.

Node types: Paper, Author, Category, Concept
Edge types: AUTHORED_BY, IN_CATEGORY, MENTIONS

Design notes worth defending in review:

* **Concepts are extracted with TF-IDF, not an LLM.** Deterministic, free, and
  reproducible — an LLM keyphrase pass would make graph construction depend on
  a rate-limited API and change between runs. The cost is that concepts are
  surface terms, not normalised entities.
* **Co-authorship is kept but is nearly useless here.** Across 40 topically
  scoped papers there are 165 authors and only 4 appear on more than one. The
  edge type is implemented because the graph should not silently omit a
  relation it models; the honest statement is that categories and concepts
  carry essentially all the connectivity on a corpus this size.
* Paper-to-paper relatedness is derived, not stored: two papers are related in
  proportion to the weighted overlap of their shared neighbours.
"""
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "that",
    "this", "is", "are", "was", "were", "be", "been", "by", "as", "at", "from",
    "it", "its", "we", "our", "us", "they", "their", "can", "may", "such", "these",
    "those", "than", "then", "but", "not", "have", "has", "had", "which", "when",
    "while", "also", "more", "most", "other", "using", "used", "use", "based",
    "show", "shows", "propose", "proposed", "paper", "method", "methods",
    "approach", "results", "however", "both", "each", "all", "new", "via",
}

# Relation weights when scoring paper-to-paper relatedness. A shared concept is
# the strongest signal: categories are coarse (cs.IR covers 29 of 40 papers
# here, so it barely discriminates), while a shared author is rare but precise.
EDGE_WEIGHTS = {"MENTIONS": 1.0, "AUTHORED_BY": 0.8, "IN_CATEGORY": 0.25}

CONCEPTS_PER_PAPER = 12
DEFAULT_GRAPH_PATH = "data/processed/knowledge_graph.json"


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _extract_concepts(docs: Dict[str, str], top_n: int = CONCEPTS_PER_PAPER) -> Dict[str, List[str]]:
    """
    Classic TF-IDF keyphrase extraction over unigrams and bigrams.

    Bigrams matter here: "retrieval augmented" and "knowledge graph" are the
    concepts that actually link papers, whereas the unigrams "retrieval" and
    "knowledge" appear nearly everywhere in this corpus and link nothing.
    """
    tokenized = {pid: _tokenize(text) for pid, text in docs.items()}
    grams: Dict[str, Counter] = {}
    for pid, words in tokenized.items():
        counter = Counter(words)
        counter.update(f"{a} {b}" for a, b in zip(words, words[1:]))
        grams[pid] = counter

    doc_freq: Counter = Counter()
    for counter in grams.values():
        doc_freq.update(counter.keys())

    n_docs = max(len(docs), 1)
    concepts: Dict[str, List[str]] = {}
    for pid, counter in grams.items():
        total = sum(counter.values()) or 1
        scored = []
        for term, count in counter.items():
            df = doc_freq[term]
            if df < 2 or df > n_docs * 0.5:
                # Appears in only one paper (links nothing) or over half of them
                # (links everything) — neither is a useful edge.
                continue
            tf = count / total
            idf = math.log(n_docs / df)
            scored.append((tf * idf, term))
        scored.sort(reverse=True)
        concepts[pid] = [term for _, term in scored[:top_n]]
    return concepts


@dataclass
class KnowledgeGraph:
    # node id -> node type ("paper" | "author" | "category" | "concept")
    nodes: Dict[str, str] = field(default_factory=dict)
    # (src, relation) -> set of destinations, plus the reverse index
    edges: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    incoming: Dict[str, List[Tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))
    paper_titles: Dict[str, str] = field(default_factory=dict)

    def add_edge(self, src: str, relation: str, dst: str) -> None:
        if dst not in self.edges[src][relation]:
            self.edges[src][relation].append(dst)
        self.incoming[dst].append((src, relation))

    def neighbours(self, paper_id: str) -> Dict[str, float]:
        """
        Papers sharing a neighbour with `paper_id`, scored by relation weight.

        Two hops: paper -> (author | category | concept) -> paper. Each shared
        neighbour contributes its relation's weight, divided by the log of that
        neighbour's degree so a category shared by 29 papers counts for far
        less than a concept shared by 3.
        """
        scores: Dict[str, float] = defaultdict(float)
        for relation, targets in self.edges.get(paper_id, {}).items():
            weight = EDGE_WEIGHTS.get(relation, 0.5)
            for target in targets:
                siblings = [s for s, r in self.incoming.get(target, []) if r == relation]
                if len(siblings) <= 1:
                    continue
                discount = 1.0 / math.log(len(siblings) + 1.5)
                for sibling in siblings:
                    if sibling != paper_id:
                        scores[sibling] += weight * discount
        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    def related_papers(self, paper_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        return list(self.neighbours(paper_id).items())[:limit]

    def stats(self) -> Dict[str, int]:
        by_type: Counter = Counter(self.nodes.values())
        edge_count = sum(len(dsts) for rels in self.edges.values() for dsts in rels.values())
        return {
            "nodes": len(self.nodes),
            "papers": by_type["paper"],
            "authors": by_type["author"],
            "categories": by_type["category"],
            "concepts": by_type["concept"],
            "edges": edge_count,
        }

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": {s: dict(r) for s, r in self.edges.items()},
            "paper_titles": self.paper_titles,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        graph = cls()
        graph.nodes = data["nodes"]
        graph.paper_titles = data.get("paper_titles", {})
        for src, relations in data["edges"].items():
            for relation, targets in relations.items():
                for dst in targets:
                    graph.add_edge(src, relation, dst)
        return graph

    def save(self, path: str = DEFAULT_GRAPH_PATH) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str = DEFAULT_GRAPH_PATH) -> "KnowledgeGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))


def build_graph(papers: List[dict]) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    docs = {p["arxiv_id"]: f"{p.get('title','')} {p.get('abstract','')}" for p in papers}
    concepts = _extract_concepts(docs)

    for paper in papers:
        pid = paper["arxiv_id"]
        graph.nodes[pid] = "paper"
        graph.paper_titles[pid] = paper.get("title", "")

        for author in paper.get("authors", []):
            node = f"author:{author}"
            graph.nodes[node] = "author"
            graph.add_edge(pid, "AUTHORED_BY", node)

        for category in paper.get("categories", []):
            node = f"category:{category}"
            graph.nodes[node] = "category"
            graph.add_edge(pid, "IN_CATEGORY", node)

        for concept in concepts.get(pid, []):
            node = f"concept:{concept}"
            graph.nodes[node] = "concept"
            graph.add_edge(pid, "MENTIONS", node)

    return graph


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the knowledge graph from paper metadata")
    parser.add_argument("--meta", default="data/raw/papers_meta.json")
    parser.add_argument("--out", default=DEFAULT_GRAPH_PATH)
    args = parser.parse_args()

    papers = json.loads(Path(args.meta).read_text())
    graph = build_graph(papers)
    path = graph.save(args.out)

    print("Knowledge graph built:")
    for key, value in graph.stats().items():
        print(f"  {key}: {value}")

    sample = papers[0]["arxiv_id"]
    print(f"\nRelated to {sample} ({graph.paper_titles[sample][:60]}...):")
    for pid, score in graph.related_papers(sample, limit=5):
        print(f"  {score:.3f}  {pid}  {graph.paper_titles.get(pid,'')[:60]}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
