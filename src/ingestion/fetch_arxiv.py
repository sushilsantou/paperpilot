"""
Pulls papers from arXiv for a given search query and downloads their PDFs.
Uses the official `arxiv` package, which wraps the public arXiv API —
no API key required.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import json

import arxiv
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    pdf_url: str
    pdf_path: str


def fetch_papers(query: str, max_results: int = 40) -> List[PaperMeta]:
    """
    Search arXiv and download PDFs for the top `max_results` matches.
    Returns metadata for every paper successfully downloaded (download
    failures are skipped, not fatal — arXiv occasionally rate-limits).
    """
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: List[PaperMeta] = []
    for result in tqdm(client.results(search), total=max_results, desc="Fetching arXiv papers"):
        arxiv_id = result.get_short_id()
        pdf_path = RAW_DIR / f"{arxiv_id}.pdf"
        try:
            if not pdf_path.exists():
                result.download_pdf(dirpath=str(RAW_DIR), filename=f"{arxiv_id}.pdf")
        except Exception as e:  # noqa: BLE001 — log and skip, don't kill the whole ingest run
            print(f"  [skip] {arxiv_id}: download failed ({e})")
            continue

        papers.append(
            PaperMeta(
                arxiv_id=arxiv_id,
                title=result.title.strip().replace("\n", " "),
                authors=[a.name for a in result.authors],
                abstract=result.summary.strip().replace("\n", " "),
                published=result.published.isoformat(),
                pdf_url=result.pdf_url,
                pdf_path=str(pdf_path),
            )
        )

    # Persist metadata alongside the PDFs so re-runs don't need to hit the API again
    meta_path = RAW_DIR / "papers_meta.json"
    meta_path.write_text(json.dumps([asdict(p) for p in papers], indent=2))
    return papers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch papers from arXiv")
    parser.add_argument("--query", default="retrieval augmented generation", help="arXiv search query")
    parser.add_argument("--max-results", type=int, default=40)
    args = parser.parse_args()

    fetched = fetch_papers(args.query, args.max_results)
    print(f"Fetched {len(fetched)} papers into {RAW_DIR}")
