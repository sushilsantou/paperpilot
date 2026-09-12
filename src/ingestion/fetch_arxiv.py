"""
Pulls papers from arXiv for a given search query and downloads their PDFs.
Uses the official `arxiv` package, which wraps the public arXiv API —
no API key required.
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List
import json
import time
import urllib.request

import arxiv
from tqdm import tqdm

# arxiv 4.x dropped Result.download_pdf (it existed in 2.x), so fetch the PDF
# over plain HTTP from result.pdf_url instead. That URL is stable across every
# version of the package, which keeps this working regardless of which one is
# installed.
_PDF_TIMEOUT_SEC = 60
_INTER_DOWNLOAD_DELAY_SEC = 1.0  # arXiv asks callers not to hammer the endpoint


def _download_pdf(pdf_url: str, dest: Path) -> None:
    request = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": "PaperPilot/0.1 (+https://github.com/sushilsantou/paperpilot)"},
    )
    with urllib.request.urlopen(request, timeout=_PDF_TIMEOUT_SEC) as response:
        dest.write_bytes(response.read())

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
    # Captured for the knowledge graph (src/graph/): arXiv categories are the
    # densest real edge available here. Co-authorship is far too sparse on a
    # topic-scoped corpus to connect anything — of 165 authors across 40
    # papers, only 4 appear on more than one.
    primary_category: str = ""
    categories: List[str] = field(default_factory=list)


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
                _download_pdf(result.pdf_url, pdf_path)
                time.sleep(_INTER_DOWNLOAD_DELAY_SEC)
        except Exception as e:  # noqa: BLE001 — log and skip, don't kill the whole ingest run
            print(f"  [skip] {arxiv_id}: download failed ({e})")
            pdf_path.unlink(missing_ok=True)  # don't leave a truncated file for the next run to trust
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
                primary_category=getattr(result.primary_category, "term", None) or str(result.primary_category or ""),
                categories=[getattr(c, "term", None) or str(c) for c in (result.categories or [])],
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
