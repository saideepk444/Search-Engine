from dataclasses import dataclass

import config
from indexer import bm25
from storage import db


@dataclass
class Result:
    doc_id: int
    url: str
    title: str
    snippet: str
    bm25_score: float
    pagerank_score: float
    semantic_score: float
    final_score: float


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def search(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    semantic_searcher=None,
) -> list[Result]:
    # --- BM25 ---
    bm25_raw = {doc_id: s for doc_id, s in bm25.search(query, top_k=50)}

    # --- PageRank ---
    pr_all = db.get_pagerank_scores()

    # --- Semantic ---
    sem_raw: dict[int, float] = {}
    if semantic_searcher is not None:
        for doc_id, s in semantic_searcher.search(query, top_k=50):
            sem_raw[doc_id] = s

    candidate_ids = set(bm25_raw) | set(sem_raw)
    if not candidate_ids:
        return []

    bm25_norm = _normalize({k: bm25_raw.get(k, 0.0) for k in candidate_ids})
    pr_norm = _normalize({k: pr_all.get(k, 0.0) for k in candidate_ids})
    sem_norm = _normalize({k: sem_raw.get(k, 0.0) for k in candidate_ids})

    final: dict[int, float] = {
        doc_id: (
            config.WEIGHT_BM25 * bm25_norm[doc_id]
            + config.WEIGHT_PAGERANK * pr_norm[doc_id]
            + config.WEIGHT_SEMANTIC * sem_norm[doc_id]
        )
        for doc_id in candidate_ids
    }

    top_ids = sorted(final, key=lambda x: final[x], reverse=True)[:top_k]

    pages = {p["id"]: p for p in db.get_all_pages() if p["id"] in set(top_ids)}

    results = []
    for doc_id in top_ids:
        page = pages.get(doc_id)
        if page is None:
            continue
        body = page["body"] or ""
        snippet = body[:200].rsplit(" ", 1)[0] + "…" if len(body) > 200 else body
        results.append(Result(
            doc_id=doc_id,
            url=page["url"],
            title=page["title"] or page["url"],
            snippet=snippet,
            bm25_score=bm25_raw.get(doc_id, 0.0),
            pagerank_score=pr_all.get(doc_id, 0.0),
            semantic_score=sem_raw.get(doc_id, 0.0),
            final_score=final[doc_id],
        ))

    return results
