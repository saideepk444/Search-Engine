import math

import config
from indexer.tokenizer import tokenize
from storage import db


def score(query: str, doc_id: int) -> float:
    tokens = tokenize(query)
    if not tokens:
        return 0.0

    stats = db.get_doc_stats(doc_id)
    if stats is None:
        return 0.0

    dl = stats["token_count"]
    avg_dl = stats["avg_dl"] or 1.0
    n = db.get_doc_count()

    total = 0.0
    for token in tokens:
        postings = db.get_postings(token)
        df = len(postings)
        if df == 0:
            continue
        tf_row = next((p for p in postings if p["doc_id"] == doc_id), None)
        if tf_row is None:
            continue
        # BM25 uses raw term frequency; tf column stores tf/dl, so recover raw count
        raw_tf = tf_row["tf"] * dl
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (raw_tf * (config.BM25_K1 + 1)) / (
            raw_tf + config.BM25_K1 * (1 - config.BM25_B + config.BM25_B * dl / avg_dl)
        )
        total += idf * tf_norm

    return total


def search(query: str, top_k: int = 20) -> list[tuple[int, float]]:
    """Return [(doc_id, score)] sorted descending, candidates only (no full scan)."""
    tokens = tokenize(query)
    if not tokens:
        return []

    candidate_ids: set[int] = set()
    for token in tokens:
        for p in db.get_postings(token):
            candidate_ids.add(p["doc_id"])

    scored = [(doc_id, score(query, doc_id)) for doc_id in candidate_ids]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
