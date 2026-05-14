import json

import numpy as np

import config
from storage import db


def compute(verbose: bool = True) -> dict[int, float]:
    pages = db.get_all_pages()
    if not pages:
        return {}

    ids = [p["id"] for p in pages]
    idx = {doc_id: i for i, doc_id in enumerate(ids)}
    url_to_id = {p["url"]: p["id"] for p in pages}
    n = len(ids)

    # Build column-stochastic transition matrix
    M = np.zeros((n, n), dtype=np.float64)
    for page in pages:
        links = json.loads(page["links_json"] or "[]")
        src = idx[page["id"]]
        targets = [idx[url_to_id[u]] for u in links if u in url_to_id]
        if targets:
            for t in targets:
                M[t, src] += 1.0
            M[:, src] /= len(targets)
        else:
            # dangling node: distribute uniformly
            M[:, src] = 1.0 / n

    d = config.PAGERANK_DAMPING
    teleport = np.full((n, n), (1 - d) / n)
    A = d * M + teleport

    ranks = np.full(n, 1.0 / n)
    for iteration in range(config.PAGERANK_MAX_ITER):
        new_ranks = A @ ranks
        delta = np.linalg.norm(new_ranks - ranks, 1)
        ranks = new_ranks
        if delta < config.PAGERANK_TOL:
            if verbose:
                print(f"PageRank converged in {iteration + 1} iterations (delta={delta:.2e})")
            break

    scores = {ids[i]: float(ranks[i]) for i in range(n)}
    db.upsert_pagerank(scores)

    if verbose:
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top-5 PageRank doc_ids: {top}")

    return scores
