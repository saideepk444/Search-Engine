import json
from collections import defaultdict

from indexer.tokenizer import tokenize
from storage import db


def build(verbose: bool = True) -> None:
    pages = db.get_all_pages()
    if not pages:
        print("No pages in DB. Run the crawler first.")
        return

    # token → {doc_id: [positions]}
    postings: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    doc_lengths: dict[int, int] = {}

    for page in pages:
        doc_id = page["id"]
        tokens = tokenize((page["title"] or "") + " " + (page["body"] or ""))
        doc_lengths[doc_id] = len(tokens)
        for pos, token in enumerate(tokens):
            postings[token][doc_id].append(pos)

    total_tokens = sum(doc_lengths.values())
    avg_dl = total_tokens / len(doc_lengths) if doc_lengths else 1.0

    entries: list[tuple] = []
    for token, doc_map in postings.items():
        for doc_id, positions in doc_map.items():
            tf = len(positions) / doc_lengths[doc_id]
            entries.append((token, doc_id, tf, json.dumps(positions)))

    db.upsert_index_entries(entries)
    db.upsert_doc_stats(list(doc_lengths.items()), avg_dl)

    if verbose:
        print(f"Indexed {len(pages)} docs, {len(postings)} unique tokens, avg_dl={avg_dl:.1f}")
