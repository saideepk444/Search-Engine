# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

All persistent state (SQLite DB, FAISS index files) is written to `data/`, which is created automatically on first run.

## Pipeline (run in order)

```bash
python scripts/run_crawl.py    # crawl seed URLs → data/search.db
python scripts/run_index.py    # build inverted index + BM25 stats + PageRank → search.db
python scripts/run_embed.py    # encode docs → data/faiss.index + data/faiss_ids.npy
python scripts/run_server.py   # start API at http://localhost:8000
```

## Dev server (auto-reload)

```bash
uvicorn api.app:app --reload --port 8000
```

## Architecture

Three-tier hybrid retrieval pipeline:

1. **Keyword tier** — BM25 over an inverted index. `indexer/inverted_index.py` builds `word → [(doc_id, tf, positions)]` from crawled pages and serializes to SQLite. `indexer/bm25.py` scores queries against it (k1=1.5, b=0.75).

2. **Graph tier** — PageRank on the crawled link graph. `indexer/pagerank.py` runs iterative power method on the adjacency matrix derived from `pages.links_json` and stores per-doc scores in SQLite.

3. **Semantic tier** — `sentence-transformers/all-MiniLM-L6-v2` encodes each page; embeddings are stored in a FAISS `IndexFlatIP` (cosine similarity via L2-normalized vectors). `embedder/embedder.py` handles both build and query.

Final scores are merged in `searcher/hybrid.py`:
```
final = α·bm25 + β·pagerank + γ·semantic
```
All three signals are min-max normalized to [0,1] before blending. Weights α/β/γ live in `config.py`.

**Single source of truth:** `config.py` controls seed URLs, crawl depth, DB path, FAISS index paths, embedding model name, and ranking weights.

**Storage:** `storage/db.py` owns all SQLite schema creation and query helpers. Tables: `pages`, `index_entries`, `doc_stats`, `pagerank`.
