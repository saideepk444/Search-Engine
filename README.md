# Search Engine

A hybrid search engine built from scratch in Python. Combines three retrieval signals — BM25 keyword matching, PageRank link-graph scoring, and FAISS semantic vector search — into a single ranked result set. Backed by SQLite and served via FastAPI.

---

## Architecture

```
Query
  │
  ├─► BM25 (inverted index, SQLite)       ─┐
  ├─► PageRank (link graph, SQLite)        ├─► Hybrid ranker → ranked results
  └─► Semantic (sentence-transformers + FAISS) ─┘
```

### Three-tier retrieval pipeline

| Tier | Signal | Implementation |
|------|--------|----------------|
| Keyword | BM25 (k1=1.5, b=0.75) | Inverted index in SQLite; `indexer/bm25.py` |
| Graph | PageRank | Iterative power method on link adjacency matrix; `indexer/pagerank.py` |
| Semantic | Cosine similarity | `all-MiniLM-L6-v2` embeddings in a FAISS `IndexFlatIP`; `embedder/embedder.py` |

Final score:
```
score = 0.5 · bm25 + 0.2 · pagerank + 0.3 · semantic
```
All three signals are min-max normalized to [0, 1] before blending. Weights are tunable in `config.py`.

### Module map

```
config.py               ← single source of truth (paths, weights, model name, seed URLs)
storage/db.py           ← all SQLite schema creation and query helpers
crawler/crawler.py      ← BFS crawler (requests + BeautifulSoup), robots.txt aware
indexer/
  tokenizer.py          ← lowercase, regex tokenize, stopword removal
  inverted_index.py     ← builds word → [(doc_id, tf, positions)], writes to SQLite
  bm25.py               ← BM25 scorer over the inverted index
  pagerank.py           ← power-method PageRank on the crawled link graph
embedder/embedder.py    ← encodes pages with sentence-transformers, builds/queries FAISS index
searcher/hybrid.py      ← merges all three signals, returns ranked Result objects
api/app.py              ← FastAPI: GET /search, GET /health, GET / (HTML UI)
scripts/                ← CLI entry points for each pipeline stage
```

### Storage

All persistent state lives in `data/` (created on first run, git-ignored):

| File | Contents |
|------|----------|
| `data/search.db` | SQLite: `pages`, `index_entries`, `doc_stats`, `pagerank` tables |
| `data/faiss.index` | FAISS flat inner-product index |
| `data/faiss_ids.npy` | NumPy array mapping FAISS position → doc_id |

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended.

---

## Running the Pipeline

Run these four steps in order. Each is idempotent — re-running overwrites the previous output.

### 1. Crawl

```bash
python scripts/run_crawl.py
```

BFS-crawls the seed URLs defined in `config.SEED_URLS` (defaults to a few Wikipedia articles on information retrieval). Pages are stored in `data/search.db`. Depth and page count caps are in `config.py`.

### 2. Index

```bash
python scripts/run_index.py
```

Builds the inverted index and per-doc BM25 stats, then computes PageRank over the crawled link graph. Writes results to SQLite.

### 3. Embed

```bash
python scripts/run_embed.py
```

Downloads `all-MiniLM-L6-v2` on first run (~90 MB), encodes all pages, writes `data/faiss.index` and `data/faiss_ids.npy`.

### 4. Serve

```bash
python scripts/run_server.py
# or with auto-reload for development:
uvicorn api.app:app --reload --port 8000
```

Opens at [http://localhost:8000](http://localhost:8000).

---

## API

### `GET /search`

```
GET /search?q=<query>&top_k=10
```

Returns ranked results with per-signal scores:

```json
{
  "query": "machine learning",
  "total": 10,
  "results": [
    {
      "url": "https://en.wikipedia.org/wiki/Machine_learning",
      "title": "Machine learning - Wikipedia",
      "snippet": "Machine learning (ML) is a field of study in artificial intelligence...",
      "bm25_score": 4.231,
      "pagerank_score": 0.00821,
      "semantic_score": 0.873,
      "final_score": 0.741
    }
  ]
}
```

### `GET /health`

```json
{ "status": "ok", "doc_count": 183, "faiss_available": true }
```

### `GET /`

Minimal HTML search UI — no build step required.

---

## Configuration

All tunables live in `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEED_URLS` | 3 Wikipedia pages | Starting URLs for the crawler |
| `CRAWL_DEPTH` | `2` | BFS depth from seed |
| `MAX_PAGES` | `200` | Hard cap on pages crawled |
| `CRAWL_DELAY` | `0.5s` | Polite delay between requests |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `BM25_K1` | `1.5` | BM25 term saturation |
| `BM25_B` | `0.75` | BM25 length normalization |
| `PAGERANK_DAMPING` | `0.85` | PageRank damping factor |
| `WEIGHT_BM25` | `0.5` | Hybrid blend weight |
| `WEIGHT_PAGERANK` | `0.2` | Hybrid blend weight |
| `WEIGHT_SEMANTIC` | `0.3` | Hybrid blend weight |

---

## Why This Stack

- **BM25 over TF-IDF** — industry standard for keyword retrieval; accounts for document length and term saturation.
- **PageRank** — boosts authoritative pages naturally linked to by many crawled documents. Computed once offline, costs nothing at query time.
- **FAISS `IndexFlatIP`** — exact cosine similarity (via L2-normalized inner product) in a flat index. No approximation error; fast enough at <1M vectors.
- **SQLite** — zero-config persistence. `journal_mode=WAL` keeps reads non-blocking while indexing writes.
- **sentence-transformers `all-MiniLM-L6-v2`** — 22M parameters, 384-dim embeddings, ~14k sentences/sec on CPU. Good quality/speed tradeoff for a local project.
