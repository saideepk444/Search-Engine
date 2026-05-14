from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
from searcher import hybrid
from storage import db


class ResultItem(BaseModel):
    url: str
    title: str
    snippet: str
    bm25_score: float
    pagerank_score: float
    semantic_score: float
    final_score: float


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[ResultItem]


class HealthResponse(BaseModel):
    status: str
    doc_count: int
    faiss_available: bool


_semantic_searcher = None


def _try_load_semantic():
    import os
    global _semantic_searcher
    if os.path.exists(config.FAISS_INDEX_PATH) and os.path.exists(config.FAISS_IDS_PATH):
        try:
            from embedder.embedder import SemanticSearcher
            _semantic_searcher = SemanticSearcher()
            print("Semantic searcher loaded.")
        except Exception as e:
            print(f"Semantic search unavailable: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load_semantic()
    yield


app = FastAPI(
    title="Hybrid Search Engine",
    description="BM25 + PageRank + FAISS semantic search",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(config.DEFAULT_TOP_K, ge=1, le=100),
):
    results = hybrid.search(q, top_k=top_k, semantic_searcher=_semantic_searcher)
    return SearchResponse(
        query=q,
        total=len(results),
        results=[
            ResultItem(
                url=r.url,
                title=r.title,
                snippet=r.snippet,
                bm25_score=round(r.bm25_score, 6),
                pagerank_score=round(r.pagerank_score, 6),
                semantic_score=round(r.semantic_score, 6),
                final_score=round(r.final_score, 6),
            )
            for r in results
        ],
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        doc_count=db.get_doc_count(),
        faiss_available=_semantic_searcher is not None,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Search Engine</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 700px; margin: 60px auto; padding: 0 1rem; }
    input { width: 80%; padding: .5rem; font-size: 1rem; }
    button { padding: .5rem 1rem; font-size: 1rem; cursor: pointer; }
    .result { margin: 1.5rem 0; }
    .result a { font-size: 1.1rem; }
    .snippet { color: #555; }
    .meta { font-size: .8rem; color: #888; }
  </style>
</head>
<body>
  <h1>Search Engine</h1>
  <form onsubmit="doSearch(event)">
    <input id="q" type="text" placeholder="Enter query..." autofocus>
    <button type="submit">Search</button>
  </form>
  <div id="results"></div>
  <script>
    async function doSearch(e) {
      e.preventDefault();
      const q = document.getElementById('q').value.trim();
      if (!q) return;
      const res = await fetch('/search?q=' + encodeURIComponent(q));
      const data = await res.json();
      const container = document.getElementById('results');
      if (!data.results.length) { container.innerHTML = '<p>No results.</p>'; return; }
      container.innerHTML = data.results.map(r => `
        <div class="result">
          <a href="${r.url}" target="_blank">${r.title}</a><br>
          <span class="snippet">${r.snippet}</span><br>
          <span class="meta">score=${r.final_score.toFixed(4)} bm25=${r.bm25_score.toFixed(3)} pr=${r.pagerank_score.toFixed(4)} sem=${r.semantic_score.toFixed(3)}</span>
        </div>`).join('');
    }
  </script>
</body>
</html>
"""
