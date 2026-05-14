import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Crawler
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Information_retrieval",
    "https://en.wikipedia.org/wiki/Search_engine",
    "https://en.wikipedia.org/wiki/PageRank",
]
CRAWL_DEPTH = 2
MAX_PAGES = 200
CRAWL_DELAY = 0.5  # seconds between requests

# Storage
DB_PATH = os.path.join(DATA_DIR, "search.db")

# Embedder
EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_IDS_PATH = os.path.join(DATA_DIR, "faiss_ids.npy")
EMBED_MAX_CHARS = 512

# BM25
BM25_K1 = 1.5
BM25_B = 0.75

# PageRank
PAGERANK_DAMPING = 0.85
PAGERANK_TOL = 1e-6
PAGERANK_MAX_ITER = 100

# Hybrid ranker weights (must sum to 1.0)
WEIGHT_BM25 = 0.5
WEIGHT_PAGERANK = 0.2
WEIGHT_SEMANTIC = 0.3

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
DEFAULT_TOP_K = 10
