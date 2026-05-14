import os

import numpy as np

import config
from storage import db


def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def build(verbose: bool = True) -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    pages = db.get_all_pages()
    if not pages:
        print("No pages in DB. Run the crawler first.")
        return

    model = _load_model()

    doc_ids = [p["id"] for p in pages]
    texts = [
        (p["title"] or "") + " " + (p["body"] or "")[:config.EMBED_MAX_CHARS]
        for p in pages
    ]

    if verbose:
        print(f"Encoding {len(texts)} documents with {config.EMBED_MODEL}...")

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=verbose, normalize_embeddings=True)

    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, config.FAISS_INDEX_PATH)
    np.save(config.FAISS_IDS_PATH, np.array(doc_ids, dtype=np.int64))

    if verbose:
        print(f"FAISS index saved ({index.ntotal} vectors, dim={dim})")


class SemanticSearcher:
    def __init__(self) -> None:
        import faiss
        self._index = faiss.read_index(config.FAISS_INDEX_PATH)
        self._ids = np.load(config.FAISS_IDS_PATH)
        self._model = _load_model()

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        vec = self._model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, positions = self._index.search(vec, top_k)
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:
                continue
            results.append((int(self._ids[pos]), float(score)))
        return results
