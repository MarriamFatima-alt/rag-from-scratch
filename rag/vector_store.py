"""
Vector store — built from scratch, with an optional FAISS-backed mode.

Default backend: a plain NumPy matrix + cosine-similarity search. This is
what "from scratch" means here — no external vector database service, no
network call, just linear algebra.

If `faiss` is installed, `VectorStore(backend="faiss")` swaps in a FAISS
IndexFlatIP for the same interface, which is useful once the corpus grows
past what a brute-force NumPy scan handles comfortably.
"""

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, backend: str = "numpy"):
        self.backend = backend
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._faiss_index = None

        if backend == "faiss":
            try:
                import faiss  # noqa: F401
                self._faiss = faiss
            except ImportError as exc:
                raise ImportError(
                    "backend='faiss' requires `pip install faiss-cpu`"
                ) from exc

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._chunks = chunks
        self._matrix = vectors.astype(np.float32)

        if self.backend == "faiss":
            dim = self._matrix.shape[1]
            index = self._faiss.IndexFlatIP(dim)  # inner product == cosine on normalized vectors
            index.add(self._matrix)
            self._faiss_index = index

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[RetrievedChunk]:
        if self._matrix is None or len(self._chunks) == 0:
            return []

        top_k = min(top_k, len(self._chunks))
        query_vector = query_vector.astype(np.float32)

        if self.backend == "faiss":
            scores, idxs = self._faiss_index.search(query_vector.reshape(1, -1), top_k)
            pairs = zip(idxs[0], scores[0])
        else:
            # cosine similarity == dot product, vectors are already L2-normalized
            scores = self._matrix @ query_vector
            top_idxs = np.argsort(-scores)[:top_k]
            pairs = zip(top_idxs, scores[top_idxs])

        results = [
            RetrievedChunk(chunk=self._chunks[idx], score=float(score))
            for idx, score in pairs
            if idx != -1
        ]
        return sorted(results, key=lambda r: r.score, reverse=True)

    def __len__(self) -> int:
        return len(self._chunks)
