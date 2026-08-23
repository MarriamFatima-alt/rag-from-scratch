"""
Embedding model — built from scratch with plain NumPy.

Implements TF-IDF (term frequency - inverse document frequency) vectorization
by hand so the retrieval step does not depend on downloading a pretrained
embedding model or calling an external embeddings API. This keeps the whole
pipeline runnable offline, which is also what makes it a genuine "from
scratch" implementation rather than a thin wrapper around someone else's
embedding service.

Swap in a real sentence-transformer or an API embedding model later by
implementing the same `fit` / `embed` interface — the rest of the pipeline
(vector store, retriever) does not care how the vectors were produced.
"""

import re
import math
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "to", "from", "of",
    "and", "or", "but", "if", "then", "so", "than", "that", "this",
    "these", "those", "it", "its", "as", "not", "no", "do", "does", "did",
    "has", "have", "had", "can", "could", "will", "would", "should",
}


def tokenize(text: str) -> list[str]:
    return [
        tok.lower()
        for tok in _TOKEN_RE.findall(text)
        if tok.lower() not in _STOPWORDS
    ]


class TfidfEmbedder:
    """A minimal, dependency-free TF-IDF vectorizer."""

    def __init__(self):
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        doc_freq: Counter = Counter()
        vocab: dict[str, int] = {}

        for doc in corpus:
            tokens = set(tokenize(doc))
            for tok in tokens:
                doc_freq[tok] += 1
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        n_docs = max(len(corpus), 1)
        idf = np.zeros(len(vocab), dtype=np.float64)
        for tok, idx in vocab.items():
            # smoothed idf, same convention as scikit-learn's default
            idf[idx] = math.log((1 + n_docs) / (1 + doc_freq[tok])) + 1

        self.vocabulary_ = vocab
        self.idf_ = idf
        self._fitted = True
        return self

    def _term_frequency_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocabulary_), dtype=np.float64)
        tokens = tokenize(text)
        if not tokens:
            return vec
        counts = Counter(tokens)
        max_count = max(counts.values())
        for tok, count in counts.items():
            idx = self.vocabulary_.get(tok)
            if idx is not None:
                vec[idx] = count / max_count  # normalized term frequency
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit(corpus) before .embed(texts)")
        vectors = np.vstack([self._term_frequency_vector(t) for t in texts])
        tfidf = vectors * self.idf_
        # L2-normalize so cosine similarity reduces to a dot product
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return tfidf / norms

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
