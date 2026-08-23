"""
RAGPipeline — ties chunking -> embedding -> retrieval -> generation together.

This is the "retrieval chain": index documents once, then answer any number
of questions against that index, retrieving fresh context for each one
before the LLM ever sees the question.
"""

from dataclasses import dataclass

from .chunking import chunk_documents, Chunk
from .embeddings import TfidfEmbedder
from .vector_store import VectorStore, RetrievedChunk
from .llm import generate_answer


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, backend: str = "numpy"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = TfidfEmbedder()
        self.store = VectorStore(backend=backend)
        self._indexed = False

    def index_documents(self, documents: list[tuple[str, str]]) -> int:
        """documents: list of (source_name, raw_text). Returns #chunks indexed."""
        chunks: list[Chunk] = chunk_documents(
            documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        if not chunks:
            raise ValueError("No text extracted from the provided documents.")

        corpus = [c.text for c in chunks]
        self.embedder.fit(corpus)
        vectors = self.embedder.embed(corpus)
        self.store.build(chunks, vectors)
        self._indexed = True
        return len(chunks)

    def retrieve(self, question: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not self._indexed:
            raise RuntimeError("Call index_documents(...) before retrieving.")
        query_vector = self.embedder.embed_query(question)
        return self.store.search(query_vector, top_k=top_k)

    def ask(self, question: str, top_k: int = 4) -> RAGAnswer:
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return RAGAnswer(answer="No indexed documents to search yet.", sources=[])

        context_blocks = [
            f"[Source: {r.chunk.source} | chunk {r.chunk.chunk_id} | "
            f"relevance {r.score:.2f}]\n{r.chunk.text}"
            for r in retrieved
        ]
        answer = generate_answer(question, context_blocks)
        return RAGAnswer(answer=answer, sources=retrieved)
