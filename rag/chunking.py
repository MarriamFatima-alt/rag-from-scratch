"""
Document chunking — built from scratch.

Splits raw text into overlapping, sentence-aware chunks so that each chunk
stays coherent (no cutting mid-sentence) while still giving the retriever
enough context per chunk.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_into_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter — no external NLP model required."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Greedily packs sentences into chunks of ~chunk_size characters, carrying
    the last `chunk_overlap` characters of context into the next chunk so
    that ideas that span a chunk boundary aren't lost to the retriever.
    """
    sentences = split_into_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    chunk_id = 0

    def flush():
        nonlocal current, current_len, chunk_id
        if not current:
            return
        chunk_text_value = " ".join(current).strip()
        if chunk_text_value:
            chunks.append(
                Chunk(text=chunk_text_value, source=source, chunk_id=chunk_id)
            )
            chunk_id += 1
        current = []
        current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            flush()
            # carry overlap from the end of the previous chunk
            overlap_text = chunks[-1].text[-chunk_overlap:] if chunks else ""
            if overlap_text:
                current = [overlap_text]
                current_len = len(overlap_text)
        current.append(sentence)
        current_len += len(sentence) + 1

    flush()
    return chunks


def chunk_documents(
    documents: list[tuple[str, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """documents: list of (source_name, raw_text) tuples."""
    all_chunks: list[Chunk] = []
    for source, text in documents:
        all_chunks.extend(chunk_text(text, source, chunk_size, chunk_overlap))
    return all_chunks
