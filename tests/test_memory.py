"""Tests for Extension 1: Agent Memory & Retrieval (RAG)."""

from __future__ import annotations

import numpy as np
import pytest

from llm_agents.memory.chunker import chunk_by_separator, chunk_text
from llm_agents.memory.store import InMemoryStore, MemoryRecord, PersistentStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic embedder that hashes text into a fixed-size vector."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            np.random.seed(hash(t) % (2**31))
            result[i] = np.random.randn(self.dim).astype(np.float32)
            result[i] /= np.linalg.norm(result[i]) + 1e-10
        return result


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_basic_chunking(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        assert len(chunks) == 5
        assert all(len(c) <= 30 for c in chunks)

    def test_overlap(self):
        text = "abcdefghij" * 5  # 50 chars
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        # Second chunk should start at position 15
        assert chunks[1][:5] == text[15:20]

    def test_no_overlap(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=25, overlap=0)
        assert len(chunks) == 4

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("text", chunk_size=0)

    def test_overlap_ge_chunk_size(self):
        with pytest.raises(ValueError, match="overlap must be less than"):
            chunk_text("text", chunk_size=10, overlap=10)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            chunk_text("text", chunk_size=10, overlap=-1)


class TestChunkBySeparator:
    def test_paragraph_split(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        chunks = chunk_by_separator(text)
        assert len(chunks) == 3
        assert chunks[0] == "First paragraph."
        assert chunks[2] == "Third."

    def test_custom_separator(self):
        text = "a---b---c"
        chunks = chunk_by_separator(text, sep="---")
        assert chunks == ["a", "b", "c"]

    def test_empty_chunks_filtered(self):
        text = "a\n\n\n\nb"
        chunks = chunk_by_separator(text, sep="\n\n")
        assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# InMemoryStore tests
# ---------------------------------------------------------------------------


class TestInMemoryStore:
    def test_add_and_len(self):
        store = InMemoryStore(FakeEmbedder())
        assert len(store) == 0
        store.add("Hello world")
        assert len(store) == 1
        store.add("Another fact", {"source": "test"})
        assert len(store) == 2

    def test_search_returns_results(self):
        store = InMemoryStore(FakeEmbedder())
        store.add("Python is a programming language.")
        store.add("The sky is blue.")
        store.add("Python was created by Guido.")
        results = store.search("Python programming", k=2)
        assert len(results) == 2
        assert all(isinstance(r, MemoryRecord) for r in results)

    def test_search_empty_store(self):
        store = InMemoryStore(FakeEmbedder())
        results = store.search("anything")
        assert results == []

    def test_search_k_exceeds_size(self):
        store = InMemoryStore(FakeEmbedder())
        store.add("Only entry")
        results = store.search("query", k=10)
        assert len(results) == 1

    def test_clear(self):
        store = InMemoryStore(FakeEmbedder())
        store.add("Entry 1")
        store.add("Entry 2")
        store.clear()
        assert len(store) == 0
        assert store.search("Entry") == []

    def test_metadata_preserved(self):
        store = InMemoryStore(FakeEmbedder())
        store.add("Data", {"key": "value"})
        results = store.search("Data", k=1)
        assert results[0].metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# PersistentStore tests
# ---------------------------------------------------------------------------


class TestPersistentStore:
    def test_add_and_len(self):
        store = PersistentStore(FakeEmbedder(), db_path=":memory:")
        assert len(store) == 0
        store.add("Test entry")
        assert len(store) == 1

    def test_search_returns_results(self):
        store = PersistentStore(FakeEmbedder(), db_path=":memory:")
        store.add("The Earth orbits the Sun.")
        store.add("Water is H2O.")
        results = store.search("planets", k=2)
        assert len(results) == 2

    def test_clear(self):
        store = PersistentStore(FakeEmbedder(), db_path=":memory:")
        store.add("Entry")
        store.clear()
        assert len(store) == 0

    def test_search_empty(self):
        store = PersistentStore(FakeEmbedder(), db_path=":memory:")
        assert store.search("query") == []
