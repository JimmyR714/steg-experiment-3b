"""Text chunking utilities for memory ingestion."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping chunks by character count.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_by_separator(text: str, sep: str = "\n\n") -> list[str]:
    """Split *text* on a separator and return non-empty chunks.

    Args:
        text: The text to split.
        sep: The separator string.

    Returns:
        A list of non-empty text chunks, stripped of leading/trailing whitespace.
    """
    return [chunk.strip() for chunk in text.split(sep) if chunk.strip()]
