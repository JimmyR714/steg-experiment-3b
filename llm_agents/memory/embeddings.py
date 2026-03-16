"""Embedding providers for memory retrieval."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Protocol for text embedding providers."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a 2-D array of shape (len(texts), dim).

        Args:
            texts: The texts to embed.

        Returns:
            A numpy array of shape ``(len(texts), embedding_dim)``.
        """
        ...


class HFEmbedder:
    """Embedding provider using HuggingFace sentence-transformers.

    Args:
        model_name: Name of a sentence-transformers model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: object | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for HFEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using a local sentence-transformers model."""
        self._load()
        return self._model.encode(texts, convert_to_numpy=True)  # type: ignore[union-attr]


class OpenAIEmbedder:
    """Embedding provider using the OpenAI embeddings API.

    Args:
        model: The OpenAI embedding model name.
        api_key: Optional API key (defaults to ``OPENAI_API_KEY`` env var).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIEmbedder."
            ) from exc
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        self._client = openai.OpenAI(**kwargs)
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the OpenAI embeddings API."""
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=texts)  # type: ignore[union-attr]
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
