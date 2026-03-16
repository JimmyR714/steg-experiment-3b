"""Abstract base class for LLM model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_agents.models.types import CompletionResult, LogProbResult


class BaseModel(ABC):
    """Abstract interface for loading LLM models and obtaining completions
    with log-probability information."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input text to complete.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Number of top tokens to consider during sampling.
            stop: Optional list of stop sequences.

        Returns:
            A CompletionResult containing the generated text and optional
            log-probability data.
        """

    @abstractmethod
    def get_logprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        top_k: int = 5,
    ) -> LogProbResult:
        """Generate a completion and return detailed log-probability info.

        Args:
            prompt: The input text to complete.
            max_tokens: Maximum number of tokens to generate.
            top_k: Number of top alternative tokens to return per position.

        Returns:
            A LogProbResult with per-token log-probabilities and top-k
            alternatives at each position.
        """
