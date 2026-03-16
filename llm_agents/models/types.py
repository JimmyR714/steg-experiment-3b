"""Data types for model completions and log-probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenLogProb:
    """A single token with its log-probability and rank among alternatives."""

    token: str
    logprob: float
    rank: int


@dataclass(frozen=True)
class LogProbResult:
    """Log-probability information for a completed sequence.

    Attributes:
        prompt: The input prompt that produced this result.
        tokens: Ordered list of chosen tokens with their log-probs.
        top_k_per_position: For each token position, the top-k alternative
            tokens and their log-probs.
    """

    prompt: str
    tokens: list[TokenLogProb] = field(default_factory=list)
    top_k_per_position: list[list[TokenLogProb]] = field(default_factory=list)


@dataclass(frozen=True)
class CompletionResult:
    """Result of a model generation call.

    Attributes:
        text: The generated text.
        logprob_result: Optional log-probability details for the generation.
        finish_reason: Why generation stopped (e.g. "stop", "length").
    """

    text: str
    logprob_result: LogProbResult | None = None
    finish_reason: str = "stop"
