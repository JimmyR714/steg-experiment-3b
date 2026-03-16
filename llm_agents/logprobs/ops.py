"""Utility functions for common operations on log-probabilities."""

from __future__ import annotations

import math

from llm_agents.models.types import LogProbResult, TokenLogProb


def entropy(logprobs_at_position: list[TokenLogProb]) -> float:
    """Compute Shannon entropy from a list of token log-probabilities.

    H = -sum(p * log(p)) where p = exp(logprob).

    Args:
        logprobs_at_position: Top-k tokens with log-probs at a single position.

    Returns:
        Shannon entropy in nats.
    """
    h = 0.0
    for tlp in logprobs_at_position:
        p = math.exp(tlp.logprob)
        if p > 0:
            h -= p * tlp.logprob
    return h


def perplexity(logprob_result: LogProbResult) -> float:
    """Compute sequence-level perplexity from a LogProbResult.

    PP = exp(-1/N * sum(log p(token_i)))

    Args:
        logprob_result: The log-probability result for a generated sequence.

    Returns:
        Perplexity of the sequence.

    Raises:
        ValueError: If the token list is empty.
    """
    tokens = logprob_result.tokens
    if not tokens:
        raise ValueError("Cannot compute perplexity for an empty token sequence.")
    avg_logprob = sum(t.logprob for t in tokens) / len(tokens)
    return math.exp(-avg_logprob)


def top_k_tokens(logprobs_at_position: list[TokenLogProb], k: int) -> list[TokenLogProb]:
    """Return the top-k tokens by log-probability at a given position.

    Args:
        logprobs_at_position: Tokens with log-probs at a single position.
        k: Number of top tokens to return.

    Returns:
        List of up to k TokenLogProb entries, sorted descending by logprob.
    """
    sorted_tokens = sorted(logprobs_at_position, key=lambda t: t.logprob, reverse=True)
    return sorted_tokens[:k]


def normalize(logprobs: list[float]) -> list[float]:
    """Re-normalize a set of log-probabilities via log-softmax.

    Args:
        logprobs: Raw log-probabilities (not necessarily summing to 1 in prob space).

    Returns:
        Normalized log-probabilities such that exp(lp) sums to 1.
    """
    max_lp = max(logprobs)
    # log-sum-exp for numerical stability
    lse = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in logprobs))
    return [lp - lse for lp in logprobs]


def kl_divergence(p_logprobs: list[float], q_logprobs: list[float]) -> float:
    """Compute KL divergence D_KL(P || Q) from log-probabilities.

    D_KL(P || Q) = sum(p * (log p - log q))

    Both inputs must be the same length and represent log-probs over the same
    set of outcomes.

    Args:
        p_logprobs: Log-probabilities of distribution P.
        q_logprobs: Log-probabilities of distribution Q.

    Returns:
        KL divergence in nats (non-negative).

    Raises:
        ValueError: If the input lists have different lengths.
    """
    if len(p_logprobs) != len(q_logprobs):
        raise ValueError("p_logprobs and q_logprobs must have the same length.")
    kl = 0.0
    for lp, lq in zip(p_logprobs, q_logprobs):
        p = math.exp(lp)
        if p > 0:
            kl += p * (lp - lq)
    return kl


def surprise(logprob: float) -> float:
    """Compute pointwise surprise (information content) from a log-probability.

    surprise = -log(p) = -logprob

    Args:
        logprob: The log-probability of a token.

    Returns:
        Surprise value in nats (non-negative).
    """
    return -logprob


def marginal_prob(logprobs_at_position: list[TokenLogProb], token_set: set[str]) -> float:
    """Sum probability mass over a set of tokens at a given position.

    Args:
        logprobs_at_position: Top-k tokens with log-probs at a single position.
        token_set: Set of token strings to include.

    Returns:
        Total probability mass for the specified tokens (between 0 and 1).
    """
    total = 0.0
    for tlp in logprobs_at_position:
        if tlp.token in token_set:
            total += math.exp(tlp.logprob)
    return total
