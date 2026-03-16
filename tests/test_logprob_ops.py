"""Tests for llm_agents.logprobs.ops with hand-computed expected values."""

from __future__ import annotations

import math

import pytest

from llm_agents.logprobs.ops import (
    entropy,
    kl_divergence,
    marginal_prob,
    normalize,
    perplexity,
    surprise,
    top_k_tokens,
)
from llm_agents.models.types import LogProbResult, TokenLogProb


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_two_tokens(self):
        """Uniform distribution over 2 tokens: H = ln(2) ≈ 0.6931."""
        tokens = [
            TokenLogProb("a", math.log(0.5), 0),
            TokenLogProb("b", math.log(0.5), 1),
        ]
        assert math.isclose(entropy(tokens), math.log(2), rel_tol=1e-9)

    def test_deterministic(self):
        """One token has all the mass: H = 0."""
        tokens = [
            TokenLogProb("a", math.log(1.0), 0),
        ]
        assert math.isclose(entropy(tokens), 0.0, abs_tol=1e-12)

    def test_skewed_distribution(self):
        """p = [0.8, 0.2].  H = -(0.8*ln(0.8) + 0.2*ln(0.2))."""
        p = [0.8, 0.2]
        expected = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
        tokens = [
            TokenLogProb("x", math.log(0.8), 0),
            TokenLogProb("y", math.log(0.2), 1),
        ]
        assert math.isclose(entropy(tokens), expected, rel_tol=1e-9)

    def test_empty_list(self):
        """Empty list gives entropy 0."""
        assert entropy([]) == 0.0


# ---------------------------------------------------------------------------
# perplexity
# ---------------------------------------------------------------------------

class TestPerplexity:
    def test_single_token(self):
        """Single token with logprob ln(0.25): PP = exp(-ln(0.25)) = 4."""
        result = LogProbResult(
            prompt="test",
            tokens=[TokenLogProb("a", math.log(0.25), 0)],
        )
        assert math.isclose(perplexity(result), 4.0, rel_tol=1e-9)

    def test_two_tokens(self):
        """Two tokens with logprobs ln(0.5) each: PP = exp(-ln(0.5)) = 2."""
        result = LogProbResult(
            prompt="test",
            tokens=[
                TokenLogProb("a", math.log(0.5), 0),
                TokenLogProb("b", math.log(0.5), 1),
            ],
        )
        assert math.isclose(perplexity(result), 2.0, rel_tol=1e-9)

    def test_perfect_prediction(self):
        """Tokens with logprob 0 (prob=1): PP = 1."""
        result = LogProbResult(
            prompt="test",
            tokens=[TokenLogProb("a", 0.0, 0), TokenLogProb("b", 0.0, 0)],
        )
        assert math.isclose(perplexity(result), 1.0, rel_tol=1e-9)

    def test_empty_raises(self):
        result = LogProbResult(prompt="test", tokens=[])
        with pytest.raises(ValueError, match="empty"):
            perplexity(result)


# ---------------------------------------------------------------------------
# top_k_tokens
# ---------------------------------------------------------------------------

class TestTopKTokens:
    def test_basic(self):
        tokens = [
            TokenLogProb("c", -3.0, 2),
            TokenLogProb("a", -1.0, 0),
            TokenLogProb("b", -2.0, 1),
        ]
        result = top_k_tokens(tokens, 2)
        assert len(result) == 2
        assert result[0].token == "a"
        assert result[1].token == "b"

    def test_k_larger_than_list(self):
        tokens = [TokenLogProb("a", -1.0, 0)]
        result = top_k_tokens(tokens, 5)
        assert len(result) == 1

    def test_k_zero(self):
        tokens = [TokenLogProb("a", -1.0, 0)]
        assert top_k_tokens(tokens, 0) == []


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_already_normalized(self):
        """ln(0.7) and ln(0.3) should stay (roughly) the same."""
        lps = [math.log(0.7), math.log(0.3)]
        normed = normalize(lps)
        probs = [math.exp(lp) for lp in normed]
        assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)
        assert math.isclose(probs[0], 0.7, rel_tol=1e-9)

    def test_unnormalized(self):
        """Raw scores [0, 0, 0] → uniform 1/3 each."""
        normed = normalize([0.0, 0.0, 0.0])
        probs = [math.exp(lp) for lp in normed]
        for p in probs:
            assert math.isclose(p, 1.0 / 3.0, rel_tol=1e-9)

    def test_single_element(self):
        """Single element always normalizes to log(1) = 0."""
        normed = normalize([-5.0])
        assert math.isclose(normed[0], 0.0, abs_tol=1e-12)

    def test_large_spread(self):
        """Numerical stability with large spread."""
        normed = normalize([1000.0, 0.0])
        probs = [math.exp(lp) for lp in normed]
        assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)
        # First element dominates
        assert math.isclose(probs[0], 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_identical_distributions(self):
        """KL(P || P) = 0."""
        lps = [math.log(0.5), math.log(0.5)]
        assert math.isclose(kl_divergence(lps, lps), 0.0, abs_tol=1e-12)

    def test_known_value(self):
        """P = [0.8, 0.2], Q = [0.5, 0.5].
        KL = 0.8*ln(0.8/0.5) + 0.2*ln(0.2/0.5)
           = 0.8*ln(1.6) + 0.2*ln(0.4)
        """
        p = [math.log(0.8), math.log(0.2)]
        q = [math.log(0.5), math.log(0.5)]
        expected = 0.8 * math.log(1.6) + 0.2 * math.log(0.4)
        assert math.isclose(kl_divergence(p, q), expected, rel_tol=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            kl_divergence([0.0], [0.0, 0.0])

    def test_non_negative(self):
        """KL divergence is always non-negative (Gibbs' inequality)."""
        p = [math.log(0.3), math.log(0.7)]
        q = [math.log(0.6), math.log(0.4)]
        assert kl_divergence(p, q) >= 0


# ---------------------------------------------------------------------------
# surprise
# ---------------------------------------------------------------------------

class TestSurprise:
    def test_certain_event(self):
        """logprob = 0 → surprise = 0."""
        assert surprise(0.0) == 0.0

    def test_unlikely_event(self):
        """logprob = ln(0.01) → surprise = -ln(0.01) ≈ 4.605."""
        lp = math.log(0.01)
        assert math.isclose(surprise(lp), -math.log(0.01), rel_tol=1e-9)

    def test_half_probability(self):
        """logprob = ln(0.5) → surprise = ln(2) ≈ 0.6931."""
        assert math.isclose(surprise(math.log(0.5)), math.log(2), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# marginal_prob
# ---------------------------------------------------------------------------

class TestMarginalProb:
    def test_single_match(self):
        tokens = [
            TokenLogProb("yes", math.log(0.7), 0),
            TokenLogProb("no", math.log(0.3), 1),
        ]
        assert math.isclose(marginal_prob(tokens, {"yes"}), 0.7, rel_tol=1e-9)

    def test_multiple_matches(self):
        tokens = [
            TokenLogProb("a", math.log(0.5), 0),
            TokenLogProb("b", math.log(0.3), 1),
            TokenLogProb("c", math.log(0.2), 2),
        ]
        # marginal over {a, c} = 0.5 + 0.2 = 0.7
        assert math.isclose(marginal_prob(tokens, {"a", "c"}), 0.7, rel_tol=1e-9)

    def test_no_match(self):
        tokens = [TokenLogProb("x", math.log(1.0), 0)]
        assert marginal_prob(tokens, {"z"}) == 0.0

    def test_empty_token_list(self):
        assert marginal_prob([], {"a"}) == 0.0
