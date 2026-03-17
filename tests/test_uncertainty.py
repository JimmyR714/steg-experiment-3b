"""Tests for Extension 17: Logprob-Based Uncertainty Quantification."""

from __future__ import annotations

import math

import pytest

from llm_agents.models.types import LogProbResult, TokenLogProb
from llm_agents.logprobs.uncertainty import (
    CalibrationPoint,
    calibration_curve,
    confidence_score,
    entropy_map,
    expected_calibration_error,
    is_hallucination_risk,
    token_uncertainty_map,
    uncertain_spans,
)
from llm_agents.logprobs.sampling import (
    ConsistencyResult,
    PredictionSet,
    diverse_sample,
    self_consistency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logprob_result(logprobs: list[float], tokens: list[str] | None = None) -> LogProbResult:
    """Create a LogProbResult from a list of log-probabilities."""
    if tokens is None:
        tokens = [f"t{i}" for i in range(len(logprobs))]
    token_list = [
        TokenLogProb(token=t, logprob=lp, rank=0)
        for t, lp in zip(tokens, logprobs)
    ]
    return LogProbResult(prompt="test", tokens=token_list)


def _make_logprob_result_with_topk(
    logprobs: list[float],
    top_k: list[list[tuple[str, float]]],
) -> LogProbResult:
    tokens = [
        TokenLogProb(token=f"t{i}", logprob=lp, rank=0)
        for i, lp in enumerate(logprobs)
    ]
    top_k_per_pos = [
        [TokenLogProb(token=t, logprob=lp, rank=j) for j, (t, lp) in enumerate(pos)]
        for pos in top_k
    ]
    return LogProbResult(prompt="test", tokens=tokens, top_k_per_position=top_k_per_pos)


# ---------------------------------------------------------------------------
# confidence_score tests
# ---------------------------------------------------------------------------


class TestConfidenceScore:
    def test_high_confidence(self):
        # log(1) = 0.0 => perplexity = 1.0 => confidence = 1.0
        result = _make_logprob_result([0.0, 0.0, 0.0])
        assert confidence_score(result) == pytest.approx(1.0)

    def test_low_confidence(self):
        # Very negative logprobs => high perplexity => low confidence
        result = _make_logprob_result([-5.0, -5.0, -5.0])
        score = confidence_score(result)
        assert 0.0 < score < 0.1

    def test_empty_tokens(self):
        result = LogProbResult(prompt="test", tokens=[])
        assert confidence_score(result) == 0.0


# ---------------------------------------------------------------------------
# token_uncertainty_map tests
# ---------------------------------------------------------------------------


class TestTokenUncertaintyMap:
    def test_basic(self):
        result = _make_logprob_result([-0.5, -2.0, -0.1])
        umap = token_uncertainty_map(result)
        assert len(umap) == 3
        assert umap[0] == ("t0", 0.5)
        assert umap[1] == ("t1", 2.0)
        assert umap[2] == ("t2", pytest.approx(0.1))

    def test_empty(self):
        result = LogProbResult(prompt="test", tokens=[])
        assert token_uncertainty_map(result) == []


# ---------------------------------------------------------------------------
# entropy_map tests
# ---------------------------------------------------------------------------


class TestEntropyMap:
    def test_with_topk(self):
        top_k = [
            [("a", -0.1), ("b", -2.3)],
            [("c", -0.5), ("d", -1.0)],
        ]
        result = _make_logprob_result_with_topk([-0.1, -0.5], top_k)
        emap = entropy_map(result)
        assert len(emap) == 2
        # Each entry should have non-negative entropy
        for _, h in emap:
            assert h >= 0

    def test_fallback_without_topk(self):
        result = _make_logprob_result([-1.0, -2.0])
        emap = entropy_map(result)
        assert len(emap) == 2
        # Without top-k, falls back to surprise
        assert emap[0][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# is_hallucination_risk tests
# ---------------------------------------------------------------------------


class TestIsHallucinationRisk:
    def test_confident_output(self):
        result = _make_logprob_result([-0.1, -0.2, -0.1, -0.3])
        assert not is_hallucination_risk(result, threshold=3.0, min_span_length=3)

    def test_uncertain_span(self):
        # 4 consecutive uncertain tokens
        result = _make_logprob_result([-0.1, -4.0, -5.0, -4.0, -4.5, -0.2])
        assert is_hallucination_risk(result, threshold=3.0, min_span_length=3)

    def test_short_uncertain_span(self):
        result = _make_logprob_result([-0.1, -4.0, -4.0, -0.2])
        assert not is_hallucination_risk(result, threshold=3.0, min_span_length=3)

    def test_empty(self):
        result = LogProbResult(prompt="test", tokens=[])
        assert not is_hallucination_risk(result)


# ---------------------------------------------------------------------------
# uncertain_spans tests
# ---------------------------------------------------------------------------


class TestUncertainSpans:
    def test_find_spans(self):
        result = _make_logprob_result([-0.1, -4.0, -5.0, -0.2, -6.0, -7.0])
        spans = uncertain_spans(result, threshold=3.0, min_length=2)
        assert len(spans) == 2

    def test_no_spans(self):
        result = _make_logprob_result([-0.1, -0.2, -0.3])
        spans = uncertain_spans(result, threshold=3.0)
        assert len(spans) == 0


# ---------------------------------------------------------------------------
# calibration_curve tests
# ---------------------------------------------------------------------------


class TestCalibrationCurve:
    def test_basic_curve(self):
        predictions = [0.9, 0.8, 0.7, 0.2, 0.1]
        actuals = [True, True, True, False, False]
        curve = calibration_curve(predictions, actuals, n_bins=5)
        assert len(curve) > 0
        for point in curve:
            assert 0.0 <= point.predicted_confidence <= 1.0
            assert 0.0 <= point.actual_accuracy <= 1.0
            assert point.count > 0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            calibration_curve([0.5], [True, False])


# ---------------------------------------------------------------------------
# expected_calibration_error tests
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        # All confident and correct
        predictions = [1.0, 1.0, 1.0]
        actuals = [True, True, True]
        ece = expected_calibration_error(predictions, actuals, n_bins=5)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_imperfect_calibration(self):
        predictions = [0.9, 0.9, 0.9]
        actuals = [True, False, False]
        ece = expected_calibration_error(predictions, actuals)
        assert ece > 0.0


# ---------------------------------------------------------------------------
# diverse_sample tests (with mock model)
# ---------------------------------------------------------------------------


class _MockModel:
    def __init__(self):
        self.temperatures: list[float] = []

    def generate(self, prompt, *, max_tokens=256, temperature=1.0, top_k=50, stop=None):
        from llm_agents.models.types import CompletionResult
        self.temperatures.append(temperature)
        return CompletionResult(text=f"temp={temperature:.1f}")

    def get_logprobs(self, prompt, **kwargs):
        return LogProbResult(prompt=prompt)


class TestDiverseSample:
    def test_generates_n_samples(self):
        model = _MockModel()
        results = diverse_sample(model, "test", n=5)
        assert len(results) == 5
        assert len(model.temperatures) == 5

    def test_varied_temperatures(self):
        model = _MockModel()
        diverse_sample(model, "test", n=3)
        temps = model.temperatures
        # Should have different temperatures
        assert temps[0] != temps[-1]

    def test_custom_schedule(self):
        model = _MockModel()
        diverse_sample(model, "test", n=2, temperature_schedule=[0.5, 1.5])
        assert model.temperatures == [0.5, 1.5]


# ---------------------------------------------------------------------------
# self_consistency tests
# ---------------------------------------------------------------------------


class _FixedModel:
    """Returns answers from a fixed list cyclically."""

    def __init__(self, answers: list[str]):
        self._answers = answers
        self._idx = 0

    def generate(self, prompt, **kwargs):
        from llm_agents.models.types import CompletionResult
        text = self._answers[self._idx % len(self._answers)]
        self._idx += 1
        return CompletionResult(text=text)

    def get_logprobs(self, prompt, **kwargs):
        return LogProbResult(prompt=prompt)


class TestSelfConsistency:
    def test_unanimous_agreement(self):
        model = _FixedModel(["42"])
        result = self_consistency(model, "What is 6*7?", n=5)
        assert result.confidence == 1.0
        assert "42" in result.answer

    def test_majority_wins(self):
        model = _FixedModel(["42", "42", "43"])
        result = self_consistency(model, "test", n=3)
        assert "42" in result.answer
        assert result.confidence == pytest.approx(2 / 3)

    def test_samples_stored(self):
        model = _FixedModel(["a", "b"])
        result = self_consistency(model, "test", n=4)
        assert len(result.samples) == 4
