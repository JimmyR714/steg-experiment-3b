"""Tests for Extension 5: Consensus & Voting Mechanisms."""

from __future__ import annotations

import math

import pytest

from llm_agents.agents.agent import Agent
from llm_agents.agents.consensus import (
    ConsensusResult,
    _confidence_from_logprobs,
    debate_consensus,
    majority_vote,
    ranked_choice,
    weighted_vote,
)
from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult, TokenLogProb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel(BaseModel):
    """Mock model returning pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def generate(self, prompt: str, **kwargs) -> CompletionResult:
        text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return CompletionResult(text=text)

    def get_logprobs(self, prompt: str, **kwargs) -> LogProbResult:
        return LogProbResult(prompt=prompt)


def _make_agent(name: str, response: str) -> Agent:
    model = MockModel([response])
    return Agent(name=name, model=model, enable_cot=False)


# ---------------------------------------------------------------------------
# ConsensusResult tests
# ---------------------------------------------------------------------------


class TestConsensusResult:
    def test_fields(self):
        r = ConsensusResult(answer="42", confidence=0.8)
        assert r.answer == "42"
        assert r.confidence == 0.8
        assert r.vote_distribution == {}
        assert r.dissenting_views == []


# ---------------------------------------------------------------------------
# _confidence_from_logprobs tests
# ---------------------------------------------------------------------------


class TestConfidenceFromLogprobs:
    def test_none_returns_default(self):
        assert _confidence_from_logprobs(None) == 0.5

    def test_empty_tokens_returns_default(self):
        lp = LogProbResult(prompt="test", tokens=[])
        assert _confidence_from_logprobs(lp) == 0.5

    def test_high_confidence(self):
        tokens = [TokenLogProb(token="yes", logprob=-0.01, rank=0)]
        lp = LogProbResult(prompt="test", tokens=tokens)
        conf = _confidence_from_logprobs(lp)
        assert conf > 0.9

    def test_low_confidence(self):
        tokens = [TokenLogProb(token="uh", logprob=-5.0, rank=0)]
        lp = LogProbResult(prompt="test", tokens=tokens)
        conf = _confidence_from_logprobs(lp)
        assert conf < 0.1


# ---------------------------------------------------------------------------
# majority_vote tests
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_clear_majority(self):
        agents = [
            _make_agent("a1", "Paris"),
            _make_agent("a2", "Paris"),
            _make_agent("a3", "London"),
        ]
        result = majority_vote(agents, "What is the capital of France?")
        assert result.answer == "Paris"
        assert result.confidence == pytest.approx(2.0 / 3.0)
        assert "London" in result.dissenting_views

    def test_unanimous(self):
        agents = [_make_agent(f"a{i}", "42") for i in range(3)]
        result = majority_vote(agents, "Meaning of life?")
        assert result.answer == "42"
        assert result.confidence == 1.0
        assert result.dissenting_views == []

    def test_single_agent(self):
        agents = [_make_agent("a1", "Yes")]
        result = majority_vote(agents, "Simple question?")
        assert result.answer == "Yes"


# ---------------------------------------------------------------------------
# weighted_vote tests
# ---------------------------------------------------------------------------


class TestWeightedVote:
    def test_basic_weighted(self):
        agents = [
            _make_agent("a1", "A"),
            _make_agent("a2", "B"),
            _make_agent("a3", "A"),
        ]
        result = weighted_vote(agents, [1.0, 2.0, 1.0], "Pick one")
        # A gets 1.0 + 1.0 = 2.0, B gets 2.0 — tie broken by order
        assert result.answer in ("A", "B")

    def test_weight_dominance(self):
        agents = [
            _make_agent("a1", "A"),
            _make_agent("a2", "B"),
        ]
        result = weighted_vote(agents, [10.0, 1.0], "Pick one")
        assert result.answer == "A"

    def test_mismatched_lengths(self):
        agents = [_make_agent("a1", "A")]
        with pytest.raises(ValueError, match="same length"):
            weighted_vote(agents, [1.0, 2.0], "Pick one")


# ---------------------------------------------------------------------------
# ranked_choice tests
# ---------------------------------------------------------------------------


class TestRankedChoice:
    def test_single_candidate(self):
        agents = [_make_agent(f"a{i}", "Same") for i in range(3)]
        result = ranked_choice(agents, "Vote", rounds=2)
        assert result.answer == "Same"

    def test_multiple_candidates(self):
        agents = [
            _make_agent("a1", "A"),
            _make_agent("a2", "B"),
            _make_agent("a3", "A"),
        ]
        result = ranked_choice(agents, "Vote", rounds=2)
        assert isinstance(result, ConsensusResult)
        assert result.answer != ""


# ---------------------------------------------------------------------------
# debate_consensus tests
# ---------------------------------------------------------------------------


class TestDebateConsensus:
    def test_immediate_convergence(self):
        agents = [_make_agent(f"a{i}", "42") for i in range(3)]
        result = debate_consensus(agents, "Answer?", max_rounds=3)
        assert result.answer == "42"
        assert result.confidence == 1.0

    def test_too_few_agents(self):
        agents = [_make_agent("a1", "X")]
        with pytest.raises(ValueError, match="at least 2"):
            debate_consensus(agents, "Question?")

    def test_non_convergence(self):
        # Agents that keep disagreeing
        agents = [
            Agent(name="a1", model=MockModel(["A", "A", "A", "A"]), enable_cot=False),
            Agent(name="a2", model=MockModel(["B", "B", "B", "B"]), enable_cot=False),
        ]
        result = debate_consensus(agents, "Agree?", max_rounds=2)
        assert isinstance(result, ConsensusResult)
        assert result.answer in ("A", "B")
