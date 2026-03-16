"""Tests for Extension 3: Agent Self-Reflection & Critique."""

from __future__ import annotations

import json

import pytest

from llm_agents.agents.agent import Agent
from llm_agents.agents.reflection import (
    CritiqueResult,
    PeerCritique,
    ReflectiveAgent,
    SelfCritique,
    _parse_critique,
    _should_reflect,
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


# ---------------------------------------------------------------------------
# _parse_critique tests
# ---------------------------------------------------------------------------


class TestParseCritique:
    def test_valid_json(self):
        text = json.dumps({"accept": True, "feedback": "Good job", "score": 0.9})
        result = _parse_critique(text)
        assert result.accept is True
        assert result.feedback == "Good job"
        assert result.score == 0.9

    def test_json_in_prose(self):
        text = 'Here is my critique: {"accept": false, "feedback": "Needs work", "score": 0.3}'
        result = _parse_critique(text)
        assert result.accept is False
        assert result.feedback == "Needs work"

    def test_invalid_json_fallback(self):
        result = _parse_critique("This is not JSON at all.")
        assert result.accept is False
        assert result.score == 0.0
        assert "not JSON" in result.feedback

    def test_partial_fields(self):
        text = '{"accept": true}'
        result = _parse_critique(text)
        assert result.accept is True
        assert result.feedback == ""
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# _should_reflect tests
# ---------------------------------------------------------------------------


class TestShouldReflect:
    def test_none_logprobs(self):
        assert _should_reflect(None) is False

    def test_empty_positions(self):
        lp = LogProbResult(prompt="test", top_k_per_position=[])
        assert _should_reflect(lp) is False

    def test_low_entropy_no_reflect(self):
        # Low entropy: model is confident
        tokens = [
            TokenLogProb(token="yes", logprob=-0.01, rank=0),
            TokenLogProb(token="no", logprob=-10.0, rank=1),
        ]
        lp = LogProbResult(prompt="test", top_k_per_position=[tokens])
        assert _should_reflect(lp, entropy_threshold=2.0) is False

    def test_high_entropy_triggers_reflect(self):
        # High entropy: model is uncertain (uniform-ish distribution)
        import math

        n = 5
        logp = math.log(1.0 / n)
        tokens = [TokenLogProb(token=f"t{i}", logprob=logp, rank=i) for i in range(n)]
        lp = LogProbResult(prompt="test", top_k_per_position=[tokens])
        # Entropy of uniform over 5 ≈ 1.609; set threshold low
        assert _should_reflect(lp, entropy_threshold=1.0) is True


# ---------------------------------------------------------------------------
# SelfCritique tests
# ---------------------------------------------------------------------------


class TestSelfCritique:
    def test_returns_critique_result(self):
        critique_json = json.dumps({"accept": True, "feedback": "Looks good", "score": 0.85})
        model = MockModel([critique_json])
        critic = SelfCritique(model=model)
        result = critic.critique("What is 2+2?", "4")
        assert isinstance(result, CritiqueResult)
        assert result.accept is True


# ---------------------------------------------------------------------------
# PeerCritique tests
# ---------------------------------------------------------------------------


class TestPeerCritique:
    def test_returns_critique_result(self):
        critique_json = json.dumps({"accept": False, "feedback": "Incomplete", "score": 0.4})
        model = MockModel([critique_json])
        critic_agent = Agent(name="critic", model=model, enable_cot=False)
        peer = PeerCritique(critic_agent)
        result = peer.critique("Explain gravity", "Stuff falls down")
        assert isinstance(result, CritiqueResult)
        assert result.accept is False


# ---------------------------------------------------------------------------
# ReflectiveAgent tests
# ---------------------------------------------------------------------------


class TestReflectiveAgent:
    def test_accepted_on_first_try(self):
        critique_json = json.dumps({"accept": True, "feedback": "OK", "score": 0.9})
        # Agent model gives answer, critic model accepts
        agent_model = MockModel(["The answer is 42."])
        critic_model = MockModel([critique_json])

        agent = Agent(name="test", model=agent_model, enable_cot=False)
        critic = SelfCritique(model=critic_model)
        reflective = ReflectiveAgent(agent=agent, critic=critic, max_rounds=2)

        resp = reflective.run("What is the meaning of life?")
        assert resp.content == "The answer is 42."

    def test_rejected_then_accepted(self):
        reject_json = json.dumps({"accept": False, "feedback": "Too vague", "score": 0.3})
        accept_json = json.dumps({"accept": True, "feedback": "Better", "score": 0.8})

        # Agent gives first answer, then improved answer
        agent_model = MockModel(["Vague answer.", "Precise answer."])
        # Critic rejects first, accepts second
        critic_model = MockModel([reject_json, accept_json])

        agent = Agent(name="test", model=agent_model, enable_cot=False)
        critic = SelfCritique(model=critic_model)
        reflective = ReflectiveAgent(agent=agent, critic=critic, max_rounds=2)

        resp = reflective.run("Explain quantum physics")
        assert resp.content == "Precise answer."

    def test_max_rounds_exhausted(self):
        reject_json = json.dumps({"accept": False, "feedback": "Still bad", "score": 0.2})
        agent_model = MockModel(["Attempt 1", "Attempt 2", "Attempt 3"])
        critic_model = MockModel([reject_json])

        agent = Agent(name="test", model=agent_model, enable_cot=False)
        critic = SelfCritique(model=critic_model)
        reflective = ReflectiveAgent(agent=agent, critic=critic, max_rounds=2)

        resp = reflective.run("Hard question")
        # Should return last attempt after exhausting rounds
        assert resp.content is not None


# ---------------------------------------------------------------------------
# CritiqueResult dataclass
# ---------------------------------------------------------------------------


class TestCritiqueResult:
    def test_fields(self):
        cr = CritiqueResult(accept=True, feedback="Good", score=0.95)
        assert cr.accept is True
        assert cr.feedback == "Good"
        assert cr.score == 0.95
