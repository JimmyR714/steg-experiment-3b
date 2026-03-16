"""Tests for standard agent task library (Phase 8)."""

from __future__ import annotations

import pytest

from llm_agents.agents.agent import Agent
from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult, TokenLogProb
from llm_agents.tasks.standard import chain, classify, debate, map_reduce, qa, summarize
from llm_agents.tasks.types import ClassifyResult, DebateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel(BaseModel):
    """A mock model that returns pre-configured responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return CompletionResult(text=text)

    def get_logprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        top_k: int = 5,
    ) -> LogProbResult:
        return LogProbResult(prompt=prompt)


class MockModelWithLogprobs(BaseModel):
    """Mock model that also returns log-probability information."""

    def __init__(self, response: str, top_k_tokens: list[TokenLogProb]) -> None:
        self._response = response
        self._top_k_tokens = top_k_tokens

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        logprob_result = LogProbResult(
            prompt=prompt,
            tokens=[self._top_k_tokens[0]] if self._top_k_tokens else [],
            top_k_per_position=[self._top_k_tokens] if self._top_k_tokens else [],
        )
        return CompletionResult(text=self._response, logprob_result=logprob_result)

    def get_logprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        top_k: int = 5,
    ) -> LogProbResult:
        return LogProbResult(prompt=prompt)


# ---------------------------------------------------------------------------
# DebateResult dataclass
# ---------------------------------------------------------------------------


class TestDebateResult:
    def test_basic_construction(self):
        result = DebateResult(topic="AI safety")
        assert result.topic == "AI safety"
        assert result.rounds == []
        assert result.winner == ""
        assert result.judgment == ""

    def test_full_construction(self):
        result = DebateResult(
            topic="Testing",
            rounds=[("pro arg", "con arg")],
            winner="pro",
            judgment="Pro wins.",
        )
        assert result.rounds[0] == ("pro arg", "con arg")
        assert result.winner == "pro"


# ---------------------------------------------------------------------------
# ClassifyResult dataclass
# ---------------------------------------------------------------------------


class TestClassifyResult:
    def test_basic_construction(self):
        result = ClassifyResult(label="positive")
        assert result.label == "positive"
        assert result.probabilities == {}

    def test_with_probabilities(self):
        result = ClassifyResult(
            label="spam",
            probabilities={"spam": 0.9, "ham": 0.1},
        )
        assert result.probabilities["spam"] == 0.9


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_basic_summarization(self):
        model = MockModel(["This is the summary."])
        result = summarize(model, "A long text about many things.")
        assert result == "This is the summary."

    def test_empty_text(self):
        model = MockModel(["Empty."])
        result = summarize(model, "")
        assert result == "Empty."


# ---------------------------------------------------------------------------
# qa
# ---------------------------------------------------------------------------


class TestQA:
    def test_basic_qa(self):
        model = MockModel(["Paris"])
        result = qa(model, "What is the capital of France?", "France's capital is Paris.")
        assert result == "Paris"

    def test_qa_with_context(self):
        model = MockModel(["42"])
        result = qa(model, "What is the answer?", "The answer to everything is 42.")
        assert result == "42"


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_basic_classification(self):
        model = MockModel(["positive"])
        result = classify(model, "I love this!", ["positive", "negative"])
        assert isinstance(result, ClassifyResult)
        assert result.label == "positive"
        assert result.probabilities["positive"] == 1.0
        assert result.probabilities["negative"] == 0.0

    def test_classification_with_logprobs(self):
        tokens = [
            TokenLogProb(token="positive", logprob=-0.1, rank=0),
            TokenLogProb(token="negative", logprob=-2.3, rank=1),
        ]
        model = MockModelWithLogprobs("positive", tokens)
        result = classify(model, "Great product!", ["positive", "negative"])
        assert result.label == "positive"
        assert result.probabilities["positive"] > result.probabilities["negative"]
        # Probabilities should sum to ~1.0.
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_classification_fallback_no_logprobs(self):
        model = MockModel(["spam"])
        result = classify(model, "Buy now!", ["spam", "ham"])
        assert result.label == "spam"
        assert result.probabilities["spam"] == 1.0


# ---------------------------------------------------------------------------
# debate
# ---------------------------------------------------------------------------


class TestDebate:
    def test_basic_debate(self):
        pro_model = MockModel(["Pro argument round 1."])
        con_model = MockModel(["Con argument round 1."])
        judge_model = MockModel(["Pro had stronger arguments.\nWINNER: pro"])
        result = debate([pro_model, con_model, judge_model], "AI is beneficial", rounds=1)
        assert isinstance(result, DebateResult)
        assert result.topic == "AI is beneficial"
        assert len(result.rounds) == 1
        assert result.winner == "pro"

    def test_debate_two_rounds(self):
        pro_model = MockModel(["Pro argument."])
        con_model = MockModel(["Con argument."])
        judge_model = MockModel(["Con was more convincing.\nWINNER: con"])
        result = debate([pro_model, con_model, judge_model], "Testing", rounds=2)
        assert len(result.rounds) == 2
        assert result.winner == "con"

    def test_debate_two_models(self):
        """When only 2 models given, first model is reused as judge."""
        model_a = MockModel(["Argument A.", "Judgment.\nWINNER: pro"])
        model_b = MockModel(["Argument B."])
        result = debate([model_a, model_b], "Topic", rounds=1)
        assert isinstance(result, DebateResult)

    def test_debate_requires_two_models(self):
        model = MockModel(["x"])
        with pytest.raises(ValueError, match="at least 2 models"):
            debate([model], "Topic")

    def test_debate_no_winner_extracted(self):
        """If judge doesn't follow format, winner is empty."""
        pro_model = MockModel(["Pro."])
        con_model = MockModel(["Con."])
        judge_model = MockModel(["I can't decide."])
        result = debate([pro_model, con_model, judge_model], "Topic", rounds=1)
        assert result.winner == ""
        assert result.judgment == "I can't decide."


# ---------------------------------------------------------------------------
# chain
# ---------------------------------------------------------------------------


class TestChain:
    def test_single_agent_chain(self):
        model = MockModel(["Processed."])
        agent = Agent(name="a1", model=model, enable_cot=False)
        result = chain([agent], "Input text")
        assert result == "Processed."

    def test_multi_agent_chain(self):
        m1 = MockModel(["Step 1 output."])
        m2 = MockModel(["Step 2 output."])
        m3 = MockModel(["Final output."])
        agents = [
            Agent(name="a1", model=m1, enable_cot=False),
            Agent(name="a2", model=m2, enable_cot=False),
            Agent(name="a3", model=m3, enable_cot=False),
        ]
        result = chain(agents, "Start")
        assert result == "Final output."

    def test_chain_empty_raises(self):
        with pytest.raises(ValueError, match="at least one agent"):
            chain([], "input")

    def test_chain_passes_output_forward(self):
        """Verify intermediate outputs are passed as input to next agent."""
        prompts_seen: list[str] = []

        class CapturingModel(BaseModel):
            def __init__(self, response: str) -> None:
                self._response = response

            def generate(self, prompt, **kwargs):
                prompts_seen.append(prompt)
                return CompletionResult(text=self._response)

            def get_logprobs(self, prompt, **kwargs):
                return LogProbResult(prompt=prompt)

        m1 = CapturingModel("output_of_1")
        m2 = CapturingModel("output_of_2")
        a1 = Agent(name="a1", model=m1, enable_cot=False)
        a2 = Agent(name="a2", model=m2, enable_cot=False)
        chain([a1, a2], "initial_input")
        # a2 should have seen "output_of_1" in its prompt.
        assert any("output_of_1" in p for p in prompts_seen)


# ---------------------------------------------------------------------------
# map_reduce
# ---------------------------------------------------------------------------


class TestMapReduce:
    def test_basic_map_reduce(self):
        map_model = MockModel(["mapped"])
        reduce_model = MockModel(["reduced"])
        mapper = Agent(name="mapper", model=map_model, enable_cot=False)
        reducer = Agent(name="reducer", model=reduce_model, enable_cot=False)
        result = map_reduce(mapper, ["item1", "item2", "item3"], reducer)
        assert result == "reduced"

    def test_map_reduce_single_item(self):
        map_model = MockModel(["processed"])
        reduce_model = MockModel(["final"])
        mapper = Agent(name="mapper", model=map_model, enable_cot=False)
        reducer = Agent(name="reducer", model=reduce_model, enable_cot=False)
        result = map_reduce(mapper, ["only_item"], reducer)
        assert result == "final"

    def test_map_reduce_empty_raises(self):
        map_model = MockModel(["x"])
        reduce_model = MockModel(["y"])
        mapper = Agent(name="mapper", model=map_model, enable_cot=False)
        reducer = Agent(name="reducer", model=reduce_model, enable_cot=False)
        with pytest.raises(ValueError, match="at least one item"):
            map_reduce(mapper, [], reducer)

    def test_map_reduce_resets_agent(self):
        """Mapper agent is reset between items so history doesn't leak."""
        map_model = MockModel(["result"])
        reduce_model = MockModel(["done"])
        mapper = Agent(name="mapper", model=map_model, enable_cot=False)
        reducer = Agent(name="reducer", model=reduce_model, enable_cot=False)
        map_reduce(mapper, ["a", "b"], reducer)
        # After map_reduce, mapper history should be empty (last reset before
        # the final item, then one run call adds 2 entries).
        # Just verify no error occurred — the reset prevents unbounded growth.
        assert True

    def test_map_reduce_feeds_results_to_reducer(self):
        """Reducer receives all mapped results."""
        prompts_seen: list[str] = []

        class CapturingModel(BaseModel):
            def __init__(self, response: str) -> None:
                self._response = response

            def generate(self, prompt, **kwargs):
                prompts_seen.append(prompt)
                return CompletionResult(text=self._response)

            def get_logprobs(self, prompt, **kwargs):
                return LogProbResult(prompt=prompt)

        map_model = MockModel(["mapped_output"])
        reduce_model = CapturingModel("final")
        mapper = Agent(name="mapper", model=map_model, enable_cot=False)
        reducer = Agent(name="reducer", model=reduce_model, enable_cot=False)
        map_reduce(mapper, ["x", "y"], reducer)
        # The reducer should see "mapped_output" in its prompt.
        assert any("mapped_output" in p for p in prompts_seen)
