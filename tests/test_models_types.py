"""Tests for llm_agents.models.types dataclasses."""

import math

from llm_agents.models.types import CompletionResult, LogProbResult, TokenLogProb


class TestTokenLogProb:
    def test_basic_construction(self):
        t = TokenLogProb(token="hello", logprob=-0.5, rank=0)
        assert t.token == "hello"
        assert t.logprob == -0.5
        assert t.rank == 0

    def test_frozen(self):
        t = TokenLogProb(token="a", logprob=-1.0, rank=1)
        try:
            t.token = "b"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass

    def test_equality(self):
        a = TokenLogProb(token="x", logprob=-2.0, rank=3)
        b = TokenLogProb(token="x", logprob=-2.0, rank=3)
        assert a == b

    def test_negative_logprob(self):
        t = TokenLogProb(token="the", logprob=-12.345, rank=5)
        assert t.logprob < 0

    def test_zero_logprob(self):
        t = TokenLogProb(token="certain", logprob=0.0, rank=0)
        assert t.logprob == 0.0


class TestLogProbResult:
    def test_empty_construction(self):
        r = LogProbResult(prompt="Hello")
        assert r.prompt == "Hello"
        assert r.tokens == []
        assert r.top_k_per_position == []

    def test_with_tokens(self):
        tokens = [
            TokenLogProb(token="world", logprob=-0.1, rank=0),
            TokenLogProb(token="!", logprob=-0.3, rank=0),
        ]
        r = LogProbResult(prompt="Hello", tokens=tokens)
        assert len(r.tokens) == 2
        assert r.tokens[0].token == "world"
        assert r.tokens[1].token == "!"

    def test_with_top_k(self):
        top_k = [
            [
                TokenLogProb(token="world", logprob=-0.1, rank=0),
                TokenLogProb(token="there", logprob=-1.2, rank=1),
            ],
            [
                TokenLogProb(token="!", logprob=-0.3, rank=0),
                TokenLogProb(token=".", logprob=-0.9, rank=1),
            ],
        ]
        r = LogProbResult(prompt="Hello", top_k_per_position=top_k)
        assert len(r.top_k_per_position) == 2
        assert r.top_k_per_position[0][1].token == "there"

    def test_frozen(self):
        r = LogProbResult(prompt="test")
        try:
            r.prompt = "changed"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass

    def test_default_factory_independence(self):
        """Each instance should get its own default list."""
        r1 = LogProbResult(prompt="a")
        r2 = LogProbResult(prompt="b")
        assert r1.tokens is not r2.tokens
        assert r1.top_k_per_position is not r2.top_k_per_position


class TestCompletionResult:
    def test_basic_construction(self):
        c = CompletionResult(text="Hello world")
        assert c.text == "Hello world"
        assert c.logprob_result is None
        assert c.finish_reason == "stop"

    def test_with_logprob_result(self):
        lpr = LogProbResult(
            prompt="Hi",
            tokens=[TokenLogProb(token="there", logprob=-0.5, rank=0)],
        )
        c = CompletionResult(text="there", logprob_result=lpr, finish_reason="stop")
        assert c.logprob_result is not None
        assert c.logprob_result.tokens[0].token == "there"

    def test_custom_finish_reason(self):
        c = CompletionResult(text="long text...", finish_reason="length")
        assert c.finish_reason == "length"

    def test_frozen(self):
        c = CompletionResult(text="test")
        try:
            c.text = "changed"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass

    def test_sequence_logprobs(self):
        """Test a realistic multi-token completion with log-probs."""
        tokens = [
            TokenLogProb(token="The", logprob=-0.2, rank=0),
            TokenLogProb(token=" cat", logprob=-1.5, rank=0),
            TokenLogProb(token=" sat", logprob=-0.8, rank=0),
        ]
        lpr = LogProbResult(prompt="Complete:", tokens=tokens)
        c = CompletionResult(text="The cat sat", logprob_result=lpr)

        total_logprob = sum(t.logprob for t in c.logprob_result.tokens)
        assert math.isclose(total_logprob, -2.5, rel_tol=1e-9)
