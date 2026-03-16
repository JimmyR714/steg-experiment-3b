"""Tests for the OpenAI model backend with mocked API responses."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from llm_agents.models.openai_model import OpenAIModel
from llm_agents.models.registry import ModelRegistry, register_model, get_model, list_models
from llm_agents.models.types import CompletionResult, LogProbResult, TokenLogProb
from llm_agents.models.base import BaseModel


# -----------------------------------------------------------------------
# Helpers to build mock OpenAI API responses
# -----------------------------------------------------------------------


def _make_top_logprob(token: str, logprob: float):
    """Create a mock top-logprob entry."""
    m = MagicMock()
    m.token = token
    m.logprob = logprob
    return m


def _make_token_logprob(token: str, logprob: float, top_logprobs: list):
    """Create a mock content token info object."""
    m = MagicMock()
    m.token = token
    m.logprob = logprob
    m.top_logprobs = top_logprobs
    return m


def _make_choice(
    text: str,
    finish_reason: str = "stop",
    logprobs_content: list | None = None,
):
    """Create a mock API response choice."""
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = finish_reason

    if logprobs_content is not None:
        choice.logprobs.content = logprobs_content
    else:
        choice.logprobs = None

    return choice


def _make_response(choice):
    """Wrap a choice into a mock API response."""
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# -----------------------------------------------------------------------
# OpenAIModel tests
# -----------------------------------------------------------------------


class TestOpenAIModelGenerate:
    """Tests for OpenAIModel.generate()."""

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_generate_basic(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        choice = _make_choice("Hello, world!")
        mock_client.chat.completions.create.return_value = _make_response(choice)

        model = OpenAIModel(model="gpt-test", api_key="fake")
        result = model.generate("Say hi")

        assert isinstance(result, CompletionResult)
        assert result.text == "Hello, world!"
        assert result.finish_reason == "stop"
        assert result.logprob_result is None

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_generate_with_stop(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        choice = _make_choice("partial", finish_reason="stop")
        mock_client.chat.completions.create.return_value = _make_response(choice)

        model = OpenAIModel(model="gpt-test", api_key="fake")
        result = model.generate("Test", stop=["\n"])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stop"] == ["\n"]
        assert result.text == "partial"

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_generate_length_finish(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        choice = _make_choice("truncated output", finish_reason="length")
        mock_client.chat.completions.create.return_value = _make_response(choice)

        model = OpenAIModel(model="gpt-test", api_key="fake")
        result = model.generate("Prompt", max_tokens=5)

        assert result.finish_reason == "length"

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_generate_empty_content(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        choice = _make_choice(None, finish_reason="stop")
        mock_client.chat.completions.create.return_value = _make_response(choice)

        model = OpenAIModel(model="gpt-test", api_key="fake")
        result = model.generate("Prompt")

        assert result.text == ""


class TestOpenAIModelLogprobs:
    """Tests for OpenAIModel.get_logprobs()."""

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_get_logprobs_basic(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        top_lps = [
            _make_top_logprob("Hello", -0.1),
            _make_top_logprob("Hi", -0.5),
            _make_top_logprob("Hey", -1.2),
        ]
        content = [
            _make_token_logprob("Hello", -0.1, top_lps),
        ]
        choice = _make_choice("Hello", logprobs_content=content)
        mock_client.chat.completions.create.return_value = _make_response(choice)

        model = OpenAIModel(model="gpt-test", api_key="fake")
        result = model.get_logprobs("Greet me", top_k=3)

        assert isinstance(result, LogProbResult)
        assert result.prompt == "Greet me"
        assert len(result.tokens) == 1
        assert result.tokens[0].token == "Hello"
        assert result.tokens[0].logprob == pytest.approx(-0.1)

        assert len(result.top_k_per_position) == 1
        assert len(result.top_k_per_position[0]) == 3
        assert result.top_k_per_position[0][0].token == "Hello"
        assert result.top_k_per_position[0][1].token == "Hi"
        assert result.top_k_per_position[0][2].token == "Hey"

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_get_logprobs_multi_token(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        content = [
            _make_token_logprob(
                "The", -0.2, [_make_top_logprob("The", -0.2), _make_top_logprob("A", -0.8)]
            ),
            _make_token_logprob(
                " cat", -0.5, [_make_top_logprob(" cat", -0.5), _make_top_logprob(" dog", -0.9)]
            ),
        ]
        choice = _make_choice("The cat", logprobs_content=content)
        mock_client.chat.completions.create.return_value = _make_response(choice)

        model = OpenAIModel(model="gpt-test", api_key="fake")
        result = model.get_logprobs("Complete:", top_k=2)

        assert len(result.tokens) == 2
        assert result.tokens[0].token == "The"
        assert result.tokens[1].token == " cat"
        assert len(result.top_k_per_position) == 2

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_get_logprobs_no_logprobs_in_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        choice = _make_choice("text", logprobs_content=None)
        choice.logprobs = None
        mock_client.chat.completions.create.return_value = _make_response(choice)

        model = OpenAIModel(model="gpt-test", api_key="fake")
        result = model.get_logprobs("Prompt")

        assert isinstance(result, LogProbResult)
        assert result.prompt == "Prompt"
        assert result.tokens == []
        assert result.top_k_per_position == []

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_get_logprobs_passes_top_logprobs_param(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        choice = _make_choice("ok", logprobs_content=[])
        mock_client.chat.completions.create.return_value = _make_response(choice)

        model = OpenAIModel(model="gpt-test", api_key="fake")
        model.get_logprobs("Test", top_k=10)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 10


class TestOpenAIModelInit:
    """Tests for OpenAIModel construction."""

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_custom_base_url(self, mock_openai_cls):
        OpenAIModel(model="local", api_key="x", base_url="http://localhost:8000/v1")
        mock_openai_cls.assert_called_once_with(
            api_key="x", base_url="http://localhost:8000/v1"
        )

    @patch("llm_agents.models.openai_model.OpenAI")
    def test_is_base_model_subclass(self, mock_openai_cls):
        model = OpenAIModel(model="gpt-test", api_key="fake")
        assert isinstance(model, BaseModel)


# -----------------------------------------------------------------------
# ModelRegistry tests
# -----------------------------------------------------------------------


class _DummyModel(BaseModel):
    """Minimal concrete model for testing the registry."""

    def generate(self, prompt, **kwargs):
        return CompletionResult(text="dummy")

    def get_logprobs(self, prompt, **kwargs):
        return LogProbResult(prompt=prompt)


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_get(self):
        reg = ModelRegistry()
        m = _DummyModel()
        reg.register_model("test", m)
        assert reg.get_model("test") is m

    def test_list_models_sorted(self):
        reg = ModelRegistry()
        reg.register_model("bravo", _DummyModel())
        reg.register_model("alpha", _DummyModel())
        reg.register_model("charlie", _DummyModel())
        assert reg.list_models() == ["alpha", "bravo", "charlie"]

    def test_get_missing_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError, match="no-such-model"):
            reg.get_model("no-such-model")

    def test_duplicate_register_raises(self):
        reg = ModelRegistry()
        reg.register_model("dup", _DummyModel())
        with pytest.raises(ValueError, match="already registered"):
            reg.register_model("dup", _DummyModel())

    def test_register_non_model_raises(self):
        reg = ModelRegistry()
        with pytest.raises(TypeError, match="BaseModel"):
            reg.register_model("bad", "not a model")  # type: ignore

    def test_list_empty(self):
        reg = ModelRegistry()
        assert reg.list_models() == []


class TestModuleLevelRegistry:
    """Tests for module-level convenience functions."""

    def test_module_functions_work(self):
        # Use a fresh registry to avoid cross-test pollution.
        import llm_agents.models.registry as reg_mod

        old = reg_mod._default_registry
        try:
            reg_mod._default_registry = ModelRegistry()
            m = _DummyModel()
            register_model("global-test", m)
            assert get_model("global-test") is m
            assert list_models() == ["global-test"]
        finally:
            reg_mod._default_registry = old
