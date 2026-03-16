"""Tests for the single agent core (Phase 6)."""

from __future__ import annotations

import json

import pytest

from llm_agents.agents.agent import (
    Agent,
    AgentResponse,
    ToolCallRecord,
    _build_prompt,
    _extract_tool_call,
)
from llm_agents.agents.cot import (
    COT_INSTRUCTION,
    inject_cot_instruction,
    parse_thinking,
)
from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult, TokenLogProb
from llm_agents.tools.base import Tool


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
    """A mock model that also returns log-prob data."""

    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, prompt: str, **kwargs) -> CompletionResult:
        token = TokenLogProb(token="hello", logprob=-0.1, rank=0)
        lp = LogProbResult(prompt=prompt, tokens=[token])
        return CompletionResult(text=self._text, logprob_result=lp)

    def get_logprobs(self, prompt: str, **kwargs) -> LogProbResult:
        return LogProbResult(prompt=prompt)


def _make_tool(name: str = "echo", return_value: str = "echoed") -> Tool:
    return Tool(
        name=name,
        description=f"A test tool called {name}.",
        parameters_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=lambda text: return_value,
    )


def _tool_call_json(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments})


# ---------------------------------------------------------------------------
# Chain-of-thought module
# ---------------------------------------------------------------------------


class TestInjectCotInstruction:
    def test_appends_instruction(self):
        result = inject_cot_instruction("You are helpful.")
        assert result.startswith("You are helpful.")
        assert "<think>" in result

    def test_idempotent_content(self):
        result = inject_cot_instruction("Base prompt")
        assert result == "Base prompt" + COT_INSTRUCTION


class TestParseThinking:
    def test_no_thinking(self):
        visible, thinking = parse_thinking("Just a normal answer.")
        assert visible == "Just a normal answer."
        assert thinking == ""

    def test_single_think_block(self):
        text = "<think>Let me reason about this.</think>The answer is 42."
        visible, thinking = parse_thinking(text)
        assert visible == "The answer is 42."
        assert thinking == "Let me reason about this."

    def test_multiple_think_blocks(self):
        text = "<think>Step 1</think>Partial.<think>Step 2</think>Final."
        visible, thinking = parse_thinking(text)
        assert visible == "Partial.Final."
        assert "Step 1" in thinking
        assert "Step 2" in thinking

    def test_multiline_thinking(self):
        text = "<think>\nLine one.\nLine two.\n</think>Answer."
        visible, thinking = parse_thinking(text)
        assert visible == "Answer."
        assert "Line one." in thinking
        assert "Line two." in thinking

    def test_only_thinking(self):
        text = "<think>All reasoning, no answer.</think>"
        visible, thinking = parse_thinking(text)
        assert visible == ""
        assert thinking == "All reasoning, no answer."


# ---------------------------------------------------------------------------
# Tool-call extraction
# ---------------------------------------------------------------------------


class TestExtractToolCall:
    def test_fenced_tool_call(self):
        text = 'Some text\n```tool_call\n{"name": "calc", "arguments": {"expr": "1+1"}}\n```'
        result = _extract_tool_call(text)
        assert result is not None
        assert result["name"] == "calc"
        assert result["arguments"]["expr"] == "1+1"

    def test_bare_json_tool_call(self):
        text = 'I will call: {"name": "search", "arguments": {"query": "test"}}'
        result = _extract_tool_call(text)
        assert result is not None
        assert result["name"] == "search"

    def test_no_tool_call(self):
        assert _extract_tool_call("Just a normal response.") is None

    def test_invalid_json(self):
        assert _extract_tool_call('```tool_call\n{bad json}\n```') is None

    def test_missing_arguments_key(self):
        text = '{"name": "tool_without_args"}'
        assert _extract_tool_call(text) is None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_basic_prompt(self):
        prompt = _build_prompt("You are helpful.", [], "")
        assert "System: You are helpful." in prompt
        assert prompt.endswith("Assistant:")

    def test_with_history(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        prompt = _build_prompt("System msg.", history, "")
        assert "User: Hi" in prompt
        assert "Assistant: Hello!" in prompt

    def test_with_tool_prompt(self):
        prompt = _build_prompt("Base.", [], "Available tools: calc")
        assert "Available tools: calc" in prompt


# ---------------------------------------------------------------------------
# Agent — basic conversation
# ---------------------------------------------------------------------------


class TestAgentBasic:
    def test_simple_response(self):
        model = MockModel(["Hello there!"])
        agent = Agent(name="test", model=model, enable_cot=False)
        resp = agent.run("Hi")
        assert isinstance(resp, AgentResponse)
        assert resp.content == "Hello there!"
        assert resp.tool_calls == []
        assert resp.thinking == ""

    def test_history_is_maintained(self):
        model = MockModel(["First reply", "Second reply"])
        agent = Agent(name="test", model=model, enable_cot=False)
        agent.run("Message 1")
        agent.run("Message 2")
        history = agent.history
        assert len(history) == 4  # 2 user + 2 assistant
        assert history[0]["content"] == "Message 1"
        assert history[1]["content"] == "First reply"

    def test_reset_clears_history(self):
        model = MockModel(["Reply"])
        agent = Agent(name="test", model=model, enable_cot=False)
        agent.run("Hi")
        assert len(agent.history) > 0
        agent.reset()
        assert len(agent.history) == 0


# ---------------------------------------------------------------------------
# Agent — chain-of-thought
# ---------------------------------------------------------------------------


class TestAgentCoT:
    def test_cot_strips_thinking_from_content(self):
        model = MockModel(["<think>Reasoning here.</think>The answer is 42."])
        agent = Agent(name="test", model=model, enable_cot=True)
        resp = agent.run("What is the answer?")
        assert resp.content == "The answer is 42."
        assert resp.thinking == "Reasoning here."

    def test_cot_disabled_keeps_think_tags(self):
        raw = "<think>Should stay.</think>Visible."
        model = MockModel([raw])
        agent = Agent(name="test", model=model, enable_cot=False)
        resp = agent.run("Q")
        assert "<think>" in resp.content

    def test_cot_with_no_think_blocks(self):
        model = MockModel(["No thinking at all."])
        agent = Agent(name="test", model=model, enable_cot=True)
        resp = agent.run("Q")
        assert resp.content == "No thinking at all."
        assert resp.thinking == ""


# ---------------------------------------------------------------------------
# Agent — tool use
# ---------------------------------------------------------------------------


class TestAgentToolUse:
    def test_single_tool_call(self):
        tool_call = _tool_call_json("echo", {"text": "hello"})
        # First response triggers tool call; second is the final answer.
        model = MockModel([tool_call, "Done!"])
        echo_tool = _make_tool("echo", return_value="echoed: hello")
        agent = Agent(
            name="test", model=model, tools=[echo_tool], enable_cot=False
        )
        resp = agent.run("Call echo")
        assert resp.content == "Done!"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "echo"
        assert resp.tool_calls[0].result == "echoed: hello"

    def test_fenced_tool_call(self):
        fenced = '```tool_call\n{"name": "echo", "arguments": {"text": "hi"}}\n```'
        model = MockModel([fenced, "Result received."])
        echo_tool = _make_tool("echo", return_value="hi back")
        agent = Agent(
            name="test", model=model, tools=[echo_tool], enable_cot=False
        )
        resp = agent.run("Call echo fenced")
        assert resp.content == "Result received."
        assert len(resp.tool_calls) == 1

    def test_tool_error_is_fed_back(self):
        tool_call = _tool_call_json("nonexistent", {"text": "x"})
        model = MockModel([tool_call, "I see the error."])
        echo_tool = _make_tool("echo")
        agent = Agent(
            name="test", model=model, tools=[echo_tool], enable_cot=False
        )
        resp = agent.run("Call bad tool")
        assert len(resp.tool_calls) == 1
        assert "Error" in resp.tool_calls[0].result

    def test_multiple_tool_rounds(self):
        call1 = _tool_call_json("echo", {"text": "a"})
        call2 = _tool_call_json("echo", {"text": "b"})
        model = MockModel([call1, call2, "All done."])
        echo_tool = _make_tool("echo", return_value="ok")
        agent = Agent(
            name="test", model=model, tools=[echo_tool], enable_cot=False
        )
        resp = agent.run("Do two calls")
        assert resp.content == "All done."
        assert len(resp.tool_calls) == 2

    def test_max_tool_rounds_enforced(self):
        # Model always returns a tool call.
        call = _tool_call_json("echo", {"text": "loop"})
        model = MockModel([call])
        echo_tool = _make_tool("echo", return_value="ok")
        agent = Agent(
            name="test",
            model=model,
            tools=[echo_tool],
            enable_cot=False,
            max_tool_rounds=3,
        )
        resp = agent.run("Infinite loop")
        assert len(resp.tool_calls) == 3

    def test_tool_call_with_cot(self):
        thinking_call = (
            '<think>I need to use echo.</think>'
            + _tool_call_json("echo", {"text": "hi"})
        )
        model = MockModel([thinking_call, "<think>Got result.</think>Final answer."])
        echo_tool = _make_tool("echo", return_value="echoed")
        agent = Agent(
            name="test", model=model, tools=[echo_tool], enable_cot=True
        )
        resp = agent.run("Think and call")
        assert resp.content == "Final answer."
        assert "I need to use echo" in resp.thinking
        assert "Got result" in resp.thinking
        assert len(resp.tool_calls) == 1


# ---------------------------------------------------------------------------
# Agent — logprobs
# ---------------------------------------------------------------------------


class TestAgentLogprobs:
    def test_logprobs_returned(self):
        model = MockModelWithLogprobs("Simple answer.")
        agent = Agent(name="test", model=model, enable_cot=False)
        resp = agent.run("Q")
        assert resp.logprobs is not None
        assert len(resp.logprobs.tokens) == 1
        assert resp.logprobs.tokens[0].token == "hello"

    def test_logprobs_none_when_absent(self):
        model = MockModel(["No logprobs here."])
        agent = Agent(name="test", model=model, enable_cot=False)
        resp = agent.run("Q")
        assert resp.logprobs is None


# ---------------------------------------------------------------------------
# AgentResponse & ToolCallRecord dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_agent_response_defaults(self):
        resp = AgentResponse(content="hi")
        assert resp.content == "hi"
        assert resp.thinking == ""
        assert resp.tool_calls == []
        assert resp.logprobs is None

    def test_tool_call_record(self):
        rec = ToolCallRecord(name="t", arguments={"a": 1}, result="ok")
        assert rec.name == "t"
        assert rec.arguments == {"a": 1}
        assert rec.result == "ok"
