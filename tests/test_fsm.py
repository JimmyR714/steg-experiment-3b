"""Tests for Extension 16: Agent State Machines."""

from __future__ import annotations

import re

import pytest

from llm_agents.agents.fsm import (
    State,
    StateMachineAgent,
    Transition,
    fsm_to_mermaid,
)
from llm_agents.models.types import CompletionResult, LogProbResult


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class _SequenceModel:
    """Model that returns a sequence of pre-defined responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def generate(self, prompt, **kwargs):
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return CompletionResult(text=text)

    def get_logprobs(self, prompt, **kwargs):
        return LogProbResult(prompt=prompt)


# ---------------------------------------------------------------------------
# Transition tests
# ---------------------------------------------------------------------------


class TestTransition:
    def test_string_regex_match(self):
        t = Transition(target_state="next", condition="hello")
        assert t.matches("say hello world")
        assert not t.matches("goodbye")

    def test_compiled_regex(self):
        t = Transition(target_state="next", condition=re.compile(r"\d+"))
        assert t.matches("code 42")
        assert not t.matches("no numbers")

    def test_callable_condition(self):
        t = Transition(target_state="next", condition=lambda out: "yes" in out.lower())
        assert t.matches("Yes, I agree")
        assert not t.matches("No way")

    def test_tool_name_match(self):
        t = Transition(target_state="next", condition="search")
        assert t.matches("some output", tool_name="search")


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


class TestState:
    def test_basic_state(self):
        s = State(name="start", prompt="You are starting.")
        assert s.name == "start"
        assert s.is_terminal is False
        assert s.transitions == []

    def test_terminal_state(self):
        s = State(name="end", prompt="Done.", is_terminal=True)
        assert s.is_terminal


# ---------------------------------------------------------------------------
# StateMachineAgent tests
# ---------------------------------------------------------------------------


class TestStateMachineAgent:
    def test_single_state_terminal(self):
        model = _SequenceModel(["Hello!"])
        states = [
            State(name="greet", prompt="Say hello.", is_terminal=True),
        ]
        fsm = StateMachineAgent(states, "greet", model)
        response = fsm.run("Hi")
        assert response.content == "Hello!"
        assert fsm.current_state == "greet"

    def test_state_transition(self):
        model = _SequenceModel(["issue detected", "fix applied"])
        states = [
            State(
                name="diagnose",
                prompt="Diagnose the issue.",
                transitions=[
                    Transition(target_state="fix", condition="issue"),
                ],
            ),
            State(name="fix", prompt="Fix the issue.", is_terminal=True),
        ]
        fsm = StateMachineAgent(states, "diagnose", model)
        response = fsm.run("Something is broken")
        assert fsm.current_state == "fix"
        assert len(fsm.transition_history) == 1
        assert fsm.transition_history[0] == ("diagnose", "fix")

    def test_no_transition_stops(self):
        model = _SequenceModel(["unrelated output"])
        states = [
            State(
                name="start",
                prompt="Check.",
                transitions=[
                    Transition(target_state="end", condition="TRIGGER"),
                ],
            ),
            State(name="end", prompt="End.", is_terminal=True),
        ]
        fsm = StateMachineAgent(states, "start", model)
        response = fsm.run("test")
        assert fsm.current_state == "start"  # No transition fired

    def test_priority_ordering(self):
        model = _SequenceModel(["matched both"])
        states = [
            State(
                name="start",
                prompt="Check.",
                transitions=[
                    Transition(target_state="low", condition="matched", priority=1),
                    Transition(target_state="high", condition="matched", priority=10),
                ],
            ),
            State(name="low", prompt="Low.", is_terminal=True),
            State(name="high", prompt="High.", is_terminal=True),
        ]
        fsm = StateMachineAgent(states, "start", model)
        fsm.run("test")
        assert fsm.current_state == "high"

    def test_invalid_initial_state(self):
        with pytest.raises(ValueError, match="not found"):
            StateMachineAgent([], "nonexistent", _SequenceModel([""]))

    def test_reset(self):
        model = _SequenceModel(["issue found", "fixed"])
        states = [
            State(
                name="check",
                prompt="Check.",
                transitions=[
                    Transition(target_state="done", condition="issue"),
                ],
            ),
            State(name="done", prompt="Done.", is_terminal=True),
        ]
        fsm = StateMachineAgent(states, "check", model)
        fsm.run("test")
        assert fsm.current_state == "done"
        fsm.reset()
        assert fsm.current_state == "check"
        assert fsm.transition_history == []

    def test_get_state(self):
        states = [State(name="a", prompt="A"), State(name="b", prompt="B")]
        fsm = StateMachineAgent(states, "a", _SequenceModel([""]))
        assert fsm.get_state("a") is not None
        assert fsm.get_state("c") is None


# ---------------------------------------------------------------------------
# fsm_to_mermaid tests
# ---------------------------------------------------------------------------


class TestFsmToMermaid:
    def test_generates_mermaid(self):
        states = [
            State(
                name="start",
                prompt="Begin here.",
                transitions=[Transition(target_state="end", condition="done")],
            ),
            State(name="end", prompt="Finished.", is_terminal=True),
        ]
        fsm = StateMachineAgent(states, "start", _SequenceModel([""]))
        mermaid = fsm_to_mermaid(fsm)
        assert "stateDiagram-v2" in mermaid
        assert "[*] --> start" in mermaid
        assert "start --> end" in mermaid
        assert "end --> [*]" in mermaid
