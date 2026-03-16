"""Tests for Extension 6: Workflow DSL & Declarative Pipelines."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from llm_agents.agents.agent import Agent
from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult
from llm_agents.workflows.engine import WorkflowEngine
from llm_agents.workflows.loader import load_workflow, validate_workflow
from llm_agents.workflows.schema import (
    AgentDef,
    Step,
    Workflow,
    interpolate,
    parse_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel(BaseModel):
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
# interpolate tests
# ---------------------------------------------------------------------------


class TestInterpolate:
    def test_basic(self):
        result = interpolate("Hello {{name}}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_vars(self):
        result = interpolate("{{a}} + {{b}}", {"a": "1", "b": "2"})
        assert result == "1 + 2"

    def test_unknown_var_preserved(self):
        result = interpolate("{{known}} and {{unknown}}", {"known": "X"})
        assert result == "X and {{unknown}}"

    def test_no_vars(self):
        result = interpolate("plain text", {"x": "y"})
        assert result == "plain text"


# ---------------------------------------------------------------------------
# parse_workflow tests
# ---------------------------------------------------------------------------


class TestParseWorkflow:
    def test_basic_workflow(self):
        data = {
            "workflow": {
                "name": "test_flow",
                "agents": {
                    "agent_a": {
                        "model": "gpt-4",
                        "system_prompt": "You are helpful.",
                        "tools": ["calculator"],
                    }
                },
                "steps": [
                    {
                        "agent": "agent_a",
                        "input": "{{task}}",
                        "output": "result",
                    }
                ],
                "output": "{{result}}",
            }
        }
        wf = parse_workflow(data)
        assert wf.name == "test_flow"
        assert "agent_a" in wf.agents
        assert wf.agents["agent_a"].model == "gpt-4"
        assert len(wf.steps) == 1
        assert wf.steps[0].agent == "agent_a"

    def test_empty_data(self):
        wf = parse_workflow({})
        assert wf.name == "unnamed"
        assert wf.steps == []


# ---------------------------------------------------------------------------
# validate_workflow tests
# ---------------------------------------------------------------------------


class TestValidateWorkflow:
    def test_valid_workflow(self):
        wf = Workflow(
            name="valid",
            agents={"a": AgentDef(name="a", model="m")},
            steps=[Step(agent="a", input="test", output="r")],
            output="{{r}}",
        )
        errors = validate_workflow(wf)
        assert errors == []

    def test_missing_name(self):
        wf = Workflow(name="", steps=[Step(agent="a")])
        errors = validate_workflow(wf)
        assert any("name" in e for e in errors)

    def test_no_steps(self):
        wf = Workflow(name="test", steps=[])
        errors = validate_workflow(wf)
        assert any("step" in e.lower() for e in errors)

    def test_undefined_agent(self):
        wf = Workflow(
            name="test",
            agents={},
            steps=[Step(agent="missing", input="x")],
        )
        errors = validate_workflow(wf)
        assert any("missing" in e for e in errors)

    def test_conditional_without_condition(self):
        wf = Workflow(
            name="test",
            agents={"a": AgentDef(name="a")},
            steps=[Step(step_type="conditional", branches={"yes": []})],
        )
        errors = validate_workflow(wf)
        assert any("condition" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# WorkflowEngine tests
# ---------------------------------------------------------------------------


class TestWorkflowEngine:
    def test_simple_pipeline(self):
        model = MockModel(["Research results here", "Summary of research"])
        engine = WorkflowEngine(models={"test_model": model})
        wf = Workflow(
            name="test",
            agents={
                "researcher": AgentDef(name="researcher", model="test_model"),
                "summarizer": AgentDef(name="summarizer", model="test_model"),
            },
            steps=[
                Step(agent="researcher", input="{{task}}", output="research"),
                Step(agent="summarizer", input="Summarize: {{research}}", output="summary"),
            ],
            output="{{summary}}",
        )
        result = engine.run(wf, initial_variables={"task": "Study AI"})
        assert result == "Summary of research"

    def test_variable_passthrough(self):
        model = MockModel(["Output A"])
        engine = WorkflowEngine(models={"m": model})
        wf = Workflow(
            name="test",
            agents={"a": AgentDef(name="a", model="m")},
            steps=[Step(agent="a", input="{{input}}", output="out")],
            output="{{out}}",
            variables={"input": "default_input"},
        )
        result = engine.run(wf)
        assert result == "Output A"

    def test_initial_variables_override(self):
        model = MockModel(["Done"])
        engine = WorkflowEngine(models={"m": model})
        wf = Workflow(
            name="test",
            agents={"a": AgentDef(name="a", model="m")},
            steps=[Step(agent="a", input="{{task}}", output="r")],
            output="{{r}}",
            variables={"task": "old"},
        )
        result = engine.run(wf, initial_variables={"task": "new"})
        assert result == "Done"


# ---------------------------------------------------------------------------
# load_workflow tests
# ---------------------------------------------------------------------------


class TestLoadWorkflow:
    def test_json_file(self):
        data = {
            "workflow": {
                "name": "json_test",
                "agents": {"a": {"model": "m"}},
                "steps": [{"agent": "a", "input": "test", "output": "r"}],
                "output": "{{r}}",
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            wf = load_workflow(path)
            assert wf.name == "json_test"
        finally:
            os.unlink(path)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_workflow("/nonexistent/workflow.json")

    def test_unsupported_format(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("not a workflow")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                load_workflow(path)
        finally:
            os.unlink(path)
