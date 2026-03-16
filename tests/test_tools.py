"""Tests for the tool system (Phase 5)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from llm_agents.tools.base import Tool, tool
from llm_agents.tools.builtin import (
    ALL_BUILTIN_TOOLS,
    calculator,
    read_file,
    web_search,
    write_file,
)
from llm_agents.tools.executor import execute_tool_call
from llm_agents.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Tool dataclass & decorator
# ---------------------------------------------------------------------------


class TestToolDecorator:
    def test_decorator_creates_tool(self):
        @tool(name="greet", description="Say hello.")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        assert greet.description == "Say hello."
        assert greet.fn("World") == "Hello, World!"

    def test_decorator_defaults_to_function_name(self):
        @tool()
        def my_tool(x: int) -> str:
            """Does stuff."""
            return str(x)

        assert my_tool.name == "my_tool"
        assert my_tool.description == "Does stuff."

    def test_inferred_schema(self):
        @tool(name="add")
        def add(a: int, b: int) -> str:
            return str(a + b)

        schema = add.parameters_schema
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert schema["properties"]["a"]["type"] == "integer"
        assert set(schema["required"]) == {"a", "b"}

    def test_optional_parameters(self):
        @tool(name="opt")
        def opt(a: str, b: str = "default") -> str:
            return a + b

        schema = opt.parameters_schema
        assert schema["required"] == ["a"]

    def test_explicit_schema_overrides_inference(self):
        custom_schema = {"type": "object", "properties": {"x": {"type": "number"}}}

        @tool(name="custom", parameters_schema=custom_schema)
        def custom(x: float) -> str:
            return str(x)

        assert custom.parameters_schema is custom_schema


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def _make_tool(self, name: str = "t") -> Tool:
        return Tool(
            name=name,
            description=f"Tool {name}",
            parameters_schema={"type": "object", "properties": {}},
            fn=lambda: "ok",
        )

    def test_register_and_get(self):
        reg = ToolRegistry()
        t = self._make_tool("echo")
        reg.register(t)
        assert reg.get("echo") is t

    def test_duplicate_raises(self):
        reg = ToolRegistry()
        reg.register(self._make_tool("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(self._make_tool("dup"))

    def test_get_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="Unknown tool"):
            reg.get("nope")

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(self._make_tool("a"))
        reg.register(self._make_tool("b"))
        names = [t.name for t in reg.list_tools()]
        assert names == ["a", "b"]

    def test_len_and_contains(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(self._make_tool("x"))
        assert len(reg) == 1
        assert "x" in reg
        assert "y" not in reg

    def test_to_system_prompt(self):
        reg = ToolRegistry()
        reg.register(self._make_tool("calc"))
        prompt = reg.to_system_prompt()
        assert "calc" in prompt
        assert "Available tools:" in prompt

    def test_to_system_prompt_empty(self):
        reg = ToolRegistry()
        assert reg.to_system_prompt() == ""

    def test_to_tool_definitions(self):
        reg = ToolRegistry()
        reg.register(self._make_tool("foo"))
        defs = reg.to_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "foo"

    def test_parse_tool_call_from_string(self):
        raw = json.dumps({"name": "calc", "arguments": {"expr": "1+1"}})
        parsed = ToolRegistry.parse_tool_call(raw)
        assert parsed["name"] == "calc"
        assert parsed["arguments"]["expr"] == "1+1"

    def test_parse_tool_call_from_dict(self):
        parsed = ToolRegistry.parse_tool_call({"name": "t", "arguments": {"a": 1}})
        assert parsed["name"] == "t"

    def test_parse_tool_call_missing_name(self):
        with pytest.raises(ValueError, match="name"):
            ToolRegistry.parse_tool_call({"arguments": {}})

    def test_parse_tool_call_bad_json(self):
        with pytest.raises(ValueError, match="Invalid"):
            ToolRegistry.parse_tool_call("{bad json")

    def test_parse_tool_call_no_arguments(self):
        parsed = ToolRegistry.parse_tool_call({"name": "t"})
        assert parsed["arguments"] == {}


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


class TestBuiltinTools:
    def test_calculator_basic(self):
        assert calculator.fn("2 + 3") == "5"

    def test_calculator_expression(self):
        assert calculator.fn("10 * (3 + 2)") == "50"

    def test_calculator_error(self):
        result = calculator.fn("undefined_var")
        assert result.startswith("Error:")

    def test_web_search_stub(self):
        result = web_search.fn("python")
        assert "stub" in result.lower()
        assert "python" in result

    def test_read_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello content")
            path = f.name
        try:
            assert read_file.fn(path) == "hello content"
        finally:
            os.unlink(path)

    def test_read_file_not_found(self):
        result = read_file.fn("/nonexistent/file.txt")
        assert "Error" in result

    def test_write_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            result = write_file.fn(path, "new content")
            assert "Successfully" in result
            with open(path) as f:
                assert f.read() == "new content"
        finally:
            os.unlink(path)

    def test_all_builtin_tools_list(self):
        assert len(ALL_BUILTIN_TOOLS) == 4
        names = {t.name for t in ALL_BUILTIN_TOOLS}
        assert names == {"calculator", "web_search", "read_file", "write_file"}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TestExecutor:
    def _make_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        for t in ALL_BUILTIN_TOOLS:
            reg.register(t)
        return reg

    def test_execute_calculator(self):
        reg = self._make_registry()
        result = execute_tool_call(reg, {"name": "calculator", "arguments": {"expression": "7 * 6"}})
        assert result == "42"

    def test_execute_from_json_string(self):
        reg = self._make_registry()
        raw = json.dumps({"name": "calculator", "arguments": {"expression": "1+1"}})
        assert execute_tool_call(reg, raw) == "2"

    def test_execute_unknown_tool(self):
        reg = self._make_registry()
        with pytest.raises(KeyError, match="Unknown tool"):
            execute_tool_call(reg, {"name": "nonexistent", "arguments": {}})

    def test_execute_missing_required_arg(self):
        reg = self._make_registry()
        with pytest.raises(ValueError, match="Missing required"):
            execute_tool_call(reg, {"name": "calculator", "arguments": {}})

    def test_execute_wrong_arg_type(self):
        reg = self._make_registry()
        with pytest.raises(ValueError, match="expected type"):
            execute_tool_call(reg, {"name": "calculator", "arguments": {"expression": 123}})

    def test_execute_web_search(self):
        reg = self._make_registry()
        result = execute_tool_call(reg, {"name": "web_search", "arguments": {"query": "test"}})
        assert "stub" in result.lower()

    def test_execute_read_write_roundtrip(self):
        reg = self._make_registry()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            execute_tool_call(
                reg,
                {"name": "write_file", "arguments": {"path": path, "content": "round-trip"}},
            )
            result = execute_tool_call(
                reg,
                {"name": "read_file", "arguments": {"path": path}},
            )
            assert result == "round-trip"
        finally:
            os.unlink(path)
