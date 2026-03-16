"""Tests for Extension 8: Sandboxed Code Execution Tool."""

from __future__ import annotations

import pytest

from llm_agents.tools.sandbox import ExecutionResult, PythonSandbox
from llm_agents.tools.sandbox_manager import SandboxManager


# ---------------------------------------------------------------------------
# ExecutionResult tests
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_fields(self):
        r = ExecutionResult(stdout="hello", stderr="", exit_code=0)
        assert r.stdout == "hello"
        assert r.exit_code == 0
        assert r.timed_out is False

    def test_timed_out(self):
        r = ExecutionResult(stdout="", stderr="timeout", exit_code=-1, timed_out=True)
        assert r.timed_out is True


# ---------------------------------------------------------------------------
# PythonSandbox tests
# ---------------------------------------------------------------------------


class TestPythonSandbox:
    def test_simple_execution(self):
        sandbox = PythonSandbox(timeout=10)
        try:
            result = sandbox.execute("print('hello world')")
            assert "hello world" in result.stdout
            assert result.exit_code == 0
        finally:
            sandbox.cleanup()

    def test_math_computation(self):
        sandbox = PythonSandbox(timeout=10)
        try:
            result = sandbox.execute("print(2 + 2)")
            assert "4" in result.stdout
        finally:
            sandbox.cleanup()

    def test_error_handling(self):
        sandbox = PythonSandbox(timeout=10)
        try:
            result = sandbox.execute("raise ValueError('test error')")
            assert result.exit_code != 0
            assert "ValueError" in result.stderr or "Error" in result.stderr
        finally:
            sandbox.cleanup()

    def test_timeout(self):
        sandbox = PythonSandbox(timeout=2)
        try:
            result = sandbox.execute("import time; time.sleep(10)")
            assert result.timed_out is True
            assert result.exit_code == -1
        finally:
            sandbox.cleanup()

    def test_multiline_code(self):
        sandbox = PythonSandbox(timeout=10)
        try:
            code = (
                "x = 10\n"
                "y = 20\n"
                "print(x + y)"
            )
            result = sandbox.execute(code)
            assert "30" in result.stdout
        finally:
            sandbox.cleanup()

    def test_import_standard_library(self):
        sandbox = PythonSandbox(timeout=10)
        try:
            result = sandbox.execute("import json; print(json.dumps({'a': 1}))")
            assert '"a"' in result.stdout
        finally:
            sandbox.cleanup()


# ---------------------------------------------------------------------------
# SandboxManager tests
# ---------------------------------------------------------------------------


class TestSandboxManager:
    def test_execute_python(self):
        manager = SandboxManager(timeout=10)
        try:
            output = manager.execute_python("test_agent", "print(42)")
            assert "42" in output
        finally:
            manager.cleanup()

    def test_execute_python_error(self):
        manager = SandboxManager(timeout=10)
        try:
            output = manager.execute_python("test_agent", "raise RuntimeError('boom')")
            assert "Error" in output
        finally:
            manager.cleanup()

    def test_resource_tracking(self):
        manager = SandboxManager(timeout=10)
        try:
            manager.execute_python("agent_a", "print(1)")
            manager.execute_python("agent_a", "raise ValueError('x')")
            usage = manager.get_usage("agent_a")
            assert usage is not None
            assert usage.total_executions == 2
            assert usage.total_errors == 1
        finally:
            manager.cleanup()

    def test_create_tools(self):
        manager = SandboxManager(timeout=10)
        try:
            tools = manager.create_tools("test_agent")
            assert len(tools) == 2
            names = {t.name for t in tools}
            assert "execute_python" in names
            assert "execute_shell" in names
        finally:
            manager.cleanup()

    def test_tool_execution(self):
        manager = SandboxManager(timeout=10)
        try:
            tools = manager.create_tools("test_agent")
            python_tool = next(t for t in tools if t.name == "execute_python")
            result = python_tool.fn(code="print('from tool')")
            assert "from tool" in result
        finally:
            manager.cleanup()

    def test_cleanup_agent(self):
        manager = SandboxManager(timeout=10)
        manager.execute_python("agent_x", "print(1)")
        assert manager.get_usage("agent_x") is not None
        manager.cleanup_agent("agent_x")
        # Sandbox removed but usage record might persist
        assert "agent_x" not in manager._sandboxes

    def test_no_usage_for_unknown_agent(self):
        manager = SandboxManager()
        assert manager.get_usage("nonexistent") is None
