"""Sandbox lifecycle management and built-in execution tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_agents.tools.base import Tool
from llm_agents.tools.sandbox import DockerSandbox, ExecutionResult, PythonSandbox


@dataclass
class ResourceUsage:
    """Tracks resource usage for an agent's sandbox sessions.

    Attributes:
        agent_name: Name of the agent.
        total_executions: Number of code executions.
        total_timeouts: Number of executions that timed out.
        total_errors: Number of executions that returned non-zero exit codes.
    """

    agent_name: str
    total_executions: int = 0
    total_timeouts: int = 0
    total_errors: int = 0


class SandboxManager:
    """Manages sandbox instances and provides built-in execution tools.

    Creates and tracks :class:`PythonSandbox` instances per agent,
    handles cleanup, and records resource usage.

    Args:
        timeout: Default timeout for sandbox executions.
        max_memory_mb: Default memory limit.
        use_docker: If *True* and Docker is available, use Docker sandboxes.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 256,
        use_docker: bool = False,
    ) -> None:
        self._timeout = timeout
        self._max_memory_mb = max_memory_mb
        self._use_docker = use_docker and DockerSandbox.is_available()
        self._sandboxes: dict[str, PythonSandbox] = {}
        self._usage: dict[str, ResourceUsage] = {}

    def _get_sandbox(self, agent_name: str) -> PythonSandbox:
        """Get or create a sandbox for the given agent."""
        if agent_name not in self._sandboxes:
            self._sandboxes[agent_name] = PythonSandbox(
                timeout=self._timeout,
                max_memory_mb=self._max_memory_mb,
            )
            self._usage[agent_name] = ResourceUsage(agent_name=agent_name)
        return self._sandboxes[agent_name]

    def _record_usage(self, agent_name: str, result: ExecutionResult) -> None:
        """Update resource usage tracking."""
        usage = self._usage.get(agent_name)
        if usage is None:
            return
        usage.total_executions += 1
        if result.timed_out:
            usage.total_timeouts += 1
        if result.exit_code != 0:
            usage.total_errors += 1

    def execute_python(self, agent_name: str, code: str) -> str:
        """Execute Python code in the agent's sandbox.

        Args:
            agent_name: The name of the agent requesting execution.
            code: Python source code to execute.

        Returns:
            A string with the execution output or error message.
        """
        sandbox = self._get_sandbox(agent_name)
        result = sandbox.execute(code)
        self._record_usage(agent_name, result)

        if result.timed_out:
            return f"Execution timed out after {self._timeout}s"
        if result.exit_code != 0:
            output = result.stderr or result.stdout
            return f"Error (exit code {result.exit_code}):\n{output}"
        return result.stdout or "(no output)"

    def execute_shell(self, agent_name: str, command: str) -> str:
        """Execute a shell command in a sandboxed environment.

        Args:
            agent_name: The name of the agent requesting execution.
            command: Shell command to execute.

        Returns:
            A string with the command output or error message.
        """
        # Wrap shell command as Python subprocess call for sandboxing
        code = (
            "import subprocess\n"
            f"result = subprocess.run({command!r}, shell=True, capture_output=True, text=True, timeout=10)\n"
            "print(result.stdout)\n"
            "if result.stderr:\n"
            "    print('STDERR:', result.stderr)\n"
        )
        return self.execute_python(agent_name, code)

    def get_usage(self, agent_name: str) -> ResourceUsage | None:
        """Get resource usage for the given agent.

        Args:
            agent_name: The agent name to look up.

        Returns:
            A :class:`ResourceUsage` or *None* if the agent has no sandbox.
        """
        return self._usage.get(agent_name)

    def create_tools(self, agent_name: str) -> list[Tool]:
        """Create execution tools bound to a specific agent.

        Returns tools that can be registered in an agent's tool registry.

        Args:
            agent_name: The agent these tools will be bound to.

        Returns:
            A list containing ``execute_python`` and ``execute_shell`` tools.
        """
        manager = self

        def _execute_python(code: str) -> str:
            return manager.execute_python(agent_name, code)

        def _execute_shell(command: str) -> str:
            return manager.execute_shell(agent_name, command)

        python_tool = Tool(
            name="execute_python",
            description="Execute Python code and return the output. Use for calculations, data processing, or any computational task.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
            fn=_execute_python,
        )

        shell_tool = Tool(
            name="execute_shell",
            description="Execute a shell command and return the output.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["command"],
            },
            fn=_execute_shell,
        )

        return [python_tool, shell_tool]

    def cleanup(self) -> None:
        """Clean up all sandbox instances."""
        for sandbox in self._sandboxes.values():
            sandbox.cleanup()
        self._sandboxes.clear()

    def cleanup_agent(self, agent_name: str) -> None:
        """Clean up a specific agent's sandbox.

        Args:
            agent_name: The agent whose sandbox should be cleaned up.
        """
        sandbox = self._sandboxes.pop(agent_name, None)
        if sandbox:
            sandbox.cleanup()
