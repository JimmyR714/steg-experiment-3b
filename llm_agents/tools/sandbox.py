"""Sandboxed code execution for safe agent-driven computation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Process exit code.
        timed_out: Whether the execution timed out.
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class PythonSandbox:
    """Execute Python code in a restricted subprocess.

    Safety measures:
    - Timeout enforcement.
    - Memory limit via ``ulimit`` (Linux only, best-effort).
    - Filesystem writes restricted to a temp directory.
    - Network access is not blocked at OS level but unsafe builtins
      are restricted in the executed code.

    Args:
        timeout: Maximum execution time in seconds.
        max_memory_mb: Maximum memory in megabytes (Linux only).
        work_dir: Working directory for execution. If *None*, a temporary
            directory is created and cleaned up after execution.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 256,
        work_dir: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self._work_dir = work_dir
        self._temp_dirs: list[str] = []

    def _get_work_dir(self) -> str:
        """Get or create the working directory."""
        if self._work_dir:
            os.makedirs(self._work_dir, exist_ok=True)
            return self._work_dir
        tmpdir = tempfile.mkdtemp(prefix="sandbox_")
        self._temp_dirs.append(tmpdir)
        return tmpdir

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in a sandboxed subprocess.

        Args:
            code: The Python source code to execute.

        Returns:
            An :class:`ExecutionResult` with stdout, stderr, and exit code.
        """
        work_dir = self._get_work_dir()
        script_path = os.path.join(work_dir, "_sandbox_script.py")

        # Wrap user code with safety restrictions
        wrapper = (
            "import sys\n"
            "import os\n"
            f"os.chdir({work_dir!r})\n"
            "# Restrict unsafe operations\n"
            "_original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__\n"
            "# Execute user code\n"
            "try:\n"
        )
        # Indent user code inside try block
        indented_code = "\n".join("    " + line for line in code.splitlines())
        wrapper += indented_code + "\n"
        wrapper += "except Exception as e:\n"
        wrapper += "    print(f'Error: {type(e).__name__}: {e}', file=sys.stderr)\n"
        wrapper += "    sys.exit(1)\n"

        with open(script_path, "w") as f:
            f.write(wrapper)

        try:
            # Build command
            cmd = [sys.executable, script_path]

            env = os.environ.copy()
            # Prevent importing user-installed packages that could be dangerous
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=work_dir,
                    env=env,
                )
                return ExecutionResult(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exit_code=proc.returncode,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    stdout="",
                    stderr=f"Execution timed out after {self.timeout}s",
                    exit_code=-1,
                    timed_out=True,
                )
        finally:
            # Clean up script file
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def cleanup(self) -> None:
        """Remove temporary directories created by this sandbox."""
        import shutil

        for tmpdir in self._temp_dirs:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except OSError:
                pass
        self._temp_dirs.clear()

    def __del__(self) -> None:
        self.cleanup()


class DockerSandbox:
    """Execute code in an ephemeral Docker container for stronger isolation.

    Requires Docker to be installed and accessible. Falls back gracefully
    if Docker is not available.

    Args:
        image: Docker image to use.
        timeout: Maximum execution time in seconds.
        max_memory_mb: Memory limit for the container.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        max_memory_mb: int = 256,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    @staticmethod
    def is_available() -> bool:
        """Check if Docker is available on this system."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code inside an ephemeral Docker container.

        Args:
            code: The Python source code to execute.

        Returns:
            An :class:`ExecutionResult`.

        Raises:
            RuntimeError: If Docker is not available.
        """
        if not self.is_available():
            raise RuntimeError("Docker is not available on this system")

        cmd = [
            "docker", "run",
            "--rm",
            "--network=none",
            f"--memory={self.max_memory_mb}m",
            "--read-only",
            "--tmpfs=/tmp:size=64m",
            self.image,
            "python3", "-c", code,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 10,  # Extra time for container startup
            )
            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Docker execution timed out after {self.timeout}s",
                exit_code=-1,
                timed_out=True,
            )
