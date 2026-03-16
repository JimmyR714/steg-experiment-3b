"""Built-in tools available to agents."""

from __future__ import annotations

from llm_agents.tools.base import Tool, tool


@tool(name="calculator", description="Evaluate a mathematical expression.")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result as a string."""
    # Allow only safe math operations
    allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


@tool(
    name="web_search",
    description="Search the web for information (stub implementation).",
)
def web_search(query: str) -> str:
    """Stub web search that returns a placeholder result."""
    return f"[web_search stub] No results for: {query}"


@tool(name="read_file", description="Read the contents of a file.")
def read_file(path: str) -> str:
    """Read and return the contents of the file at *path*."""
    try:
        with open(path) as f:
            return f.read()
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool(name="write_file", description="Write content to a file.")
def write_file(path: str, content: str) -> str:
    """Write *content* to the file at *path*, creating or overwriting it."""
    try:
        with open(path, "w") as f:
            f.write(content)
    except Exception as exc:
        return f"Error writing file: {exc}"
    return f"Successfully wrote {len(content)} characters to {path}"


# Convenience list of all built-in tools
ALL_BUILTIN_TOOLS: list[Tool] = [calculator, web_search, read_file, write_file]  # type: ignore[list-item]
