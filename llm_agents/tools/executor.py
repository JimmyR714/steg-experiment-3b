"""Tool-call execution with argument validation."""

from __future__ import annotations

import json
from typing import Any

from llm_agents.tools.registry import ToolRegistry


def _validate_args(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    """Validate *arguments* against a JSON-schema-style *schema*.

    Performs lightweight checks for required fields and basic type
    conformance.  This is intentionally not a full JSON-schema validator.

    Raises:
        ValueError: If validation fails.
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field in required:
        if field not in arguments:
            raise ValueError(f"Missing required argument: {field!r}")

    json_type_to_python: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": (int, float),  # type: ignore[dict-item]
        "boolean": bool,
    }

    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            continue
        expected_type_str = properties[arg_name].get("type")
        if expected_type_str is None:
            continue
        expected_type = json_type_to_python.get(expected_type_str)
        if expected_type is not None and not isinstance(arg_value, expected_type):
            raise ValueError(
                f"Argument {arg_name!r} expected type {expected_type_str!r}, "
                f"got {type(arg_value).__name__!r}"
            )


def execute_tool_call(
    registry: ToolRegistry,
    tool_call: str | dict[str, Any],
) -> str:
    """Validate arguments and execute a tool call.

    Args:
        registry: The tool registry to look up the tool in.
        tool_call: A JSON string or dict with ``"name"`` and ``"arguments"``
            keys.

    Returns:
        The string result produced by the tool function.

    Raises:
        KeyError: If the requested tool is not registered.
        ValueError: If the arguments fail schema validation or the
            tool-call cannot be parsed.
    """
    parsed = ToolRegistry.parse_tool_call(tool_call)
    name: str = parsed["name"]
    arguments: dict[str, Any] = parsed["arguments"]

    tool = registry.get(name)
    _validate_args(tool.parameters_schema, arguments)

    return tool.fn(**arguments)
