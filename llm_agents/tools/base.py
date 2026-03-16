"""Tool definition and decorator for registering callable tools."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    """A tool that can be called by an LLM agent.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters_schema: JSON-schema dict describing the expected parameters.
        fn: The underlying Python callable.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    fn: Callable[..., str]


def _build_schema_from_annotations(fn: Callable) -> dict[str, Any]:
    """Infer a JSON-schema-style dict from a function's type annotations."""
    sig = inspect.signature(fn)
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:
        hints = {}
    type_map: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        annotation = hints.get(param_name, param.annotation)
        json_type = type_map.get(annotation, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def tool(
    name: str | None = None,
    description: str = "",
    parameters_schema: dict[str, Any] | None = None,
) -> Callable[[Callable[..., str]], Tool]:
    """Decorator that wraps a function into a :class:`Tool` instance.

    Args:
        name: Tool name. Defaults to the function name.
        description: Tool description. Defaults to the function docstring.
        parameters_schema: Explicit JSON-schema for parameters. If *None*,
            a schema is inferred from the function's type annotations.

    Returns:
        A decorator that produces a ``Tool``.
    """

    def decorator(fn: Callable[..., str]) -> Tool:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip()
        schema = parameters_schema or _build_schema_from_annotations(fn)
        return Tool(
            name=tool_name,
            description=tool_desc,
            parameters_schema=schema,
            fn=fn,
        )

    return decorator
