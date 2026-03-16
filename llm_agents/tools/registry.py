"""Tool registry for storing, formatting, and looking up tools."""

from __future__ import annotations

import json
from typing import Any

from llm_agents.tools.base import Tool


class ToolRegistry:
    """Stores tools and provides helpers for LLM integration.

    The registry can format tools into system-prompt text or into the
    JSON tool-definition list that chat-completion APIs expect, and can
    parse tool-call responses back into structured data.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool:
        """Return the tool with the given name.

        Raises:
            KeyError: If no tool with that name exists.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name!r}") from None

    def list_tools(self) -> list[Tool]:
        """Return all registered tools in insertion order."""
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_system_prompt(self) -> str:
        """Format all tools as a human-readable system-prompt block."""
        if not self._tools:
            return ""
        lines = ["Available tools:"]
        for t in self._tools.values():
            lines.append(f"\n- **{t.name}**: {t.description}")
            lines.append(f"  Parameters: {json.dumps(t.parameters_schema)}")
        return "\n".join(lines)

    def to_tool_definitions(self) -> list[dict[str, Any]]:
        """Format tools as JSON tool-definition dicts (OpenAI-style)."""
        definitions: list[dict[str, Any]] = []
        for t in self._tools.values():
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters_schema,
                    },
                }
            )
        return definitions

    @staticmethod
    def parse_tool_call(raw: str | dict[str, Any]) -> dict[str, Any]:
        """Parse a tool-call response into a structured dict.

        Accepts either a JSON string or an already-parsed dict.  The
        returned dict always has ``"name"`` and ``"arguments"`` keys.

        Raises:
            ValueError: If the input cannot be parsed or is missing
                required fields.
        """
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid tool-call JSON: {exc}") from exc
        else:
            parsed = raw

        if "name" not in parsed:
            raise ValueError("Tool-call must include a 'name' field")

        return {
            "name": parsed["name"],
            "arguments": parsed.get("arguments", {}),
        }
