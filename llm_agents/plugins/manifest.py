"""Extension 18: Plugin manifest definition and validation.

Provides a YAML-based manifest format for describing plugins, their tools,
dependencies, and required permissions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Permission(Enum):
    """Permissions that a plugin can request."""

    NETWORK = "network"
    FILESYSTEM = "filesystem"
    SUBPROCESS = "subprocess"
    ENVIRONMENT = "environment"


@dataclass
class ToolDefinition:
    """A tool defined in a plugin manifest.

    Attributes:
        name: Tool name.
        module: Python module path containing the tool.
        function: Function name within the module.
        description: Human-readable description.
    """

    name: str
    module: str
    function: str
    description: str = ""


@dataclass
class PluginManifest:
    """Metadata describing a plugin.

    Attributes:
        name: Plugin name.
        version: Semantic version string.
        description: What the plugin does.
        tools: List of tool definitions.
        dependencies: Python package dependencies (e.g. ["requests>=2.28"]).
        permissions: Required permissions.
        author: Plugin author.
    """

    name: str
    version: str = "0.1.0"
    description: str = ""
    tools: list[ToolDefinition] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    permissions: list[Permission] = field(default_factory=list)
    author: str = ""

    def validate(self) -> list[str]:
        """Validate the manifest and return a list of error messages.

        Returns:
            List of validation errors. Empty if valid.
        """
        errors: list[str] = []

        if not self.name:
            errors.append("Plugin name is required.")
        if not self.version:
            errors.append("Plugin version is required.")

        for i, tool in enumerate(self.tools):
            if not tool.name:
                errors.append(f"Tool {i} is missing a name.")
            if not tool.module:
                errors.append(f"Tool '{tool.name}' is missing a module path.")
            if not tool.function:
                errors.append(f"Tool '{tool.name}' is missing a function name.")

        return errors


def parse_manifest(data: dict[str, Any]) -> PluginManifest:
    """Parse a manifest from a dict (e.g. loaded from YAML).

    Args:
        data: Dict with manifest fields.

    Returns:
        A PluginManifest instance.

    Raises:
        ValueError: If the manifest is invalid.
    """
    tools: list[ToolDefinition] = []
    for tool_data in data.get("tools", []):
        tools.append(
            ToolDefinition(
                name=tool_data.get("name", ""),
                module=tool_data.get("module", ""),
                function=tool_data.get("function", ""),
                description=tool_data.get("description", ""),
            )
        )

    permissions: list[Permission] = []
    for perm_str in data.get("permissions", []):
        try:
            permissions.append(Permission(perm_str))
        except ValueError:
            pass  # Ignore unknown permissions

    manifest = PluginManifest(
        name=data.get("name", ""),
        version=data.get("version", "0.1.0"),
        description=data.get("description", ""),
        tools=tools,
        dependencies=data.get("dependencies", []),
        permissions=permissions,
        author=data.get("author", ""),
    )

    errors = manifest.validate()
    if errors:
        raise ValueError(f"Invalid manifest: {'; '.join(errors)}")

    return manifest
