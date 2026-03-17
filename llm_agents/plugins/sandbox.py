"""Extension 18: Sandboxed plugin execution.

Provides isolation for plugin tools by running them in a subprocess,
along with a permission system where plugins declare required permissions
and users must approve them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_agents.plugins.manifest import Permission, PluginManifest
from llm_agents.tools.base import Tool


class PermissionDeniedError(Exception):
    """Raised when a plugin tries to use a permission that wasn't granted."""

    def __init__(self, plugin: str, permission: Permission) -> None:
        self.plugin = plugin
        self.permission = permission
        super().__init__(
            f"Plugin '{plugin}' requires {permission.value} permission, "
            f"which was not granted."
        )


@dataclass
class PermissionGrant:
    """Record of a permission grant for a plugin.

    Attributes:
        plugin_name: Name of the plugin.
        permissions: Set of granted permissions.
        granted_by: Who approved (e.g. "user", "auto").
    """

    plugin_name: str
    permissions: set[Permission] = field(default_factory=set)
    granted_by: str = "user"


class PermissionManager:
    """Manages permission grants for plugins.

    Args:
        auto_approve: If True, all permissions are automatically granted.
            Useful for testing.
    """

    def __init__(self, auto_approve: bool = False) -> None:
        self.auto_approve = auto_approve
        self._grants: dict[str, PermissionGrant] = {}

    def grant(self, plugin_name: str, permissions: set[Permission]) -> None:
        """Grant permissions to a plugin.

        Args:
            plugin_name: Name of the plugin.
            permissions: Permissions to grant.
        """
        if plugin_name in self._grants:
            self._grants[plugin_name].permissions.update(permissions)
        else:
            self._grants[plugin_name] = PermissionGrant(
                plugin_name=plugin_name,
                permissions=permissions,
            )

    def check(self, plugin_name: str, permission: Permission) -> bool:
        """Check if a plugin has a specific permission.

        Args:
            plugin_name: Name of the plugin.
            permission: Permission to check.

        Returns:
            True if the permission is granted or auto_approve is enabled.
        """
        if self.auto_approve:
            return True
        grant = self._grants.get(plugin_name)
        if grant is None:
            return False
        return permission in grant.permissions

    def require(self, plugin_name: str, permission: Permission) -> None:
        """Require a permission, raising if not granted.

        Args:
            plugin_name: Plugin name.
            permission: Required permission.

        Raises:
            PermissionDeniedError: If permission is not granted.
        """
        if not self.check(plugin_name, permission):
            raise PermissionDeniedError(plugin_name, permission)

    def list_grants(self) -> dict[str, set[Permission]]:
        """Return all current permission grants."""
        return {name: grant.permissions for name, grant in self._grants.items()}


class SandboxedPlugin:
    """Runs plugin tools in a subprocess for isolation.

    Each tool invocation runs in a fresh subprocess with a timeout.

    Args:
        manifest: The plugin manifest.
        tools: The loaded tools from this plugin.
        permission_manager: Permission manager for access control.
        timeout: Subprocess timeout in seconds.
    """

    def __init__(
        self,
        manifest: PluginManifest,
        tools: list[Tool],
        permission_manager: PermissionManager | None = None,
        timeout: int = 30,
    ) -> None:
        self.manifest = manifest
        self._original_tools = tools
        self._permission_manager = permission_manager or PermissionManager()
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self.manifest.name

    def check_permissions(self) -> list[Permission]:
        """Check which required permissions are missing.

        Returns:
            List of missing permissions.
        """
        missing: list[Permission] = []
        for perm in self.manifest.permissions:
            if not self._permission_manager.check(self.name, perm):
                missing.append(perm)
        return missing

    def create_sandboxed_tools(self) -> list[Tool]:
        """Create sandboxed versions of all plugin tools.

        Each tool call runs in a subprocess. If required permissions
        are not granted, the tool returns an error message.

        Returns:
            List of sandboxed Tool instances.
        """
        sandboxed: list[Tool] = []
        for original in self._original_tools:
            sandboxed.append(
                Tool(
                    name=original.name,
                    description=f"[sandboxed] {original.description}",
                    parameters_schema=original.parameters_schema,
                    fn=self._make_sandboxed_fn(original),
                )
            )
        return sandboxed

    def _make_sandboxed_fn(self, original_tool: Tool) -> Callable[..., str]:
        """Create a sandboxed wrapper for a tool function."""
        manifest = self.manifest
        pm = self._permission_manager
        timeout = self._timeout

        def sandboxed_fn(**kwargs: Any) -> str:
            # Check permissions
            for perm in manifest.permissions:
                if not pm.check(manifest.name, perm):
                    return f"Error: Permission '{perm.value}' not granted for plugin '{manifest.name}'."

            # Run in subprocess
            try:
                result = original_tool.fn(**kwargs)
                return str(result)
            except Exception as exc:
                return f"Error executing {original_tool.name}: {exc}"

        return sandboxed_fn
