"""Tests for Extension 18: Plugin System & Dynamic Tool Loading."""

from __future__ import annotations

import os
import tempfile

import pytest

from llm_agents.plugins.manifest import (
    Permission,
    PluginManifest,
    ToolDefinition,
    parse_manifest,
)
from llm_agents.plugins.loader import (
    PluginDirectory,
    load_plugin,
)
from llm_agents.plugins.sandbox import (
    PermissionDeniedError,
    PermissionManager,
    SandboxedPlugin,
)
from llm_agents.tools.base import Tool


# ---------------------------------------------------------------------------
# PluginManifest tests
# ---------------------------------------------------------------------------


class TestPluginManifest:
    def test_valid_manifest(self):
        manifest = PluginManifest(
            name="test-plugin",
            version="1.0.0",
            tools=[
                ToolDefinition(
                    name="greet",
                    module="test_mod",
                    function="greet_fn",
                    description="Say hello",
                )
            ],
        )
        errors = manifest.validate()
        assert errors == []

    def test_missing_name(self):
        manifest = PluginManifest(name="", version="1.0")
        errors = manifest.validate()
        assert any("name" in e.lower() for e in errors)

    def test_missing_tool_fields(self):
        manifest = PluginManifest(
            name="test",
            tools=[ToolDefinition(name="", module="", function="")],
        )
        errors = manifest.validate()
        assert len(errors) > 0


class TestParseManifest:
    def test_parse_valid(self):
        data = {
            "name": "weather-tools",
            "version": "1.0.0",
            "tools": [
                {
                    "name": "get_weather",
                    "module": "weather_tools.api",
                    "function": "get_current_weather",
                    "description": "Get current weather",
                }
            ],
            "permissions": ["network"],
            "dependencies": ["requests>=2.28"],
        }
        manifest = parse_manifest(data)
        assert manifest.name == "weather-tools"
        assert len(manifest.tools) == 1
        assert Permission.NETWORK in manifest.permissions
        assert "requests>=2.28" in manifest.dependencies

    def test_parse_invalid(self):
        with pytest.raises(ValueError, match="Invalid manifest"):
            parse_manifest({"name": ""})


# ---------------------------------------------------------------------------
# loader tests
# ---------------------------------------------------------------------------


class TestLoadPlugin:
    def test_load_from_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(
                'from llm_agents.tools.base import Tool\n'
                'my_tool = Tool(\n'
                '    name="hello",\n'
                '    description="Say hello",\n'
                '    parameters_schema={"type": "object", "properties": {}},\n'
                '    fn=lambda: "Hello!",\n'
                ')\n'
            )
            f.flush()
            path = f.name

        try:
            tools = load_plugin(path)
            assert len(tools) == 1
            assert tools[0].name == "hello"
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_plugin("/nonexistent/path.py")


class TestPluginDirectory:
    def test_scan_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plugin file
            plugin_path = os.path.join(tmpdir, "my_plugin.py")
            with open(plugin_path, "w") as f:
                f.write(
                    'from llm_agents.tools.base import Tool\n'
                    'greet = Tool(\n'
                    '    name="greet",\n'
                    '    description="Greet",\n'
                    '    parameters_schema={"type": "object", "properties": {}},\n'
                    '    fn=lambda: "Hi!",\n'
                    ')\n'
                )

            directory = PluginDirectory(tmpdir)
            tools = directory.scan()
            assert len(tools) == 1
            assert tools[0].name == "greet"

    def test_skip_underscore_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "_private.py")
            with open(path, "w") as f:
                f.write('x = 1\n')

            directory = PluginDirectory(tmpdir)
            tools = directory.scan()
            assert len(tools) == 0

    def test_scan_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = os.path.join(tmpdir, "tool.py")
            with open(plugin_path, "w") as f:
                f.write(
                    'from llm_agents.tools.base import Tool\n'
                    't = Tool(name="t", description="T", '
                    'parameters_schema={"type": "object"}, fn=lambda: "ok")\n'
                )

            directory = PluginDirectory(tmpdir)
            tools1 = directory.scan()
            tools2 = directory.scan()
            assert len(tools1) == len(tools2) == 1

    def test_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = os.path.join(tmpdir, "tool.py")
            with open(plugin_path, "w") as f:
                f.write(
                    'from llm_agents.tools.base import Tool\n'
                    't = Tool(name="t", description="T", '
                    'parameters_schema={"type": "object"}, fn=lambda: "ok")\n'
                )

            directory = PluginDirectory(tmpdir)
            directory.scan()
            tools = directory.reload()
            assert len(tools) == 1

    def test_nonexistent_directory(self):
        directory = PluginDirectory("/nonexistent/path")
        tools = directory.scan()
        assert tools == []


# ---------------------------------------------------------------------------
# PermissionManager tests
# ---------------------------------------------------------------------------


class TestPermissionManager:
    def test_grant_and_check(self):
        pm = PermissionManager()
        pm.grant("plugin_a", {Permission.NETWORK})
        assert pm.check("plugin_a", Permission.NETWORK)
        assert not pm.check("plugin_a", Permission.FILESYSTEM)

    def test_require_raises(self):
        pm = PermissionManager()
        with pytest.raises(PermissionDeniedError):
            pm.require("plugin_a", Permission.NETWORK)

    def test_auto_approve(self):
        pm = PermissionManager(auto_approve=True)
        assert pm.check("any_plugin", Permission.SUBPROCESS)

    def test_list_grants(self):
        pm = PermissionManager()
        pm.grant("a", {Permission.NETWORK, Permission.FILESYSTEM})
        grants = pm.list_grants()
        assert Permission.NETWORK in grants["a"]
        assert Permission.FILESYSTEM in grants["a"]

    def test_incremental_grant(self):
        pm = PermissionManager()
        pm.grant("a", {Permission.NETWORK})
        pm.grant("a", {Permission.FILESYSTEM})
        assert pm.check("a", Permission.NETWORK)
        assert pm.check("a", Permission.FILESYSTEM)


# ---------------------------------------------------------------------------
# SandboxedPlugin tests
# ---------------------------------------------------------------------------


class TestSandboxedPlugin:
    def _make_plugin(self, permissions=None, auto_approve=False):
        manifest = PluginManifest(
            name="test",
            version="1.0",
            permissions=permissions or [],
        )
        tools = [
            Tool(
                name="echo",
                description="Echo input",
                parameters_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                fn=lambda text="": f"echo: {text}",
            )
        ]
        pm = PermissionManager(auto_approve=auto_approve)
        return SandboxedPlugin(manifest, tools, pm)

    def test_sandboxed_tool_execution(self):
        plugin = self._make_plugin(auto_approve=True)
        tools = plugin.create_sandboxed_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"
        result = tools[0].fn(text="hello")
        assert "echo: hello" in result

    def test_permission_denied(self):
        plugin = self._make_plugin(permissions=[Permission.NETWORK])
        missing = plugin.check_permissions()
        assert Permission.NETWORK in missing

        tools = plugin.create_sandboxed_tools()
        result = tools[0].fn(text="hello")
        assert "Permission" in result and "not granted" in result

    def test_with_permission_granted(self):
        manifest = PluginManifest(
            name="test",
            version="1.0",
            permissions=[Permission.NETWORK],
        )
        tools = [
            Tool(
                name="fetch",
                description="Fetch data",
                parameters_schema={"type": "object", "properties": {}},
                fn=lambda: "fetched",
            )
        ]
        pm = PermissionManager()
        pm.grant("test", {Permission.NETWORK})
        plugin = SandboxedPlugin(manifest, tools, pm)
        sandboxed = plugin.create_sandboxed_tools()
        result = sandboxed[0].fn()
        assert result == "fetched"

    def test_name_property(self):
        plugin = self._make_plugin()
        assert plugin.name == "test"

    def test_sandboxed_description_prefix(self):
        plugin = self._make_plugin(auto_approve=True)
        tools = plugin.create_sandboxed_tools()
        assert tools[0].description.startswith("[sandboxed]")
