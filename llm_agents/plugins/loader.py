"""Extension 18: Dynamic plugin and tool loading.

Discovers and loads tools from Python modules, directories, and packages.
Supports the ``@tool`` decorator for automatic discovery.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from llm_agents.plugins.manifest import PluginManifest, parse_manifest
from llm_agents.tools.base import Tool, tool


def load_plugin(path_or_module: str) -> list[Tool]:
    """Load tools from a Python file path or module name.

    Discovers all ``Tool`` instances in the module, including those
    created with the ``@tool`` decorator.

    Args:
        path_or_module: Either a filesystem path to a .py file or a
            dotted Python module name.

    Returns:
        List of discovered Tool instances.

    Raises:
        ImportError: If the module cannot be loaded.
        FileNotFoundError: If the file path doesn't exist.
    """
    if path_or_module.endswith(".py"):
        return _load_from_file(path_or_module)
    else:
        return _load_from_module(path_or_module)


def _load_from_file(file_path: str) -> list[Tool]:
    """Load tools from a Python file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Plugin file not found: {file_path}")

    module_name = f"_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return _extract_tools(module)


def _load_from_module(module_name: str) -> list[Tool]:
    """Load tools from an importable module."""
    module = importlib.import_module(module_name)
    return _extract_tools(module)


def _extract_tools(module: Any) -> list[Tool]:
    """Extract all Tool instances from a module."""
    tools: list[Tool] = []
    seen_names: set[str] = set()

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, Tool) and attr.name not in seen_names:
            tools.append(attr)
            seen_names.add(attr.name)

    return tools


def load_from_manifest(manifest: PluginManifest) -> list[Tool]:
    """Load tools described in a plugin manifest.

    Args:
        manifest: The plugin manifest describing tools to load.

    Returns:
        List of loaded Tool instances.

    Raises:
        ImportError: If a tool's module cannot be imported.
        AttributeError: If a tool's function cannot be found.
    """
    tools: list[Tool] = []

    for tool_def in manifest.tools:
        module = importlib.import_module(tool_def.module)
        fn = getattr(module, tool_def.function)

        if isinstance(fn, Tool):
            tools.append(fn)
        elif callable(fn):
            # Wrap the callable as a Tool
            from llm_agents.tools.base import _build_schema_from_annotations

            tools.append(
                Tool(
                    name=tool_def.name,
                    description=tool_def.description or (fn.__doc__ or "").strip(),
                    parameters_schema=_build_schema_from_annotations(fn),
                    fn=fn,
                )
            )

    return tools


class PluginDirectory:
    """Watches a directory for plugin files and loads tools from them.

    Args:
        path: Directory path to watch.
        pattern: Glob pattern for plugin files (default: ``*.py``).
    """

    def __init__(self, path: str, pattern: str = "*.py") -> None:
        self._path = Path(path)
        self._pattern = pattern
        self._loaded_files: set[str] = set()
        self._tools: list[Tool] = []

    def scan(self) -> list[Tool]:
        """Scan the directory for new plugin files and load them.

        Returns:
            List of all tools discovered (cumulative).
        """
        if not self._path.exists():
            return self._tools

        for file_path in sorted(self._path.glob(self._pattern)):
            str_path = str(file_path)
            if str_path in self._loaded_files:
                continue
            if file_path.name.startswith("_"):
                continue

            try:
                new_tools = load_plugin(str_path)
                self._tools.extend(new_tools)
                self._loaded_files.add(str_path)
            except Exception:
                # Skip files that fail to load
                pass

        return list(self._tools)

    def reload(self) -> list[Tool]:
        """Reload all plugins from scratch.

        Returns:
            Freshly loaded list of tools.
        """
        self._loaded_files.clear()
        self._tools.clear()
        return self.scan()

    @property
    def loaded_files(self) -> set[str]:
        """Return set of loaded file paths."""
        return set(self._loaded_files)

    @property
    def tools(self) -> list[Tool]:
        """Return currently loaded tools."""
        return list(self._tools)
