"""Workflow loading and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_agents.workflows.schema import Workflow, parse_workflow


def load_workflow(path: str | Path) -> Workflow:
    """Load a workflow from a YAML or JSON file.

    Supports ``.yaml``, ``.yml``, and ``.json`` extensions.

    Args:
        path: Path to the workflow definition file.

    Returns:
        A :class:`Workflow` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or parsing fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {path}")

    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML workflow files. "
                "Install with: pip install pyyaml"
            )
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported workflow file format: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Workflow file must contain a mapping at the top level")

    return parse_workflow(data)


def validate_workflow(workflow: Workflow) -> list[str]:
    """Validate a workflow for common errors.

    Checks for:
    - Missing agent definitions referenced in steps.
    - Steps without agent or input.
    - Circular variable references (simple check).

    Args:
        workflow: The workflow to validate.

    Returns:
        A list of error messages. An empty list means the workflow is valid.
    """
    errors: list[str] = []

    if not workflow.name:
        errors.append("Workflow must have a name")

    if not workflow.steps:
        errors.append("Workflow must have at least one step")

    defined_agents = set(workflow.agents.keys())
    defined_outputs: set[str] = set(workflow.variables.keys())

    for i, step in enumerate(workflow.steps):
        step_label = f"Step {i + 1}"

        if step.step_type in ("sequential",) and not step.agent:
            errors.append(f"{step_label}: missing agent name")

        if step.agent and step.agent not in defined_agents:
            errors.append(
                f"{step_label}: agent {step.agent!r} is not defined in workflow"
            )

        if step.step_type == "conditional" and not step.condition:
            errors.append(f"{step_label}: conditional step has no condition")

        if step.step_type == "conditional" and not step.branches:
            errors.append(f"{step_label}: conditional step has no branches")

        if step.output:
            defined_outputs.add(step.output)

    return errors
