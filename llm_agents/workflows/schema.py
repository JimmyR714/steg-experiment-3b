"""Workflow schema definitions for declarative multi-agent pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentDef:
    """Definition of an agent within a workflow.

    Attributes:
        name: Unique identifier for this agent within the workflow.
        model: Model name or identifier.
        system_prompt: System prompt for the agent.
        tools: List of tool names the agent should have access to.
    """

    name: str
    model: str = ""
    system_prompt: str = "You are a helpful assistant."
    tools: list[str] = field(default_factory=list)


@dataclass
class Step:
    """A single step in a workflow pipeline.

    Attributes:
        agent: Name of the agent to execute this step.
        input: Input template with ``{{variable}}`` interpolation.
        output: Name of the variable to store the result in.
        step_type: One of ``"sequential"``, ``"parallel"``, ``"conditional"``,
            or ``"loop"``.
        condition: For conditional steps, the condition to evaluate.
        branches: For conditional steps, mapping of condition values to
            sub-step lists.
        items: For loop steps, the variable name holding items to iterate.
        max_iterations: For loop steps, maximum number of iterations.
        steps: For parallel/composite steps, nested sub-steps.
    """

    agent: str = ""
    input: str = ""
    output: str = ""
    step_type: str = "sequential"
    condition: str = ""
    branches: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    items: str = ""
    max_iterations: int = 10
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Workflow:
    """A complete workflow definition.

    Attributes:
        name: Human-readable name for the workflow.
        agents: Mapping of agent name to :class:`AgentDef`.
        steps: Ordered list of :class:`Step` objects.
        output: Output template for the final result.
        variables: Initial variables available to the workflow.
    """

    name: str
    agents: dict[str, AgentDef] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    output: str = ""
    variables: dict[str, Any] = field(default_factory=dict)


_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def interpolate(template: str, variables: dict[str, Any]) -> str:
    """Replace ``{{variable}}`` placeholders with values from *variables*.

    Unknown variables are left as-is.

    Args:
        template: A string containing ``{{variable}}`` placeholders.
        variables: Mapping of variable names to their values.

    Returns:
        The interpolated string.
    """

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        if name in variables:
            return str(variables[name])
        return match.group(0)

    return _VAR_PATTERN.sub(_replace, template)


def parse_workflow(data: dict[str, Any]) -> Workflow:
    """Parse a raw dict (from YAML/JSON) into a :class:`Workflow`.

    Expected structure::

        workflow:
          name: ...
          agents:
            agent_name:
              model: ...
              tools: [...]
              system_prompt: ...
          steps:
            - agent: agent_name
              input: "{{variable}}"
              output: result_name
          output: "{{result_name}}"

    Args:
        data: The raw workflow dict.

    Returns:
        A :class:`Workflow` instance.

    Raises:
        ValueError: If the data is malformed.
    """
    wf_data = data.get("workflow", data)

    name = wf_data.get("name", "unnamed")

    agents: dict[str, AgentDef] = {}
    for agent_name, agent_conf in wf_data.get("agents", {}).items():
        if isinstance(agent_conf, dict):
            agents[agent_name] = AgentDef(
                name=agent_name,
                model=agent_conf.get("model", ""),
                system_prompt=agent_conf.get("system_prompt", "You are a helpful assistant."),
                tools=agent_conf.get("tools", []),
            )
        else:
            agents[agent_name] = AgentDef(name=agent_name)

    steps: list[Step] = []
    for step_data in wf_data.get("steps", []):
        step_type = step_data.get("type", "sequential")
        steps.append(
            Step(
                agent=step_data.get("agent", ""),
                input=step_data.get("input", ""),
                output=step_data.get("output", ""),
                step_type=step_type,
                condition=step_data.get("condition", ""),
                branches=step_data.get("branches", {}),
                items=step_data.get("items", ""),
                max_iterations=step_data.get("max_iterations", 10),
                steps=step_data.get("steps", []),
            )
        )

    output = wf_data.get("output", "")
    variables = wf_data.get("variables", {})

    return Workflow(
        name=name,
        agents=agents,
        steps=steps,
        output=output,
        variables=variables,
    )
