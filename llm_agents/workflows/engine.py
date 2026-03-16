"""Workflow execution engine for declarative multi-agent pipelines."""

from __future__ import annotations

from typing import Any

from llm_agents.agents.agent import Agent
from llm_agents.models.base import BaseModel
from llm_agents.models.registry import get_model
from llm_agents.tools.base import Tool
from llm_agents.tools.builtin import ALL_BUILTIN_TOOLS
from llm_agents.tools.registry import ToolRegistry
from llm_agents.workflows.schema import Step, Workflow, interpolate


class WorkflowEngine:
    """Executes a :class:`Workflow` by instantiating agents and running steps.

    The engine resolves agent definitions to real :class:`Agent` instances,
    manages a variable store for inter-step data flow, and supports
    sequential, parallel, conditional, and loop control flow.

    Args:
        models: Mapping of model name to :class:`BaseModel` instance.
            Agent definitions reference these by name.
        available_tools: Optional mapping of tool name to :class:`Tool`
            instance for resolving tool references in agent definitions.
    """

    def __init__(
        self,
        models: dict[str, BaseModel] | None = None,
        available_tools: dict[str, Tool] | None = None,
    ) -> None:
        self._models = models or {}
        self._available_tools: dict[str, Tool] = {}
        if available_tools:
            self._available_tools.update(available_tools)
        # Index built-in tools by name
        for t in ALL_BUILTIN_TOOLS:
            self._available_tools.setdefault(t.name, t)

    def _resolve_model(self, model_name: str) -> BaseModel:
        """Look up a model by name, checking local mapping then registry."""
        if model_name in self._models:
            return self._models[model_name]
        return get_model(model_name)

    def _make_agent(self, name: str, workflow: Workflow) -> Agent:
        """Instantiate an Agent from a workflow's agent definition."""
        agent_def = workflow.agents.get(name)
        if agent_def is None:
            raise ValueError(f"Agent {name!r} not defined in workflow")

        model = self._resolve_model(agent_def.model)
        tools: list[Tool] = []
        for tool_name in agent_def.tools:
            if tool_name in self._available_tools:
                tools.append(self._available_tools[tool_name])

        return Agent(
            name=name,
            model=model,
            system_prompt=agent_def.system_prompt,
            tools=tools,
            enable_cot=False,
        )

    def _execute_step(
        self,
        step: Step,
        variables: dict[str, Any],
        workflow: Workflow,
    ) -> dict[str, Any]:
        """Execute a single step and return updated variables."""
        if step.step_type == "parallel":
            return self._execute_parallel(step, variables, workflow)
        if step.step_type == "conditional":
            return self._execute_conditional(step, variables, workflow)
        if step.step_type == "loop":
            return self._execute_loop(step, variables, workflow)
        return self._execute_sequential(step, variables, workflow)

    def _execute_sequential(
        self,
        step: Step,
        variables: dict[str, Any],
        workflow: Workflow,
    ) -> dict[str, Any]:
        """Run a single agent step."""
        agent = self._make_agent(step.agent, workflow)
        input_text = interpolate(step.input, variables)
        response = agent.run(input_text)
        if step.output:
            variables[step.output] = response.content
        return variables

    def _execute_parallel(
        self,
        step: Step,
        variables: dict[str, Any],
        workflow: Workflow,
    ) -> dict[str, Any]:
        """Run nested sub-steps (sequentially here; true parallelism
        would require async, which we keep out of scope for now)."""
        for sub_step_data in step.steps:
            sub_step = Step(
                agent=sub_step_data.get("agent", ""),
                input=sub_step_data.get("input", ""),
                output=sub_step_data.get("output", ""),
                step_type=sub_step_data.get("type", "sequential"),
            )
            variables = self._execute_step(sub_step, variables, workflow)
        return variables

    def _execute_conditional(
        self,
        step: Step,
        variables: dict[str, Any],
        workflow: Workflow,
    ) -> dict[str, Any]:
        """Evaluate a condition and execute the matching branch."""
        condition_value = interpolate(step.condition, variables).strip().lower()

        # Try to match a branch
        branch_steps = step.branches.get(condition_value)
        if branch_steps is None:
            # Try "else" / "default" branch
            branch_steps = step.branches.get("else", step.branches.get("default"))
        if branch_steps is None:
            return variables

        for sub_step_data in branch_steps:
            sub_step = Step(
                agent=sub_step_data.get("agent", ""),
                input=sub_step_data.get("input", ""),
                output=sub_step_data.get("output", ""),
                step_type=sub_step_data.get("type", "sequential"),
            )
            variables = self._execute_step(sub_step, variables, workflow)
        return variables

    def _execute_loop(
        self,
        step: Step,
        variables: dict[str, Any],
        workflow: Workflow,
    ) -> dict[str, Any]:
        """Repeat nested steps up to max_iterations times.

        If step.items is set and refers to a list variable, iterates over
        the items.  Otherwise loops for max_iterations.
        """
        items_var = variables.get(step.items) if step.items else None
        if isinstance(items_var, list):
            iterations = items_var
        else:
            iterations = range(min(step.max_iterations, 100))

        results: list[str] = []
        for i, item in enumerate(iterations):
            if step.items and isinstance(items_var, list):
                variables["_item"] = item
                variables["_index"] = i

            for sub_step_data in step.steps:
                sub_step = Step(
                    agent=sub_step_data.get("agent", ""),
                    input=sub_step_data.get("input", ""),
                    output=sub_step_data.get("output", ""),
                    step_type=sub_step_data.get("type", "sequential"),
                )
                variables = self._execute_step(sub_step, variables, workflow)

            if step.output and step.output in variables:
                results.append(str(variables[step.output]))

        if step.output and results:
            variables[step.output] = "\n".join(results)

        return variables

    def run(
        self,
        workflow: Workflow,
        initial_variables: dict[str, Any] | None = None,
    ) -> str:
        """Execute a workflow and return the final output.

        Args:
            workflow: The workflow to execute.
            initial_variables: Variables to make available to the workflow.

        Returns:
            The interpolated output string.
        """
        variables: dict[str, Any] = dict(workflow.variables)
        if initial_variables:
            variables.update(initial_variables)

        for step in workflow.steps:
            variables = self._execute_step(step, variables, workflow)

        return interpolate(workflow.output, variables)
