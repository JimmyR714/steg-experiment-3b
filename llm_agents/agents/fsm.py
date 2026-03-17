"""Extension 16: Finite State Machine agent.

Provides a state-machine-based agent where behavior is controlled by
explicit states and transitions, enabling structured multi-step
conversational flows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from llm_agents.agents.agent import AgentResponse
from llm_agents.models.base import BaseModel
from llm_agents.tools.base import Tool

if TYPE_CHECKING:
    from llm_agents.tracing.tracer import Tracer


@dataclass
class Transition:
    """A transition between states.

    Attributes:
        target_state: Name of the state to transition to.
        condition: A regex pattern, tool call name, or callable that
            determines if this transition should fire. Callables receive
            the agent output string and return bool.
        priority: Higher priority transitions are checked first.
    """

    target_state: str
    condition: str | re.Pattern[str] | Callable[[str], bool]
    priority: int = 0

    def matches(self, output: str, tool_name: str | None = None) -> bool:
        """Check if this transition's condition is met.

        Args:
            output: The agent's output text.
            tool_name: Name of the last tool called, if any.

        Returns:
            True if the transition should fire.
        """
        cond = self.condition
        if callable(cond) and not isinstance(cond, re.Pattern):
            return cond(output)
        if isinstance(cond, re.Pattern):
            return bool(cond.search(output))
        if isinstance(cond, str):
            # Check if it matches a tool name
            if tool_name is not None and cond == tool_name:
                return True
            # Otherwise treat as regex
            return bool(re.search(cond, output, re.IGNORECASE))
        return False


@dataclass
class State:
    """A state in the agent state machine.

    Attributes:
        name: Unique state name.
        prompt: System prompt or instruction for this state.
        tools: Tools available in this state.
        transitions: Possible transitions from this state.
        is_terminal: If True, the FSM stops when entering this state.
    """

    name: str
    prompt: str
    tools: list[Tool] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    is_terminal: bool = False


class StateMachineAgent:
    """Agent whose behavior is governed by a finite state machine.

    The agent processes user input through the current state's prompt
    and tools, then checks transitions to determine the next state.
    Execution continues until a terminal state is reached or the
    maximum number of transitions is exhausted.

    Args:
        states: List of State definitions.
        initial_state: Name of the starting state.
        model: LLM model for generation.
        max_transitions: Maximum state transitions before forced stop.
    """

    def __init__(
        self,
        states: list[State],
        initial_state: str,
        model: BaseModel,
        max_transitions: int = 20,
    ) -> None:
        self._states: dict[str, State] = {s.name: s for s in states}
        self._initial_state = initial_state
        self._current_state = initial_state
        self._model = model
        self._max_transitions = max_transitions
        self._transition_history: list[tuple[str, str]] = []

        if initial_state not in self._states:
            raise ValueError(f"Initial state {initial_state!r} not found in states.")

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def state(self) -> State:
        return self._states[self._current_state]

    @property
    def transition_history(self) -> list[tuple[str, str]]:
        """Return list of (from_state, to_state) transitions."""
        return list(self._transition_history)

    def run(
        self,
        user_message: str,
        tracer: Tracer | None = None,
    ) -> AgentResponse:
        """Run the state machine for a user message.

        Processes the input through states until a terminal state is
        reached or max transitions are exhausted.

        Args:
            user_message: The user's input.
            tracer: Optional tracer for recording.

        Returns:
            AgentResponse from the final state.
        """
        from llm_agents.agents.agent import Agent

        responses: list[str] = []
        current_input = user_message

        for _ in range(self._max_transitions):
            state = self._states[self._current_state]

            # Create a temporary agent for this state
            agent = Agent(
                name=f"fsm_{state.name}",
                model=self._model,
                system_prompt=state.prompt,
                tools=state.tools,
                enable_cot=False,
            )

            response = agent.run(current_input, tracer=tracer)
            responses.append(response.content)

            # Check if this is a terminal state
            if state.is_terminal:
                break

            # Find matching transition
            last_tool = response.tool_calls[-1].name if response.tool_calls else None
            next_state = self._find_transition(state, response.content, last_tool)

            if next_state is None:
                # No transition matches — stay in current state and stop
                break

            self._transition_history.append((self._current_state, next_state))
            self._current_state = next_state

            # Feed output as input to the next state
            current_input = response.content

        return AgentResponse(
            content=responses[-1] if responses else "",
            thinking="",
            tool_calls=[],
        )

    def _find_transition(
        self,
        state: State,
        output: str,
        tool_name: str | None,
    ) -> str | None:
        """Find the first matching transition from the current state.

        Transitions are checked in priority order (highest first).
        """
        sorted_transitions = sorted(
            state.transitions, key=lambda t: t.priority, reverse=True
        )
        for transition in sorted_transitions:
            if transition.matches(output, tool_name):
                target = transition.target_state
                if target in self._states:
                    return target
        return None

    def reset(self) -> None:
        """Reset the state machine to the initial state."""
        self._current_state = self._initial_state
        self._transition_history.clear()

    def get_state(self, name: str) -> State | None:
        """Look up a state by name."""
        return self._states.get(name)


def fsm_to_mermaid(agent: StateMachineAgent) -> str:
    """Generate a Mermaid state diagram from a StateMachineAgent.

    Args:
        agent: The state machine agent to visualize.

    Returns:
        A Mermaid diagram string.
    """
    lines: list[str] = ["stateDiagram-v2"]

    # Mark initial state
    lines.append(f"    [*] --> {agent._initial_state}")

    for state in agent._states.values():
        # Add state description
        lines.append(f"    {state.name} : {state.prompt[:50]}")

        for transition in state.transitions:
            cond_str = (
                str(transition.condition.pattern)
                if isinstance(transition.condition, re.Pattern)
                else str(transition.condition)
            )
            # Truncate long conditions
            if len(cond_str) > 30:
                cond_str = cond_str[:27] + "..."
            lines.append(
                f"    {state.name} --> {transition.target_state} : {cond_str}"
            )

        if state.is_terminal:
            lines.append(f"    {state.name} --> [*]")

    return "\n".join(lines)
