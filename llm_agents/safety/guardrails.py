"""Extension 15: Composable guardrail chains and guarded agent wrapper.

Combines input and output filters into a Guardrail chain, and provides
a GuardedAgent that transparently applies safety checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_agents.safety.input_filter import InputFilter, InjectionResult, ThreatLevel
from llm_agents.safety.output_filter import FilterAction, OutputFilter, ViolationResult

if TYPE_CHECKING:
    from llm_agents.agents.agent import Agent, AgentResponse


@dataclass
class AuditEntry:
    """A log entry for a filter trigger.

    Attributes:
        timestamp: Unix timestamp of the event.
        direction: "input" or "output".
        trigger: Name of the filter/pattern that triggered.
        action_taken: What action was taken.
        text_snippet: First 200 chars of the text that triggered.
    """

    timestamp: float
    direction: str
    trigger: str
    action_taken: str
    text_snippet: str


class Guardrail:
    """Composable filter chain combining input and output safety checks.

    Args:
        input_filters: List of InputFilter instances to apply to inputs.
        output_filters: List of OutputFilter instances to apply to outputs.
        block_on_input_threat: Minimum threat level to block input. None
            means never block (only log).
        block_on_output_violation: If True, block output that triggers a
            BLOCK-level policy.
    """

    def __init__(
        self,
        input_filters: list[InputFilter] | None = None,
        output_filters: list[OutputFilter] | None = None,
        block_on_input_threat: ThreatLevel | None = ThreatLevel.HIGH,
        block_on_output_violation: bool = True,
    ) -> None:
        self.input_filters = input_filters or [InputFilter()]
        self.output_filters = output_filters or [OutputFilter()]
        self.block_on_input_threat = block_on_input_threat
        self.block_on_output_violation = block_on_output_violation
        self._audit_log: list[AuditEntry] = []

    def check_input(self, text: str) -> InjectionResult | None:
        """Run all input filters on the text.

        Returns the most severe InjectionResult, or None if clean.
        """
        worst: InjectionResult | None = None
        _levels = list(ThreatLevel)

        for filt in self.input_filters:
            result = filt.scan(text)
            if result.is_suspicious:
                self._log_audit("input", result.matched_patterns, "detected", text)
                if worst is None or _levels.index(result.threat_level) > _levels.index(worst.threat_level):
                    worst = result

        return worst

    def check_output(self, text: str) -> ViolationResult | None:
        """Run all output filters on the text.

        Returns the most severe ViolationResult, or None if clean.
        """
        worst: ViolationResult | None = None
        _actions = list(FilterAction)

        for filt in self.output_filters:
            result = filt.scan(text)
            if result.is_violation:
                self._log_audit("output", result.categories, result.action.value, text)
                if worst is None or _actions.index(result.action) > _actions.index(worst.action):
                    worst = result

        return worst

    def should_block_input(self, result: InjectionResult) -> bool:
        """Determine if an input should be blocked based on threat level."""
        if self.block_on_input_threat is None:
            return False
        _levels = list(ThreatLevel)
        return _levels.index(result.threat_level) >= _levels.index(self.block_on_input_threat)

    def should_block_output(self, result: ViolationResult) -> bool:
        """Determine if an output should be blocked."""
        if not self.block_on_output_violation:
            return False
        return result.action == FilterAction.BLOCK

    def _log_audit(
        self,
        direction: str,
        triggers: tuple[str, ...] | Any,
        action: str,
        text: str,
    ) -> None:
        self._audit_log.append(
            AuditEntry(
                timestamp=time.time(),
                direction=direction,
                trigger=", ".join(triggers) if isinstance(triggers, (list, tuple)) else str(triggers),
                action_taken=action,
                text_snippet=text[:200],
            )
        )

    @property
    def audit_log(self) -> list[AuditEntry]:
        """Return the full audit log."""
        return list(self._audit_log)


class GuardedAgent:
    """Wraps an Agent with input/output safety guardrails.

    If the input is blocked, ``run()`` returns a safe refusal message
    instead of calling the underlying agent. If the output is blocked,
    it returns a safe replacement.

    Args:
        agent: The underlying agent to wrap.
        guardrail: The guardrail chain to apply.
        blocked_input_message: Message returned when input is blocked.
        blocked_output_message: Message returned when output is blocked.
    """

    def __init__(
        self,
        agent: Agent,
        guardrail: Guardrail,
        blocked_input_message: str = "I cannot process this request due to safety concerns.",
        blocked_output_message: str = "The response was filtered due to safety concerns.",
    ) -> None:
        self.agent = agent
        self.guardrail = guardrail
        self.blocked_input_message = blocked_input_message
        self.blocked_output_message = blocked_output_message

    @property
    def name(self) -> str:
        return self.agent.name

    def run(self, user_message: str, tracer: Any = None) -> AgentResponse:
        """Run the agent with safety checks on input and output.

        Args:
            user_message: The user's input.
            tracer: Optional tracer for execution recording.

        Returns:
            An AgentResponse, potentially replaced if safety checks fail.
        """
        from llm_agents.agents.agent import AgentResponse

        # Check input
        input_result = self.guardrail.check_input(user_message)
        if input_result is not None and self.guardrail.should_block_input(input_result):
            return AgentResponse(content=self.blocked_input_message)

        # Run the agent
        response = self.agent.run(user_message, tracer=tracer)

        # Check output
        output_result = self.guardrail.check_output(response.content)
        if output_result is not None and self.guardrail.should_block_output(output_result):
            return AgentResponse(
                content=self.blocked_output_message,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
            )

        return response

    def reset(self) -> None:
        """Clear conversation history."""
        self.agent.reset()
