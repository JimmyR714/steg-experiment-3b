"""Task result types for multi-agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_agents.agents.message_bus import Message
from llm_agents.models.types import LogProbResult


@dataclass
class TaskResult:
    """The outcome of a multi-agent task.

    Attributes:
        result: The final answer or output produced by the system.
        agent_trace: Ordered list of all messages exchanged between agents
            during execution.
        logprobs: Optional log-probability result from the coordinator's
            final generation.
    """

    result: str
    agent_trace: list[Message] = field(default_factory=list)
    logprobs: LogProbResult | None = None
