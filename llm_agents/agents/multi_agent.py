"""Multi-agent system with inter-agent communication via message bus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from llm_agents.agents.agent import Agent
from llm_agents.agents.message_bus import Message, MessageBus
from llm_agents.agents.task import TaskResult
from llm_agents.tools.base import Tool

if TYPE_CHECKING:
    from llm_agents.tracing.tracer import Tracer


def _make_send_message_tool(bus: MessageBus, sender_name: str) -> Tool:
    """Create a ``send_message`` tool bound to a specific agent and bus."""

    def send_message(to: str, content: str) -> str:
        """Send a message to another agent."""
        msg = Message(sender=sender_name, recipient=to, content=content)
        try:
            bus.send(msg)
        except ValueError as exc:
            return f"Error: {exc}"
        return f"Message sent to {to}."

    return Tool(
        name="send_message",
        description="Send a message to another agent by name.",
        parameters_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["to", "content"],
        },
        fn=send_message,
    )


def _make_broadcast_tool(bus: MessageBus, sender_name: str) -> Tool:
    """Create a ``broadcast`` tool bound to a specific agent and bus."""

    def broadcast(content: str) -> str:
        """Broadcast a message to all agents."""
        msg = Message(sender=sender_name, recipient="*", content=content)
        bus.send(msg)
        return "Broadcast sent."

    return Tool(
        name="broadcast",
        description="Broadcast a message to all other agents.",
        parameters_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
            },
            "required": ["content"],
        },
        fn=broadcast,
    )


def _make_wait_for_reply_tool(bus: MessageBus, receiver_name: str) -> Tool:
    """Create a ``wait_for_reply`` tool that retrieves messages from a sender."""

    def wait_for_reply(from_agent: str, timeout: int = 30) -> str:
        """Wait for a reply from a specific agent."""
        messages = bus.receive(receiver_name)
        from_msgs = [m for m in messages if m.sender == from_agent]
        if from_msgs:
            # Return the most recent message from the specified sender.
            return from_msgs[-1].content
        # Put back messages not from the requested sender.
        other_msgs = [m for m in messages if m.sender != from_agent]
        for m in other_msgs:
            bus._queues[receiver_name].append(m)
        return f"No reply from {from_agent} yet."

    return Tool(
        name="wait_for_reply",
        description="Wait for a reply from a specific agent.",
        parameters_schema={
            "type": "object",
            "properties": {
                "from_agent": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["from_agent"],
        },
        fn=wait_for_reply,
    )


def _make_list_agents_tool(bus: MessageBus) -> Tool:
    """Create a ``list_agents`` tool that returns subscribed agent names."""

    def list_agents() -> str:
        """List all available agents in the system."""
        return json.dumps(bus.subscribers)

    return Tool(
        name="list_agents",
        description="List all available agents in the system.",
        parameters_schema={
            "type": "object",
            "properties": {},
        },
        fn=list_agents,
    )


class MultiAgentSystem:
    """Orchestrates multiple agents communicating via a shared message bus.

    Communication tools (``send_message``, ``broadcast``, ``wait_for_reply``,
    ``list_agents``) are automatically injected into each agent's tool
    registry on construction.

    Args:
        agents: The agents participating in the system.
        message_bus: The message bus used for inter-agent communication.
            If *None*, a new :class:`MessageBus` is created.
    """

    def __init__(
        self,
        agents: list[Agent],
        message_bus: MessageBus | None = None,
    ) -> None:
        self.bus = message_bus or MessageBus()
        self._agents: dict[str, Agent] = {}

        for agent in agents:
            self._agents[agent.name] = agent
            self.bus.subscribe(agent.name)
            self._inject_communication_tools(agent)

    def _inject_communication_tools(self, agent: Agent) -> None:
        """Add inter-agent communication tools to an agent's registry."""
        tools = [
            _make_send_message_tool(self.bus, agent.name),
            _make_broadcast_tool(self.bus, agent.name),
            _make_wait_for_reply_tool(self.bus, agent.name),
            _make_list_agents_tool(self.bus),
        ]
        for t in tools:
            if t.name not in agent._registry:
                agent._registry.register(t)

    @property
    def agents(self) -> dict[str, Agent]:
        """Return a mapping of agent names to agents."""
        return dict(self._agents)

    def run_task(
        self, task: str, coordinator: str, tracer: Tracer | None = None
    ) -> TaskResult:
        """Run a collaborative task.

        The *coordinator* agent receives the task as a user message.  During
        its execution it may use communication tools to delegate sub-tasks to
        other agents.  When the coordinator produces a final answer (a
        response with no further tool calls), each pending message for other
        agents is delivered by running that agent, and the coordinator's
        answer is returned as the task result.

        Args:
            task: A natural-language description of the task.
            coordinator: Name of the agent that will orchestrate the task.
            tracer: Optional :class:`Tracer` for recording execution events.

        Returns:
            A :class:`TaskResult` with the final answer and the full trace
            of inter-agent messages.

        Raises:
            KeyError: If *coordinator* is not one of the registered agents.
        """
        if coordinator not in self._agents:
            raise KeyError(f"Unknown coordinator agent: {coordinator!r}")

        if tracer:
            tracer.event("task_start", coordinator, {"task": task})

        coord_agent = self._agents[coordinator]
        coord_response = coord_agent.run(task, tracer=tracer)

        # Process pending messages for non-coordinator agents.  Each agent
        # that has pending messages gets to run with those messages as input.
        for name, agent in self._agents.items():
            if name == coordinator:
                continue
            pending = self.bus.receive(name)
            for msg in pending:
                if tracer:
                    tracer.event("message_delivered", name, {
                        "from": msg.sender,
                        "content_length": len(msg.content),
                    })
                agent.run(msg.content, tracer=tracer)

        if tracer:
            tracer.event("task_end", coordinator, {"result_length": len(coord_response.content)})

        return TaskResult(
            result=coord_response.content,
            agent_trace=self.bus.all_messages,
            logprobs=coord_response.logprobs,
        )
