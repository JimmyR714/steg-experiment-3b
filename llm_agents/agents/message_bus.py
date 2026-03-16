"""In-memory message bus for inter-agent communication."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A message sent between agents.

    Attributes:
        sender: Name of the sending agent.
        recipient: Name of the target agent, or ``"*"`` for broadcast.
        content: The message body.
        metadata: Arbitrary key-value metadata attached to the message.
    """

    sender: str
    recipient: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """In-memory pub/sub message bus for multi-agent communication.

    Agents subscribe by name.  Messages addressed to a specific agent are
    queued for that agent.  Broadcast messages (recipient ``"*"``) are
    delivered to **all** subscribed agents except the sender.
    """

    def __init__(self) -> None:
        self._subscribers: set[str] = set()
        self._queues: dict[str, list[Message]] = defaultdict(list)
        self._all_messages: list[Message] = []

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, agent_name: str) -> None:
        """Register an agent to receive messages."""
        self._subscribers.add(agent_name)

    def unsubscribe(self, agent_name: str) -> None:
        """Remove an agent from the bus."""
        self._subscribers.discard(agent_name)
        self._queues.pop(agent_name, None)

    @property
    def subscribers(self) -> list[str]:
        """Return sorted list of subscribed agent names."""
        return sorted(self._subscribers)

    # ------------------------------------------------------------------
    # Send / Receive
    # ------------------------------------------------------------------

    def send(self, message: Message) -> None:
        """Deliver a message.

        * If ``message.recipient`` is ``"*"`` the message is broadcast to
          every subscribed agent except the sender.
        * Otherwise it is queued only for the named recipient.

        Raises:
            ValueError: If the recipient is not ``"*"`` and is not a
                subscribed agent.
        """
        self._all_messages.append(message)

        if message.recipient == "*":
            for name in self._subscribers:
                if name != message.sender:
                    self._queues[name].append(message)
        else:
            if message.recipient not in self._subscribers:
                raise ValueError(
                    f"Unknown recipient: {message.recipient!r}"
                )
            self._queues[message.recipient].append(message)

    def receive(self, agent_name: str) -> list[Message]:
        """Return and clear all pending messages for *agent_name*."""
        msgs = list(self._queues.get(agent_name, []))
        self._queues[agent_name] = []
        return msgs

    def peek(self, agent_name: str) -> list[Message]:
        """Return pending messages without clearing them."""
        return list(self._queues.get(agent_name, []))

    @property
    def all_messages(self) -> list[Message]:
        """Return a copy of every message ever sent through the bus."""
        return list(self._all_messages)
