"""Core tracing types and context manager."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """A single trace event.

    Attributes:
        timestamp: Unix timestamp when the event occurred.
        agent_name: Name of the agent that produced the event.
        event_type: Type of event (e.g. ``"model_call"``, ``"tool_call"``,
            ``"message_sent"``, ``"message_received"``, ``"thinking"``).
        data: Arbitrary data associated with the event.
        span_id: ID of the span this event belongs to.
    """

    timestamp: float
    agent_name: str
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    span_id: str = ""


@dataclass
class Span:
    """Groups related trace events (e.g. one agent turn).

    Attributes:
        span_id: Unique identifier for this span.
        name: Human-readable span name.
        agent_name: Name of the agent this span belongs to.
        start_time: Unix timestamp when the span started.
        end_time: Unix timestamp when the span ended (0 if still open).
        events: Events within this span.
        parent_id: ID of the parent span, if any.
    """

    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = ""
    agent_name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    events: list[TraceEvent] = field(default_factory=list)
    parent_id: str = ""

    @property
    def duration_ms(self) -> float:
        """Duration of this span in milliseconds."""
        if self.end_time <= 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Records trace events and spans for agent execution.

    Can be used as a context manager to automatically track timing.

    Example::

        tracer = Tracer()
        with tracer.span("agent_turn", agent_name="assistant"):
            tracer.event("model_call", "assistant", {"prompt": "..."})
            # ... agent work ...
        print(tracer.events)
    """

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._spans: list[Span] = []
        self._active_span: Span | None = None

    @property
    def events(self) -> list[TraceEvent]:
        """All recorded events."""
        return list(self._events)

    @property
    def spans(self) -> list[Span]:
        """All recorded spans."""
        return list(self._spans)

    def event(
        self,
        event_type: str,
        agent_name: str,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Record a trace event.

        Args:
            event_type: Type of the event.
            agent_name: Name of the agent producing the event.
            data: Arbitrary event data.

        Returns:
            The created :class:`TraceEvent`.
        """
        evt = TraceEvent(
            timestamp=time.time(),
            agent_name=agent_name,
            event_type=event_type,
            data=data or {},
            span_id=self._active_span.span_id if self._active_span else "",
        )
        self._events.append(evt)
        if self._active_span:
            self._active_span.events.append(evt)
        return evt

    def span(self, name: str, agent_name: str = "") -> _SpanContext:
        """Create a span context manager.

        Args:
            name: Human-readable span name.
            agent_name: Agent this span belongs to.

        Returns:
            A context manager that starts/ends the span.
        """
        return _SpanContext(self, name, agent_name)

    def start_span(self, name: str, agent_name: str = "") -> Span:
        """Manually start a span.

        Args:
            name: Human-readable span name.
            agent_name: Agent this span belongs to.

        Returns:
            The created :class:`Span`.
        """
        s = Span(
            name=name,
            agent_name=agent_name,
            start_time=time.time(),
            parent_id=self._active_span.span_id if self._active_span else "",
        )
        self._spans.append(s)
        self._active_span = s
        return s

    def end_span(self, span: Span | None = None) -> None:
        """End a span (defaults to the active span)."""
        target = span or self._active_span
        if target:
            target.end_time = time.time()
        if target is self._active_span:
            self._active_span = None


class _SpanContext:
    """Context manager for spans."""

    def __init__(self, tracer: Tracer, name: str, agent_name: str) -> None:
        self._tracer = tracer
        self._name = name
        self._agent_name = agent_name
        self._span: Span | None = None

    def __enter__(self) -> Span:
        self._span = self._tracer.start_span(self._name, self._agent_name)
        return self._span

    def __exit__(self, *exc: object) -> None:
        self._tracer.end_span(self._span)
