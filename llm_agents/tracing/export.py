"""Export trace data to various formats."""

from __future__ import annotations

import json
from typing import Any

from llm_agents.tracing.tracer import Tracer


def to_json(tracer: Tracer) -> str:
    """Serialize a tracer's events and spans to JSON.

    Args:
        tracer: The tracer to export.

    Returns:
        A JSON string with ``events`` and ``spans`` keys.
    """
    data = {
        "events": [
            {
                "timestamp": e.timestamp,
                "agent_name": e.agent_name,
                "event_type": e.event_type,
                "data": e.data,
                "span_id": e.span_id,
            }
            for e in tracer.events
        ],
        "spans": [
            {
                "span_id": s.span_id,
                "name": s.name,
                "agent_name": s.agent_name,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration_ms": s.duration_ms,
                "parent_id": s.parent_id,
                "event_count": len(s.events),
            }
            for s in tracer.spans
        ],
    }
    return json.dumps(data, indent=2)


def to_chrome_trace(tracer: Tracer) -> str:
    """Export to Chrome Tracing format for visualization in ``chrome://tracing``.

    Args:
        tracer: The tracer to export.

    Returns:
        A JSON string in Chrome Trace Event format.
    """
    trace_events: list[dict[str, Any]] = []

    for span in tracer.spans:
        # Duration event (B/E pair)
        trace_events.append({
            "name": span.name,
            "cat": "agent",
            "ph": "B",
            "ts": span.start_time * 1_000_000,  # microseconds
            "pid": 1,
            "tid": span.agent_name or "main",
        })
        if span.end_time > 0:
            trace_events.append({
                "name": span.name,
                "cat": "agent",
                "ph": "E",
                "ts": span.end_time * 1_000_000,
                "pid": 1,
                "tid": span.agent_name or "main",
            })

    for event in tracer.events:
        trace_events.append({
            "name": event.event_type,
            "cat": "event",
            "ph": "i",
            "ts": event.timestamp * 1_000_000,
            "pid": 1,
            "tid": event.agent_name or "main",
            "s": "t",
            "args": event.data,
        })

    return json.dumps({"traceEvents": trace_events}, indent=2)


def to_opentelemetry(tracer: Tracer) -> dict[str, Any]:
    """Convert trace data to OpenTelemetry-compatible span format.

    Returns a dict structure compatible with OTLP JSON export,
    suitable for Jaeger, Zipkin, or Grafana integration.

    Args:
        tracer: The tracer to export.

    Returns:
        A dict with ``resourceSpans`` following OTLP conventions.
    """
    otlp_spans: list[dict[str, Any]] = []

    for span in tracer.spans:
        otlp_span: dict[str, Any] = {
            "traceId": "0" * 32,  # placeholder
            "spanId": span.span_id,
            "parentSpanId": span.parent_id or "",
            "name": span.name,
            "kind": 1,  # SPAN_KIND_INTERNAL
            "startTimeUnixNano": int(span.start_time * 1e9),
            "endTimeUnixNano": int(span.end_time * 1e9) if span.end_time > 0 else 0,
            "attributes": [
                {"key": "agent.name", "value": {"stringValue": span.agent_name}},
            ],
            "events": [
                {
                    "timeUnixNano": int(e.timestamp * 1e9),
                    "name": e.event_type,
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in e.data.items()
                    ],
                }
                for e in span.events
            ],
        }
        otlp_spans.append(otlp_span)

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "llm-agents"},
                        }
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "llm_agents.tracing"},
                        "spans": otlp_spans,
                    }
                ],
            }
        ]
    }
