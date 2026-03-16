"""Tests for Extension 4: Execution Tracing & Observability."""

from __future__ import annotations

import json
import time

import pytest

from llm_agents.tracing.cost import (
    BudgetExceededError,
    BudgetGuard,
    CostEstimator,
    TokenCounter,
)
from llm_agents.tracing.export import to_chrome_trace, to_json, to_opentelemetry
from llm_agents.tracing.tracer import Span, TraceEvent, Tracer


# ---------------------------------------------------------------------------
# TraceEvent tests
# ---------------------------------------------------------------------------


class TestTraceEvent:
    def test_fields(self):
        evt = TraceEvent(
            timestamp=1000.0,
            agent_name="agent1",
            event_type="model_call",
            data={"prompt": "hello"},
            span_id="abc123",
        )
        assert evt.timestamp == 1000.0
        assert evt.agent_name == "agent1"
        assert evt.event_type == "model_call"
        assert evt.data == {"prompt": "hello"}
        assert evt.span_id == "abc123"

    def test_default_data(self):
        evt = TraceEvent(timestamp=0, agent_name="a", event_type="test")
        assert evt.data == {}
        assert evt.span_id == ""


# ---------------------------------------------------------------------------
# Span tests
# ---------------------------------------------------------------------------


class TestSpan:
    def test_duration_ms(self):
        span = Span(start_time=1.0, end_time=1.5)
        assert span.duration_ms == pytest.approx(500.0)

    def test_duration_ms_unclosed(self):
        span = Span(start_time=1.0)
        assert span.duration_ms == 0.0

    def test_auto_id(self):
        span = Span()
        assert len(span.span_id) == 16


# ---------------------------------------------------------------------------
# Tracer tests
# ---------------------------------------------------------------------------


class TestTracer:
    def test_record_event(self):
        tracer = Tracer()
        evt = tracer.event("model_call", "agent1", {"tokens": 100})
        assert len(tracer.events) == 1
        assert tracer.events[0].event_type == "model_call"
        assert tracer.events[0].agent_name == "agent1"

    def test_span_context_manager(self):
        tracer = Tracer()
        with tracer.span("test_span", "agent1") as span:
            tracer.event("inner_event", "agent1")
            assert span.start_time > 0

        assert len(tracer.spans) == 1
        assert tracer.spans[0].end_time > 0
        assert tracer.spans[0].name == "test_span"
        assert len(tracer.spans[0].events) == 1

    def test_events_linked_to_span(self):
        tracer = Tracer()
        with tracer.span("s1", "a1") as span:
            tracer.event("e1", "a1")
            tracer.event("e2", "a1")

        assert all(e.span_id == span.span_id for e in tracer.events)

    def test_events_without_span(self):
        tracer = Tracer()
        tracer.event("standalone", "a1")
        assert tracer.events[0].span_id == ""

    def test_manual_span(self):
        tracer = Tracer()
        span = tracer.start_span("manual", "agent1")
        tracer.event("inside", "agent1")
        tracer.end_span(span)
        assert span.end_time > 0

    def test_multiple_spans(self):
        tracer = Tracer()
        with tracer.span("first", "a1"):
            tracer.event("e1", "a1")
        with tracer.span("second", "a2"):
            tracer.event("e2", "a2")
        assert len(tracer.spans) == 2
        assert len(tracer.events) == 2


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_to_json_valid(self):
        tracer = Tracer()
        with tracer.span("s1", "a1"):
            tracer.event("e1", "a1", {"key": "val"})

        result = to_json(tracer)
        data = json.loads(result)
        assert "events" in data
        assert "spans" in data
        assert len(data["events"]) == 1
        assert len(data["spans"]) == 1
        assert data["events"][0]["event_type"] == "e1"

    def test_empty_tracer(self):
        tracer = Tracer()
        result = to_json(tracer)
        data = json.loads(result)
        assert data["events"] == []
        assert data["spans"] == []


class TestExportChromeTrace:
    def test_chrome_trace_format(self):
        tracer = Tracer()
        with tracer.span("span1", "agent1"):
            tracer.event("evt1", "agent1")

        result = to_chrome_trace(tracer)
        data = json.loads(result)
        assert "traceEvents" in data
        # Should have B, E for span + i for event
        events = data["traceEvents"]
        phases = [e["ph"] for e in events]
        assert "B" in phases
        assert "E" in phases
        assert "i" in phases


class TestExportOpenTelemetry:
    def test_otlp_structure(self):
        tracer = Tracer()
        with tracer.span("s1", "a1"):
            tracer.event("e1", "a1", {"key": "value"})

        result = to_opentelemetry(tracer)
        assert "resourceSpans" in result
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 1
        assert spans[0]["name"] == "s1"
        assert len(spans[0]["events"]) == 1


# ---------------------------------------------------------------------------
# TokenCounter tests
# ---------------------------------------------------------------------------


class TestTokenCounter:
    def test_initial_state(self):
        counter = TokenCounter()
        assert counter.prompt_tokens == 0
        assert counter.completion_tokens == 0
        assert counter.total_tokens == 0
        assert counter.call_count == 0

    def test_record(self):
        counter = TokenCounter()
        counter.record(100, 50)
        assert counter.prompt_tokens == 100
        assert counter.completion_tokens == 50
        assert counter.total_tokens == 150
        assert counter.call_count == 1

    def test_multiple_records(self):
        counter = TokenCounter()
        counter.record(100, 50)
        counter.record(200, 100)
        assert counter.total_tokens == 450
        assert counter.call_count == 2

    def test_reset(self):
        counter = TokenCounter()
        counter.record(100, 50)
        counter.reset()
        assert counter.total_tokens == 0
        assert counter.call_count == 0


# ---------------------------------------------------------------------------
# CostEstimator tests
# ---------------------------------------------------------------------------


class TestCostEstimator:
    def test_estimate(self):
        estimator = CostEstimator(
            price_per_1k_prompt=0.01,
            price_per_1k_completion=0.03,
        )
        counter = TokenCounter()
        counter.record(1000, 1000)
        cost = estimator.estimate(counter)
        assert cost == pytest.approx(0.04)

    def test_zero_usage(self):
        estimator = CostEstimator()
        counter = TokenCounter()
        assert estimator.estimate(counter) == 0.0


# ---------------------------------------------------------------------------
# BudgetGuard tests
# ---------------------------------------------------------------------------


class TestBudgetGuard:
    def test_within_budget(self):
        guard = BudgetGuard(max_tokens=1000)
        guard.record_and_check(100, 50)
        assert guard.remaining == 850

    def test_exceeds_budget(self):
        guard = BudgetGuard(max_tokens=100)
        with pytest.raises(BudgetExceededError):
            guard.record_and_check(80, 50)

    def test_check_raises_after_exceed(self):
        guard = BudgetGuard(max_tokens=100)
        guard.counter.record(80, 30)
        with pytest.raises(BudgetExceededError):
            guard.check()

    def test_remaining_never_negative(self):
        guard = BudgetGuard(max_tokens=100)
        guard.counter.record(200, 200)
        assert guard.remaining == 0
