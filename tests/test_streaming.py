"""Tests for Extension 13: Streaming & Real-Time Output."""

from __future__ import annotations

import asyncio

import pytest

from llm_agents.models.streaming import (
    CollectStreamCallback,
    StreamChunk,
    StreamingResult,
    simulate_stream,
    stream_with_callback,
)


# ---------------------------------------------------------------------------
# StreamChunk tests
# ---------------------------------------------------------------------------


class TestStreamChunk:
    def test_basic_chunk(self):
        chunk = StreamChunk(token="hello", logprob=-0.5)
        assert chunk.token == "hello"
        assert chunk.logprob == -0.5
        assert chunk.finish_reason is None

    def test_final_chunk(self):
        chunk = StreamChunk(token=".", finish_reason="stop")
        assert chunk.finish_reason == "stop"


# ---------------------------------------------------------------------------
# StreamingResult tests
# ---------------------------------------------------------------------------


class TestStreamingResult:
    def test_empty_result(self):
        result = StreamingResult()
        assert result.text == ""
        assert not result.is_complete
        assert result.chunks == []

    def test_add_chunks(self):
        result = StreamingResult()
        result.add_chunk(StreamChunk(token="Hello"))
        result.add_chunk(StreamChunk(token=" world", finish_reason="stop"))
        assert result.text == "Hello world"
        assert result.is_complete
        assert len(result.chunks) == 2

    def test_async_iteration(self):
        result = StreamingResult()
        result.add_chunk(StreamChunk(token="A"))
        result.add_chunk(StreamChunk(token="B", finish_reason="stop"))

        async def collect():
            tokens = []
            async for chunk in result:
                tokens.append(chunk.token)
            return tokens

        tokens = asyncio.run(collect())
        assert tokens == ["A", "B"]


# ---------------------------------------------------------------------------
# simulate_stream tests
# ---------------------------------------------------------------------------


class TestSimulateStream:
    def test_basic_simulation(self):
        result = simulate_stream("Hello")
        assert result.text == "Hello"
        assert result.is_complete
        assert len(result.chunks) == 5  # one per char

    def test_chunk_size(self):
        result = simulate_stream("Hello World", chunk_size=5)
        assert result.text == "Hello World"
        assert len(result.chunks) == 3  # "Hello", " Worl", "d"

    def test_empty_text(self):
        result = simulate_stream("")
        assert result.text == ""
        assert result.is_complete

    def test_last_chunk_has_finish_reason(self):
        result = simulate_stream("ab", chunk_size=1)
        assert result.chunks[-1].finish_reason == "stop"
        assert result.chunks[0].finish_reason is None


# ---------------------------------------------------------------------------
# CollectStreamCallback tests
# ---------------------------------------------------------------------------


class TestCollectStreamCallback:
    def test_collect_tokens(self):
        cb = CollectStreamCallback()
        cb.on_token("Hello", -0.1)
        cb.on_token(" world", -0.2)
        cb.on_complete("Hello world")

        assert cb.text == "Hello world"
        assert len(cb.tokens) == 2
        assert cb.final_response == "Hello world"

    def test_collect_tool_events(self):
        cb = CollectStreamCallback()
        cb.on_tool_call("search", {"query": "test"})
        cb.on_tool_result("search", "found it")

        assert len(cb.tool_calls) == 1
        assert cb.tool_calls[0] == ("search", {"query": "test"})
        assert cb.tool_results[0] == ("search", "found it")

    def test_thinking_blocks(self):
        cb = CollectStreamCallback()
        cb.on_thinking("Let me think...")
        assert cb.thinking_blocks == ["Let me think..."]


# ---------------------------------------------------------------------------
# stream_with_callback tests
# ---------------------------------------------------------------------------


class TestStreamWithCallback:
    def test_processes_all_chunks(self):
        result = simulate_stream("test")
        cb = CollectStreamCallback()
        text = stream_with_callback(result, cb)
        assert text == "test"
        assert cb.final_response == "test"

    def test_no_callback(self):
        result = simulate_stream("test")
        text = stream_with_callback(result, None)
        assert text == "test"
