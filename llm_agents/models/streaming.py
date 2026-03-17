"""Extension 13: Streaming & Real-Time Output.

Provides async streaming support for LLM model backends, yielding tokens
one-by-one with optional log-probability information.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk emitted during streaming generation.

    Attributes:
        token: The token text.
        logprob: Log-probability of the token (None if unavailable).
        finish_reason: If set, indicates why generation stopped.
    """

    token: str
    logprob: float | None = None
    finish_reason: str | None = None


@runtime_checkable
class StreamCallback(Protocol):
    """Protocol for objects that receive streaming events."""

    def on_token(self, token: str, logprob: float | None) -> None: ...
    def on_thinking(self, text: str) -> None: ...
    def on_tool_call(self, name: str, args: dict[str, Any]) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...
    def on_complete(self, response: str) -> None: ...


class PrintStreamCallback:
    """Simple callback that prints tokens to stdout."""

    def on_token(self, token: str, logprob: float | None) -> None:
        print(token, end="", flush=True)

    def on_thinking(self, text: str) -> None:
        pass

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        print(f"\n[Calling tool: {name}]", flush=True)

    def on_tool_result(self, name: str, result: str) -> None:
        print(f"\n[Tool {name} returned: {result[:100]}]", flush=True)

    def on_complete(self, response: str) -> None:
        print(flush=True)


class CollectStreamCallback:
    """Callback that collects all events for later inspection."""

    def __init__(self) -> None:
        self.tokens: list[tuple[str, float | None]] = []
        self.thinking_blocks: list[str] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_results: list[tuple[str, str]] = []
        self.final_response: str | None = None

    def on_token(self, token: str, logprob: float | None) -> None:
        self.tokens.append((token, logprob))

    def on_thinking(self, text: str) -> None:
        self.thinking_blocks.append(text)

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self.tool_calls.append((name, args))

    def on_tool_result(self, name: str, result: str) -> None:
        self.tool_results.append((name, result))

    def on_complete(self, response: str) -> None:
        self.final_response = response

    @property
    def text(self) -> str:
        """Return concatenated token text."""
        return "".join(t for t, _ in self.tokens)


@dataclass
class StreamingResult:
    """Result that can be iterated asynchronously for streaming chunks.

    Can be used as::

        async for chunk in streaming_result:
            print(chunk.token, end="")
    """

    _chunks: list[StreamChunk] = field(default_factory=list)
    _complete: bool = False

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to the result."""
        self._chunks.append(chunk)
        if chunk.finish_reason is not None:
            self._complete = True

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def text(self) -> str:
        """Concatenate all chunk tokens into the full text."""
        return "".join(c.token for c in self._chunks)

    @property
    def chunks(self) -> list[StreamChunk]:
        return list(self._chunks)

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        return _StreamIterator(self._chunks)


class _StreamIterator:
    """Async iterator over a list of StreamChunks."""

    def __init__(self, chunks: list[StreamChunk]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> _StreamIterator:
        return self

    async def __anext__(self) -> StreamChunk:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def simulate_stream(text: str, chunk_size: int = 1) -> StreamingResult:
    """Create a StreamingResult from a complete text string.

    Useful for testing or for models that don't natively support streaming.

    Args:
        text: The full text to simulate streaming for.
        chunk_size: Number of characters per chunk.

    Returns:
        A completed StreamingResult.
    """
    result = StreamingResult()
    for i in range(0, len(text), chunk_size):
        token = text[i : i + chunk_size]
        is_last = (i + chunk_size) >= len(text)
        chunk = StreamChunk(
            token=token,
            finish_reason="stop" if is_last else None,
        )
        result.add_chunk(chunk)
    if not text:
        result.add_chunk(StreamChunk(token="", finish_reason="stop"))
    return result


def stream_with_callback(
    result: StreamingResult,
    callback: StreamCallback | None,
) -> str:
    """Process a StreamingResult synchronously, invoking callback for each chunk.

    Args:
        result: The streaming result to process.
        callback: Optional callback to invoke per token.

    Returns:
        The full concatenated text.
    """
    tokens: list[str] = []
    for chunk in result.chunks:
        tokens.append(chunk.token)
        if callback is not None:
            callback.on_token(chunk.token, chunk.logprob)
    full_text = "".join(tokens)
    if callback is not None:
        callback.on_complete(full_text)
    return full_text
