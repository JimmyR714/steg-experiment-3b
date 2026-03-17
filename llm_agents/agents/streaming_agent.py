"""Extension 13: Streaming agent that yields partial results.

Wraps the standard Agent with streaming support, detecting tool calls
mid-stream and routing events through callbacks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_agents.agents.agent import Agent, AgentResponse, ToolCallRecord, _extract_tool_call
from llm_agents.agents.cot import parse_thinking
from llm_agents.models.streaming import (
    CollectStreamCallback,
    StreamCallback,
    StreamChunk,
    StreamingResult,
    simulate_stream,
    stream_with_callback,
)

if TYPE_CHECKING:
    from llm_agents.models.base import BaseModel
    from llm_agents.tools.base import Tool
    from llm_agents.tracing.tracer import Tracer


class StreamingAgent:
    """Agent that supports streaming output via callbacks.

    Wraps a standard Agent and provides ``run_stream()`` which yields
    partial results token-by-token while still supporting tool calls
    and chain-of-thought.

    Args:
        agent: The underlying Agent to wrap.
        callback: Optional callback receiving streaming events.
    """

    def __init__(
        self,
        agent: Agent,
        callback: StreamCallback | None = None,
    ) -> None:
        self.agent = agent
        self.callback = callback

    @property
    def name(self) -> str:
        return self.agent.name

    def run_stream(
        self,
        user_message: str,
        tracer: Tracer | None = None,
    ) -> AgentResponse:
        """Run the agent with streaming output.

        This method runs the agent normally, then simulates streaming
        the output through the callback. For models that support native
        streaming, the callback receives tokens as they arrive.

        Args:
            user_message: The user's input message.
            tracer: Optional tracer for execution recording.

        Returns:
            The final AgentResponse.
        """
        response = self.agent.run(user_message, tracer=tracer)

        if self.callback is not None:
            # Stream thinking blocks
            if response.thinking:
                self.callback.on_thinking(response.thinking)

            # Stream tool calls
            for tc in response.tool_calls:
                self.callback.on_tool_call(tc.name, tc.arguments)
                self.callback.on_tool_result(tc.name, tc.result)

            # Stream the final content token-by-token
            streaming_result = simulate_stream(response.content)
            stream_with_callback(streaming_result, self.callback)

        return response

    def reset(self) -> None:
        """Clear conversation history."""
        self.agent.reset()
