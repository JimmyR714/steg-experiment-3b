"""Single autonomous agent with tool use and chain-of-thought support."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from llm_agents.agents.cot import inject_cot_instruction, parse_thinking
from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult
from llm_agents.tools.base import Tool
from llm_agents.tools.executor import execute_tool_call
from llm_agents.tools.registry import ToolRegistry

# Pattern to detect a tool call in model output: a JSON block with "name" and
# "arguments" keys, optionally wrapped in ```tool_call ... ``` fences.
_TOOL_CALL_FENCED = re.compile(
    r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL
)
_TOOL_CALL_JSON = re.compile(
    r'(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})',
    re.DOTALL,
)

MAX_TOOL_ROUNDS = 10


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation during an agent run."""

    name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class AgentResponse:
    """The final result of an agent run.

    Attributes:
        content: The user-visible response text.
        thinking: Concatenated chain-of-thought reasoning (empty if CoT
            is disabled or the model did not use ``<think>`` tags).
        tool_calls: Ordered list of tool calls made during the run.
        logprobs: Log-probability result from the final generation, if
            available.
    """

    content: str
    thinking: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    logprobs: LogProbResult | None = None


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    """Try to extract a tool-call JSON object from model output.

    Returns a dict with ``name`` and ``arguments`` keys, or *None* if no
    tool call is found.
    """
    # Try fenced format first, then bare JSON.
    match = _TOOL_CALL_FENCED.search(text)
    if match is None:
        match = _TOOL_CALL_JSON.search(text)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if "name" in parsed and "arguments" in parsed:
        return {"name": parsed["name"], "arguments": parsed["arguments"]}
    return None


def _build_prompt(
    system_prompt: str,
    history: list[dict[str, str]],
    tool_prompt: str,
) -> str:
    """Serialize a conversation into a single text prompt."""
    parts: list[str] = []
    full_system = system_prompt
    if tool_prompt:
        full_system = full_system + "\n\n" + tool_prompt
    parts.append(f"System: {full_system}")
    for msg in history:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


class Agent:
    """An autonomous agent that can converse, use tools, and reason.

    Args:
        name: A human-readable name for the agent.
        model: The LLM backend used for generation.
        system_prompt: Base system prompt describing the agent's role.
        tools: Optional list of :class:`Tool` instances the agent may call.
        enable_cot: Whether to inject chain-of-thought instructions and
            parse ``<think>`` blocks from model output.
        max_tool_rounds: Maximum number of consecutive tool-call rounds
            before the agent stops and returns whatever it has.
    """

    def __init__(
        self,
        name: str,
        model: BaseModel,
        system_prompt: str = "You are a helpful assistant.",
        tools: list[Tool] | None = None,
        enable_cot: bool = True,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.enable_cot = enable_cot
        self.max_tool_rounds = max_tool_rounds

        self._registry = ToolRegistry()
        for t in tools or []:
            self._registry.register(t)

        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> AgentResponse:
        """Run the agent loop for a single user turn.

        Steps:
            1. Append the user message to conversation history.
            2. Build the prompt (system + tools + history).
            3. Generate a completion.
            4. If CoT is enabled, extract ``<think>`` blocks.
            5. If the output contains a tool call, execute it, append the
               result to history, and go back to step 2.
            6. Return the final ``AgentResponse``.
        """
        self._history.append({"role": "user", "content": user_message})

        all_thinking: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        last_completion: CompletionResult | None = None

        effective_system = (
            inject_cot_instruction(self.system_prompt)
            if self.enable_cot
            else self.system_prompt
        )
        tool_prompt = self._registry.to_system_prompt()

        for _ in range(self.max_tool_rounds):
            prompt = _build_prompt(effective_system, self._history, tool_prompt)
            completion = self.model.generate(prompt, max_tokens=1024)
            last_completion = completion
            raw_text = completion.text

            # -- CoT extraction --
            if self.enable_cot:
                visible, thinking = parse_thinking(raw_text)
                if thinking:
                    all_thinking.append(thinking)
            else:
                visible = raw_text

            # -- Tool call detection --
            tool_call = _extract_tool_call(visible)
            if tool_call is not None:
                try:
                    result = execute_tool_call(self._registry, tool_call)
                except (KeyError, ValueError) as exc:
                    result = f"Error: {exc}"

                tool_calls.append(
                    ToolCallRecord(
                        name=tool_call["name"],
                        arguments=tool_call["arguments"],
                        result=result,
                    )
                )

                # Feed tool result back into conversation.
                self._history.append({"role": "assistant", "content": raw_text})
                self._history.append(
                    {"role": "tool", "content": f"[{tool_call['name']}] {result}"}
                )
                continue

            # No tool call — we have the final answer.
            self._history.append({"role": "assistant", "content": visible})
            break
        else:
            # Exhausted tool rounds; use last visible output as the answer.
            visible = raw_text  # type: ignore[possibly-undefined]
            if self.enable_cot:
                visible, _ = parse_thinking(visible)
            self._history.append({"role": "assistant", "content": visible})

        logprobs = (
            last_completion.logprob_result if last_completion else None
        )

        return AgentResponse(
            content=visible,
            thinking="\n".join(all_thinking),
            tool_calls=tool_calls,
            logprobs=logprobs,
        )

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a copy of the conversation history."""
        return list(self._history)
