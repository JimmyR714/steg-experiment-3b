"""Chain-of-thought support for LLM agents."""

from __future__ import annotations

import re

COT_INSTRUCTION = (
    "\n\nYou may use <think>...</think> tags to reason through problems "
    "step by step before giving your final answer. The content inside "
    "<think> tags will not be shown to the user."
)

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def inject_cot_instruction(system_prompt: str) -> str:
    """Append chain-of-thought instructions to a system prompt.

    Args:
        system_prompt: The original system prompt.

    Returns:
        The system prompt with CoT instructions appended.
    """
    return system_prompt + COT_INSTRUCTION


def parse_thinking(text: str) -> tuple[str, str]:
    """Separate ``<think>`` blocks from visible content.

    Args:
        text: Raw model output that may contain ``<think>...</think>`` blocks.

    Returns:
        A ``(visible, thinking)`` tuple.  *visible* is the text with all
        ``<think>`` blocks removed (and leading/trailing whitespace stripped).
        *thinking* is the concatenated content of all ``<think>`` blocks,
        separated by newlines if there are multiple.
    """
    thinking_parts = _THINK_PATTERN.findall(text)
    thinking = "\n".join(part.strip() for part in thinking_parts)
    visible = _THINK_PATTERN.sub("", text).strip()
    return visible, thinking
