"""Pre-built prompt templates for common patterns."""

from __future__ import annotations

from llm_agents.prompts.template import PromptTemplate

CHAIN_OF_THOUGHT = PromptTemplate(
    template_str=(
        "{{ task }}\n\n"
        "Think through this step-by-step:\n"
        "1. First, identify the key aspects of the problem.\n"
        "2. Consider each aspect carefully.\n"
        "3. Arrive at your conclusion.\n\n"
        "Show your reasoning inside <think> tags, then provide your final answer."
    ),
    variables=["task"],
)
"""Standard chain-of-thought elicitation template."""

FEW_SHOT = PromptTemplate(
    template_str=(
        "{{ instruction }}\n\n"
        "Here are some examples:\n\n"
        "{{ examples }}\n\n"
        "Now complete the following:\n"
        "{{ input }}"
    ),
    variables=["instruction", "examples", "input"],
)
"""Few-shot prompt with instruction, examples, and input."""

PERSONA = PromptTemplate(
    template_str=(
        "You are {{ name }}, {{ description }}.\n\n"
        "Your key traits:\n{{ traits }}\n\n"
        "Always stay in character and respond accordingly."
    ),
    variables=["name", "description", "traits"],
)
"""Role-play system prompt with configurable persona traits."""

STRUCTURED_OUTPUT = PromptTemplate(
    template_str=(
        "{{ task }}\n\n"
        "You MUST respond in the following format:\n"
        "{{ format_spec }}\n\n"
        "Do not include any text outside of this format."
    ),
    variables=["task", "format_spec"],
)
"""Instructs the model to output in a specific format."""

SELF_CRITIQUE = PromptTemplate(
    template_str=(
        "Review the following response for accuracy, completeness, and quality:\n\n"
        "Original task: {{ task }}\n\n"
        "Response to review:\n{{ response }}\n\n"
        "Provide your critique as JSON:\n"
        '{"accept": true/false, "feedback": "...", "score": 0.0-1.0}'
    ),
    variables=["task", "response"],
)
"""Reflection-prompting template for self-critique."""

TOOL_USE = PromptTemplate(
    template_str=(
        "You have access to the following tools:\n\n"
        "{{ tool_descriptions }}\n\n"
        "To use a tool, respond with a JSON block:\n"
        "```tool_call\n"
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "```\n\n"
        "Use tools when they would help you answer the user's question.\n\n"
        "{{ task }}"
    ),
    variables=["tool_descriptions", "task"],
)
"""Standardized tool-use instructions."""


def format_examples(examples: list[dict[str, str]]) -> str:
    """Format a list of input/output example dicts into a string.

    Args:
        examples: A list of dicts with ``"input"`` and ``"output"`` keys.

    Returns:
        Formatted string suitable for the ``examples`` variable of
        :data:`FEW_SHOT`.
    """
    parts: list[str] = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"  Input: {ex['input']}")
        parts.append(f"  Output: {ex['output']}")
        parts.append("")
    return "\n".join(parts).rstrip()
