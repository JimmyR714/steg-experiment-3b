"""Prompt templating with variable substitution and multi-turn support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_VARIABLE_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass
class PromptTemplate:
    """A template string with ``{{ variable }}`` placeholders.

    Attributes:
        template_str: The raw template string with ``{{ var }}`` placeholders.
        variables: Names of variables expected by this template.  If not
            provided they are extracted automatically from *template_str*.
    """

    template_str: str
    variables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.variables:
            self.variables = list(dict.fromkeys(_VARIABLE_RE.findall(self.template_str)))

    def render(self, **kwargs: str) -> str:
        """Render the template by substituting variables.

        Args:
            **kwargs: Variable values keyed by name.

        Returns:
            The rendered string.

        Raises:
            ValueError: If a required variable is missing.
        """
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing template variables: {', '.join(missing)}")

        def _replace(match: re.Match) -> str:
            name = match.group(1)
            return kwargs.get(name, match.group(0))

        return _VARIABLE_RE.sub(_replace, self.template_str)


@dataclass
class ChatTemplate:
    """Structured multi-turn template with system, user, and assistant parts.

    Attributes:
        system: Template for the system message.
        user: Template for the user message.
        assistant: Optional template for a prefilled assistant message.
    """

    system: PromptTemplate
    user: PromptTemplate
    assistant: PromptTemplate | None = None

    def render(self, **kwargs: str) -> list[dict[str, str]]:
        """Render into a list of message dicts.

        Args:
            **kwargs: Variable values shared across all parts.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system.render(**kwargs)},
            {"role": "user", "content": self.user.render(**kwargs)},
        ]
        if self.assistant is not None:
            messages.append(
                {"role": "assistant", "content": self.assistant.render(**kwargs)}
            )
        return messages


def render(template: PromptTemplate | str, **kwargs: str) -> str:
    """Convenience function to render a template or raw string.

    Args:
        template: A :class:`PromptTemplate` or a plain string with
            ``{{ variable }}`` placeholders.
        **kwargs: Variable values.

    Returns:
        The rendered string.
    """
    if isinstance(template, str):
        template = PromptTemplate(template)
    return template.render(**kwargs)
