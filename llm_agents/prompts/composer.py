"""Compose complex prompts by layering templates with conditional sections."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_agents.prompts.template import PromptTemplate


@dataclass
class _Section:
    """Internal representation of a prompt section."""

    template: PromptTemplate
    condition: str | None = None


class PromptComposer:
    """Build complex prompts by layering multiple templates.

    Supports unconditional sections, conditional sections (included only
    when a named variable is truthy), loops over example lists, and
    template includes.

    Example::

        composer = PromptComposer()
        composer.add(PromptTemplate("You are a {{ role }}."))
        composer.add_conditional(
            "tools",
            PromptTemplate("Available tools: {{ tools }}"),
        )
        composer.add_loop("examples", PromptTemplate("- {{ item }}"))
        result = composer.render(
            role="analyst",
            tools="calculator, search",
            examples=["2+2=4", "3+3=6"],
        )
    """

    def __init__(self, separator: str = "\n\n") -> None:
        self._sections: list[_Section] = []
        self._loops: list[tuple[str, PromptTemplate]] = []
        self._includes: list[PromptTemplate] = []
        self._separator = separator

    def add(self, template: PromptTemplate) -> PromptComposer:
        """Add an unconditional section.

        Args:
            template: Template to always include.

        Returns:
            Self for chaining.
        """
        self._sections.append(_Section(template=template))
        return self

    def add_conditional(
        self, condition_var: str, template: PromptTemplate
    ) -> PromptComposer:
        """Add a section included only when *condition_var* is provided and truthy.

        Args:
            condition_var: Name of the variable that gates this section.
            template: Template to include when condition is met.

        Returns:
            Self for chaining.
        """
        self._sections.append(_Section(template=template, condition=condition_var))
        return self

    def add_loop(self, list_var: str, item_template: PromptTemplate) -> PromptComposer:
        """Add a section that repeats for each item in a list variable.

        The list variable should be a ``list[str]`` in the render kwargs.
        Each item is available as ``{{ item }}`` in *item_template*.

        Args:
            list_var: Name of the list variable.
            item_template: Template rendered once per item.

        Returns:
            Self for chaining.
        """
        self._loops.append((list_var, item_template))
        return self

    def add_include(self, template: PromptTemplate) -> PromptComposer:
        """Add a template that is always appended after all sections and loops.

        Args:
            template: Template to include at the end.

        Returns:
            Self for chaining.
        """
        self._includes.append(template)
        return self

    def render(self, **kwargs: str | list[str]) -> str:
        """Render all sections into a single prompt string.

        Args:
            **kwargs: Variable values. List values are used by loop sections;
                string values are used by regular and conditional sections.

        Returns:
            The composed prompt string.
        """
        str_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, str)}
        list_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, list)}

        parts: list[str] = []

        for section in self._sections:
            if section.condition is not None:
                if not kwargs.get(section.condition):
                    continue
            try:
                parts.append(section.template.render(**str_kwargs))
            except ValueError:
                # Skip sections with missing variables rather than failing
                continue

        for list_var, item_template in self._loops:
            items = list_kwargs.get(list_var, [])
            for item in items:
                parts.append(item_template.render(item=item, **str_kwargs))

        for include in self._includes:
            try:
                parts.append(include.render(**str_kwargs))
            except ValueError:
                continue

        return self._separator.join(parts)
