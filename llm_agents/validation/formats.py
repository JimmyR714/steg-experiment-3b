"""Pre-built output schemas for common formats."""

from __future__ import annotations

from llm_agents.validation.schema import OutputSchema


def json_object(
    required_fields: list[str] | None = None,
    properties: dict | None = None,
) -> OutputSchema:
    """Schema for a JSON object with optional required fields.

    Args:
        required_fields: List of required field names.
        properties: JSON Schema properties definition.

    Returns:
        An :class:`OutputSchema` for a JSON object.
    """
    schema: dict = {"type": "object"}
    if required_fields:
        schema["required"] = required_fields
    if properties:
        schema["properties"] = properties
    return OutputSchema(
        json_schema=schema,
        description="A JSON object" + (
            f" with required fields: {', '.join(required_fields)}"
            if required_fields else ""
        ),
    )


def json_array(items_schema: dict | None = None) -> OutputSchema:
    """Schema for a JSON array.

    Args:
        items_schema: Optional JSON Schema for array items.

    Returns:
        An :class:`OutputSchema` for a JSON array.
    """
    schema: dict = {"type": "array"}
    if items_schema:
        schema["items"] = items_schema
    return OutputSchema(json_schema=schema, description="A JSON array")


def csv_row() -> OutputSchema:
    """Schema that accepts a JSON array of strings (representing a CSV row)."""
    return OutputSchema(
        json_schema={"type": "array", "items": {"type": "string"}},
        description="A JSON array of strings representing a CSV row",
    )


def markdown_table() -> OutputSchema:
    """Schema for a JSON array of objects (representing a markdown table).

    Each object in the array represents a row, with keys as column headers.
    """
    return OutputSchema(
        json_schema={"type": "array", "items": {"type": "object"}},
        description="A JSON array of objects representing table rows",
    )


def yaml_document() -> OutputSchema:
    """Schema for a JSON object (representing a YAML document)."""
    return OutputSchema(
        json_schema={"type": "object"},
        description="A JSON object (YAML-compatible document)",
    )


def constrained_choice(options: list[str]) -> OutputSchema:
    """Schema that constrains output to one of the given options.

    The output must be a JSON string matching one of *options*.

    Args:
        options: Allowed output values.

    Returns:
        An :class:`OutputSchema` that only accepts the given choices.
    """
    return OutputSchema(
        json_schema={"type": "string", "enum": options},
        description=f"One of: {', '.join(options)}",
    )
