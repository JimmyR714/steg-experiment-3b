"""Output schema validation and JSON extraction utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Result of validating agent output against a schema.

    Attributes:
        valid: Whether the output passed validation.
        data: The parsed/extracted data if valid, else *None*.
        errors: List of validation error messages.
    """

    valid: bool
    data: Any = None
    errors: list[str] = field(default_factory=list)


@dataclass
class OutputSchema:
    """Schema definition for structured output validation.

    Args:
        json_schema: A JSON Schema dict that the output must conform to.
        description: Human-readable description of the expected format.
    """

    json_schema: dict[str, Any]
    description: str = ""

    def validate(self, text: str) -> ValidationResult:
        """Validate *text* against this schema.

        Attempts to extract JSON from the text, then checks it against the
        schema using lightweight validation.
        """
        return validate(text, self)


def extract_json(text: str) -> Any:
    """Robustly extract JSON from text that may contain markdown fences or prose.

    Handles:
    - ````` ```json ... ``` ````` fences
    - ````` ``` ... ``` ````` fences without language tag
    - Bare JSON objects/arrays/strings/numbers
    - Trailing commas (removed before parsing)

    Args:
        text: The text to extract JSON from.

    Returns:
        The parsed JSON value.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    # Try markdown code fences first
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    for match in fence_pattern.finditer(text):
        candidate = match.group(1).strip()
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    # Try to find bare JSON objects or arrays — collect all candidates and
    # return the longest valid parse (avoids matching an inner object when
    # the whole text contains an array of objects).
    candidates: list[tuple[int, Any]] = []
    for pattern in [
        re.compile(r"(\[.*\])", re.DOTALL),
        re.compile(r"(\{.*\})", re.DOTALL),
    ]:
        match = pattern.search(text)
        if match:
            parsed = _try_parse(match.group(1))
            if parsed is not None:
                candidates.append((len(match.group(1)), parsed))

    if candidates:
        # Return the candidate from the longest match
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    # Try parsing the entire stripped text as JSON (handles bare strings, numbers)
    stripped = text.strip()
    parsed = _try_parse(stripped)
    if parsed is not None:
        return parsed

    raise ValueError(f"No valid JSON found in text: {text[:200]!r}...")


def _try_parse(text: str) -> Any:
    """Attempt to parse JSON, fixing trailing commas. Returns None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try removing trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


def _validate_type(value: Any, schema: dict[str, Any]) -> list[str]:
    """Lightweight JSON Schema type validation.  Returns a list of errors."""
    errors: list[str] = []
    expected_type = schema.get("type")

    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }

    if expected_type and expected_type in type_map:
        py_type = type_map[expected_type]
        if not isinstance(value, py_type):
            errors.append(
                f"Expected type {expected_type!r}, got {type(value).__name__!r}"
            )
            return errors

    if expected_type == "object" and isinstance(value, dict):
        # Check required fields
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"Missing required field: {req!r}")
        # Check property types
        props = schema.get("properties", {})
        for key, prop_schema in props.items():
            if key in value:
                errors.extend(_validate_type(value[key], prop_schema))

    if expected_type == "array" and isinstance(value, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(value):
                item_errors = _validate_type(item, items_schema)
                errors.extend(f"[{i}]: {e}" for e in item_errors)

    # Enum check
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"Value {value!r} not in enum {schema['enum']}")

    return errors


def validate(text: str, schema: OutputSchema) -> ValidationResult:
    """Validate *text* against an :class:`OutputSchema`.

    Args:
        text: The agent output text.
        schema: The schema to validate against.

    Returns:
        A :class:`ValidationResult` with validation outcome.
    """
    try:
        data = extract_json(text)
    except ValueError as exc:
        return ValidationResult(valid=False, errors=[str(exc)])

    errors = _validate_type(data, schema.json_schema)
    if errors:
        return ValidationResult(valid=False, data=data, errors=errors)
    return ValidationResult(valid=True, data=data)
