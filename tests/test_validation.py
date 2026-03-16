"""Tests for Extension 2: Structured Output & Validation."""

from __future__ import annotations

import json

import pytest

from llm_agents.validation.formats import (
    constrained_choice,
    csv_row,
    json_array,
    json_object,
    markdown_table,
    yaml_document,
)
from llm_agents.validation.schema import (
    OutputSchema,
    ValidationResult,
    extract_json,
    validate,
)


# ---------------------------------------------------------------------------
# extract_json tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_bare_object(self):
        text = 'Here is the answer: {"key": "value"}'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_bare_array(self):
        text = "The list is: [1, 2, 3]"
        result = extract_json(text)
        assert result == [1, 2, 3]

    def test_fenced_json(self):
        text = '```json\n{"name": "test"}\n```'
        result = extract_json(text)
        assert result == {"name": "test"}

    def test_fenced_without_language(self):
        text = '```\n{"a": 1}\n```'
        result = extract_json(text)
        assert result == {"a": 1}

    def test_trailing_comma_removed(self):
        text = '{"a": 1, "b": 2,}'
        result = extract_json(text)
        assert result == {"a": 1, "b": 2}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("Just plain text with no JSON at all.")

    def test_nested_object(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        text = f"Result: {json.dumps(data)}"
        result = extract_json(text)
        assert result == data


# ---------------------------------------------------------------------------
# validate tests
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_object(self):
        schema = OutputSchema(
            json_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            }
        )
        result = validate('{"name": "Alice"}', schema)
        assert result.valid is True
        assert result.data == {"name": "Alice"}

    def test_missing_required_field(self):
        schema = OutputSchema(
            json_schema={
                "type": "object",
                "required": ["name", "age"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            }
        )
        result = validate('{"name": "Alice"}', schema)
        assert result.valid is False
        assert any("age" in e for e in result.errors)

    def test_wrong_type(self):
        schema = OutputSchema(json_schema={"type": "array"})
        result = validate('{"not": "array"}', schema)
        assert result.valid is False

    def test_no_json_in_text(self):
        schema = OutputSchema(json_schema={"type": "object"})
        result = validate("No JSON here.", schema)
        assert result.valid is False

    def test_enum_valid(self):
        schema = OutputSchema(
            json_schema={"type": "string", "enum": ["yes", "no"]}
        )
        result = validate('"yes"', schema)
        assert result.valid is True

    def test_enum_invalid(self):
        schema = OutputSchema(
            json_schema={"type": "string", "enum": ["yes", "no"]}
        )
        result = validate('"maybe"', schema)
        assert result.valid is False

    def test_array_items_validation(self):
        schema = OutputSchema(
            json_schema={
                "type": "array",
                "items": {"type": "string"},
            }
        )
        result = validate('["a", "b", "c"]', schema)
        assert result.valid is True

    def test_array_items_wrong_type(self):
        schema = OutputSchema(
            json_schema={
                "type": "array",
                "items": {"type": "string"},
            }
        )
        result = validate('[1, 2, 3]', schema)
        assert result.valid is False


# ---------------------------------------------------------------------------
# OutputSchema.validate shortcut
# ---------------------------------------------------------------------------


class TestOutputSchemaValidate:
    def test_method_delegates(self):
        schema = OutputSchema(json_schema={"type": "object"})
        result = schema.validate('{"key": "val"}')
        assert result.valid is True


# ---------------------------------------------------------------------------
# Pre-built formats
# ---------------------------------------------------------------------------


class TestFormats:
    def test_json_object(self):
        schema = json_object(required_fields=["id"])
        result = validate('{"id": 1}', schema)
        assert result.valid is True

    def test_json_object_missing_field(self):
        schema = json_object(required_fields=["id"])
        result = validate('{"name": "test"}', schema)
        assert result.valid is False

    def test_json_array(self):
        schema = json_array()
        result = validate("[1, 2, 3]", schema)
        assert result.valid is True

    def test_csv_row(self):
        schema = csv_row()
        result = validate('["a", "b", "c"]', schema)
        assert result.valid is True

    def test_markdown_table(self):
        schema = markdown_table()
        result = validate('[{"col1": "a", "col2": "b"}]', schema)
        assert result.valid is True

    def test_yaml_document(self):
        schema = yaml_document()
        result = validate('{"key": "value"}', schema)
        assert result.valid is True

    def test_constrained_choice_valid(self):
        schema = constrained_choice(["yes", "no", "maybe"])
        result = validate('"yes"', schema)
        assert result.valid is True

    def test_constrained_choice_invalid(self):
        schema = constrained_choice(["yes", "no"])
        result = validate('"maybe"', schema)
        assert result.valid is False
