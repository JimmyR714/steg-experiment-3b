"""Evaluation datasets and example loaders."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalExample:
    """A single evaluation example.

    Attributes:
        input: The input prompt or question.
        expected: The expected or reference answer.
        metadata: Optional extra information (e.g. category, difficulty).
    """

    input: str
    expected: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalDataset:
    """A collection of evaluation examples.

    Args:
        examples: Initial list of examples.
        name: Optional dataset name.
    """

    def __init__(
        self,
        examples: list[EvalExample] | None = None,
        name: str = "",
    ) -> None:
        self.name = name
        self._examples: list[EvalExample] = list(examples or [])

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> EvalExample:
        return self._examples[index]

    def __iter__(self):
        return iter(self._examples)

    def add(self, example: EvalExample) -> None:
        """Append an example to the dataset.

        Args:
            example: The example to add.
        """
        self._examples.append(example)

    @property
    def examples(self) -> list[EvalExample]:
        """Return a copy of the examples list."""
        return list(self._examples)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path, name: str = "") -> EvalDataset:
        """Load a dataset from a JSON file.

        Expected format: a list of objects with ``"input"``, ``"expected"``,
        and optionally ``"metadata"`` keys.

        Args:
            path: Path to the JSON file.
            name: Optional dataset name.

        Returns:
            A new :class:`EvalDataset`.
        """
        with open(path) as f:
            data = json.load(f)
        examples = [
            EvalExample(
                input=item["input"],
                expected=item["expected"],
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]
        return cls(examples, name=name or Path(path).stem)

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str = "") -> EvalDataset:
        """Load a dataset from a JSONL (JSON Lines) file.

        Each line should be a JSON object with ``"input"`` and ``"expected"``.

        Args:
            path: Path to the JSONL file.
            name: Optional dataset name.

        Returns:
            A new :class:`EvalDataset`.
        """
        examples: list[EvalExample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                examples.append(
                    EvalExample(
                        input=item["input"],
                        expected=item["expected"],
                        metadata=item.get("metadata", {}),
                    )
                )
        return cls(examples, name=name or Path(path).stem)

    @classmethod
    def from_csv(
        cls, path: str | Path, name: str = "",
        input_col: str = "input", expected_col: str = "expected",
    ) -> EvalDataset:
        """Load a dataset from a CSV file.

        Args:
            path: Path to the CSV file.
            name: Optional dataset name.
            input_col: Column name for inputs.
            expected_col: Column name for expected outputs.

        Returns:
            A new :class:`EvalDataset`.
        """
        examples: list[EvalExample] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata = {
                    k: v for k, v in row.items()
                    if k not in (input_col, expected_col)
                }
                examples.append(
                    EvalExample(
                        input=row[input_col],
                        expected=row[expected_col],
                        metadata=metadata,
                    )
                )
        return cls(examples, name=name or Path(path).stem)

    @classmethod
    def from_list(
        cls,
        pairs: list[tuple[str, str]],
        name: str = "",
    ) -> EvalDataset:
        """Create a dataset from a list of (input, expected) tuples.

        Args:
            pairs: List of ``(input, expected)`` tuples.
            name: Optional dataset name.

        Returns:
            A new :class:`EvalDataset`.
        """
        examples = [EvalExample(input=inp, expected=exp) for inp, exp in pairs]
        return cls(examples, name=name)
