"""Result dataclasses for standard agent task workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DebateResult:
    """Result of a multi-agent debate.

    Attributes:
        topic: The debate topic.
        rounds: List of (pro_argument, con_argument) pairs for each round.
        winner: The side chosen by the judge ("pro" or "con").
        judgment: The judge's reasoning.
    """

    topic: str
    rounds: list[tuple[str, str]] = field(default_factory=list)
    winner: str = ""
    judgment: str = ""


@dataclass
class ClassifyResult:
    """Result of a classification task.

    Attributes:
        label: The predicted label.
        probabilities: Mapping of label to probability score.
    """

    label: str
    probabilities: dict[str, float] = field(default_factory=dict)
