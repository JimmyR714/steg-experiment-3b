"""Extension 15: Output filtering for policy violation detection.

Scans agent output for content that may violate safety policies and
provides configurable block/warn/log behavior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FilterAction(Enum):
    """What to do when a policy violation is detected."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    LOG = "log"


@dataclass(frozen=True)
class ViolationResult:
    """Result of an output policy check.

    Attributes:
        is_violation: Whether a violation was detected.
        action: The recommended action to take.
        categories: Categories of violations found.
        details: Human-readable explanation.
    """

    is_violation: bool
    action: FilterAction = FilterAction.ALLOW
    categories: tuple[str, ...] = ()
    details: str = ""


# Output policy patterns
_OUTPUT_POLICIES: list[tuple[str, re.Pattern[str], FilterAction]] = [
    (
        "system_prompt_leak",
        re.compile(
            r"(my\s+system\s+prompt|my\s+instructions\s+are|I\s+was\s+told\s+to)",
            re.IGNORECASE,
        ),
        FilterAction.BLOCK,
    ),
    (
        "harmful_instructions",
        re.compile(
            r"(how\s+to\s+(hack|exploit|attack)\b|step.by.step.*(malware|virus|ransomware))",
            re.IGNORECASE,
        ),
        FilterAction.BLOCK,
    ),
    (
        "personal_info_pattern",
        re.compile(
            r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN-like pattern
        ),
        FilterAction.WARN,
    ),
    (
        "credential_leak",
        re.compile(
            r"(password|api[_\s]?key|secret[_\s]?key|access[_\s]?token)\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
        FilterAction.BLOCK,
    ),
]


class OutputFilter:
    """Scans agent output for policy violations.

    Args:
        policies: Custom policies as (name, pattern, action) tuples.
            If None, uses built-in policies.
        default_action: Action for violations from built-in policies.
    """

    def __init__(
        self,
        policies: list[tuple[str, re.Pattern[str], FilterAction]] | None = None,
        default_action: FilterAction = FilterAction.WARN,
    ) -> None:
        self._policies = policies if policies is not None else list(_OUTPUT_POLICIES)
        self.default_action = default_action

    def scan(self, text: str) -> ViolationResult:
        """Scan output text against all policies.

        Args:
            text: Agent output to check.

        Returns:
            ViolationResult describing any violations found.
        """
        violations: list[str] = []
        max_action = FilterAction.ALLOW

        _actions = list(FilterAction)

        for name, pattern, action in self._policies:
            if pattern.search(text):
                violations.append(name)
                if _actions.index(action) > _actions.index(max_action):
                    max_action = action

        if not violations:
            return ViolationResult(is_violation=False)

        return ViolationResult(
            is_violation=True,
            action=max_action,
            categories=tuple(violations),
            details=f"Policy violations: {', '.join(violations)}",
        )

    def add_policy(
        self,
        name: str,
        pattern: re.Pattern[str],
        action: FilterAction = FilterAction.WARN,
    ) -> None:
        """Add a custom policy rule.

        Args:
            name: Name of the policy.
            pattern: Regex pattern to match.
            action: Action to take on match.
        """
        self._policies.append((name, pattern, action))


class ContentClassifier:
    """Lightweight content safety classifier.

    Uses keyword-based heuristics to classify content into safety categories.
    For production use, this would wrap a dedicated classification model.

    Args:
        categories: Dict mapping category name to list of indicator keywords.
    """

    DEFAULT_CATEGORIES: dict[str, list[str]] = {
        "violence": ["kill", "murder", "attack", "weapon", "bomb"],
        "hate_speech": ["racist", "sexist", "homophobic", "slur"],
        "self_harm": ["suicide", "self-harm", "cutting"],
        "sexual": ["explicit", "pornographic", "nsfw"],
    }

    def __init__(
        self,
        categories: dict[str, list[str]] | None = None,
    ) -> None:
        self._categories = categories or self.DEFAULT_CATEGORIES

    def classify(self, text: str) -> dict[str, float]:
        """Classify text into safety categories.

        Returns a dict mapping category name to a confidence score (0-1).
        Higher scores indicate stronger match.

        Args:
            text: Text to classify.

        Returns:
            Dict of category -> confidence score.
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        results: dict[str, float] = {}

        for category, keywords in self._categories.items():
            matches = sum(1 for kw in keywords if kw in words)
            results[category] = min(1.0, matches / max(1, len(keywords) * 0.3))

        return results
