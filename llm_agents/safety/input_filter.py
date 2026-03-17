"""Extension 15: Input filtering for prompt injection detection.

Scans incoming text for common prompt injection patterns and provides
sanitization utilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ThreatLevel(Enum):
    """Severity level for detected threats."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class InjectionResult:
    """Result of an injection detection scan.

    Attributes:
        is_suspicious: Whether any injection patterns were detected.
        threat_level: Severity of the detected threat.
        matched_patterns: List of pattern names that matched.
        details: Human-readable description of findings.
    """

    is_suspicious: bool
    threat_level: ThreatLevel = ThreatLevel.NONE
    matched_patterns: tuple[str, ...] = ()
    details: str = ""


# Common prompt injection patterns
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], ThreatLevel]] = [
    (
        "ignore_previous",
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
            re.IGNORECASE,
        ),
        ThreatLevel.HIGH,
    ),
    (
        "system_prompt_leak",
        re.compile(
            r"(reveal|show|print|output|repeat)\s+(your\s+)?(system\s+prompt|instructions|rules)",
            re.IGNORECASE,
        ),
        ThreatLevel.MEDIUM,
    ),
    (
        "role_override",
        re.compile(
            r"you\s+are\s+now\s+(a|an|the)\s+",
            re.IGNORECASE,
        ),
        ThreatLevel.MEDIUM,
    ),
    (
        "delimiter_attack",
        re.compile(
            r"(---+|===+|###)\s*(system|admin|root|override)",
            re.IGNORECASE,
        ),
        ThreatLevel.HIGH,
    ),
    (
        "encoding_bypass",
        re.compile(
            r"(base64|rot13|hex)\s*(encode|decode|convert)",
            re.IGNORECASE,
        ),
        ThreatLevel.LOW,
    ),
    (
        "jailbreak_attempt",
        re.compile(
            r"(DAN|do\s+anything\s+now|jailbreak|bypass\s+restrictions)",
            re.IGNORECASE,
        ),
        ThreatLevel.CRITICAL,
    ),
    (
        "instruction_injection",
        re.compile(
            r"\[?(system|admin|instruction)\]?\s*:\s*",
            re.IGNORECASE,
        ),
        ThreatLevel.MEDIUM,
    ),
]


def detect_injection(text: str) -> InjectionResult:
    """Scan text for prompt injection patterns.

    Args:
        text: The input text to scan.

    Returns:
        An InjectionResult describing what was found.
    """
    matched: list[str] = []
    max_threat = ThreatLevel.NONE
    _levels = list(ThreatLevel)

    for name, pattern, threat in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched.append(name)
            if _levels.index(threat) > _levels.index(max_threat):
                max_threat = threat

    if not matched:
        return InjectionResult(is_suspicious=False)

    details = f"Detected {len(matched)} suspicious pattern(s): {', '.join(matched)}"
    return InjectionResult(
        is_suspicious=True,
        threat_level=max_threat,
        matched_patterns=tuple(matched),
        details=details,
    )


def sanitize(text: str) -> str:
    """Remove or escape suspicious patterns from input text.

    This is a best-effort sanitization that escapes common injection
    delimiters and removes obvious override attempts.

    Args:
        text: The input text to sanitize.

    Returns:
        Sanitized text.
    """
    # Escape delimiter sequences
    text = re.sub(r"---+", "—", text)
    text = re.sub(r"===+", "≡", text)

    # Remove instruction injection markers
    text = re.sub(
        r"\[?(system|admin|instruction)\]?\s*:",
        r"[\1]",
        text,
        flags=re.IGNORECASE,
    )

    # Neutralize "ignore previous" patterns
    text = re.sub(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
        "[filtered]",
        text,
        flags=re.IGNORECASE,
    )

    return text


class InputFilter:
    """Configurable input filter that scans text for injection patterns.

    Args:
        min_threat_level: Minimum threat level to flag. Anything below
            this level is considered safe.
        custom_patterns: Additional (name, regex, threat_level) tuples.
    """

    def __init__(
        self,
        min_threat_level: ThreatLevel = ThreatLevel.LOW,
        custom_patterns: list[tuple[str, re.Pattern[str], ThreatLevel]] | None = None,
    ) -> None:
        self.min_threat_level = min_threat_level
        self._patterns = list(_INJECTION_PATTERNS)
        if custom_patterns:
            self._patterns.extend(custom_patterns)

    def scan(self, text: str) -> InjectionResult:
        """Scan text against all configured patterns.

        Args:
            text: Input text to scan.

        Returns:
            InjectionResult with matches at or above the minimum threat level.
        """
        matched: list[str] = []
        max_threat = ThreatLevel.NONE

        _levels = list(ThreatLevel)

        for name, pattern, threat in self._patterns:
            if _levels.index(threat) < _levels.index(self.min_threat_level):
                continue
            if pattern.search(text):
                matched.append(name)
                if _levels.index(threat) > _levels.index(max_threat):
                    max_threat = threat

        if not matched:
            return InjectionResult(is_suspicious=False)

        return InjectionResult(
            is_suspicious=True,
            threat_level=max_threat,
            matched_patterns=tuple(matched),
            details=f"Detected: {', '.join(matched)}",
        )
