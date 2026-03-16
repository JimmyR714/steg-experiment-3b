"""Prompt complexity classification for intelligent model routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Complexity(Enum):
    """Prompt complexity level."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ClassificationResult:
    """Result of complexity classification.

    Attributes:
        complexity: The assessed complexity level.
        score: Numeric score in [0, 1] where higher means more complex.
        features: Dictionary of feature names to their computed values.
    """

    complexity: Complexity
    score: float
    features: dict[str, float]


# Patterns indicating higher complexity
_CODE_PATTERN = re.compile(r"```|def |class |function |import |#include")
_MATH_PATTERN = re.compile(r"[∫∑∏√]|\\frac|\\int|equation|derivative|integral|matrix")
_MULTI_STEP_INDICATORS = re.compile(
    r"\b(first|then|next|after that|finally|step \d|compare|analyze|evaluate"
    r"|design|implement|architect|optimize)\b",
    re.IGNORECASE,
)
_QUESTION_WORDS = re.compile(
    r"\b(what|who|when|where|why|how|which|explain|describe|compare)\b",
    re.IGNORECASE,
)


class ComplexityClassifier:
    """Classifies prompts by complexity using heuristic features.

    Features computed:
    - Token count (approximated by word count).
    - Presence of code or math patterns.
    - Number of multi-step indicators.
    - Question type complexity.

    Args:
        simple_threshold: Score below which prompts are classified as simple.
        hard_threshold: Score above which prompts are classified as hard.
    """

    def __init__(
        self,
        simple_threshold: float = 0.3,
        hard_threshold: float = 0.7,
    ) -> None:
        self.simple_threshold = simple_threshold
        self.hard_threshold = hard_threshold

    def classify(self, prompt: str) -> ClassificationResult:
        """Classify the complexity of a prompt.

        Args:
            prompt: The input prompt to classify.

        Returns:
            A :class:`ClassificationResult` with complexity and features.
        """
        features: dict[str, float] = {}

        # Feature 1: Length (word count as proxy for token count)
        words = prompt.split()
        word_count = len(words)
        features["word_count"] = float(word_count)
        # Normalize: 0-50 words = low, 50-200 = medium, 200+ = high
        length_score = min(word_count / 200.0, 1.0)
        features["length_score"] = length_score

        # Feature 2: Code presence
        code_matches = len(_CODE_PATTERN.findall(prompt))
        features["code_indicators"] = float(code_matches)
        code_score = min(code_matches / 3.0, 1.0)

        # Feature 3: Math presence
        math_matches = len(_MATH_PATTERN.findall(prompt))
        features["math_indicators"] = float(math_matches)
        math_score = min(math_matches / 2.0, 1.0)

        # Feature 4: Multi-step indicators
        multi_step_matches = len(_MULTI_STEP_INDICATORS.findall(prompt))
        features["multi_step_indicators"] = float(multi_step_matches)
        multi_step_score = min(multi_step_matches / 4.0, 1.0)

        # Feature 5: Question complexity
        question_matches = _QUESTION_WORDS.findall(prompt)
        features["question_words"] = float(len(question_matches))
        # "why" and "how" are typically harder than "what" or "when"
        hard_questions = sum(
            1 for q in question_matches if q.lower() in ("why", "how", "explain", "compare")
        )
        question_score = min(hard_questions / 2.0, 1.0)

        # Feature 6: Sentence count (more sentences → more complex)
        sentence_count = max(len(re.split(r"[.!?]+", prompt)) - 1, 1)
        features["sentence_count"] = float(sentence_count)
        sentence_score = min(sentence_count / 5.0, 1.0)

        # Weighted aggregate
        score = (
            0.20 * length_score
            + 0.20 * code_score
            + 0.15 * math_score
            + 0.20 * multi_step_score
            + 0.15 * question_score
            + 0.10 * sentence_score
        )
        features["aggregate_score"] = score

        if score <= self.simple_threshold:
            complexity = Complexity.SIMPLE
        elif score >= self.hard_threshold:
            complexity = Complexity.HARD
        else:
            complexity = Complexity.MEDIUM

        return ClassificationResult(
            complexity=complexity,
            score=score,
            features=features,
        )
