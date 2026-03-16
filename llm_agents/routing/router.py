"""Model routers that dispatch to appropriate models based on criteria."""

from __future__ import annotations

from typing import Any, Callable

from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult
from llm_agents.routing.classifier import Complexity, ComplexityClassifier


class ModelRouter:
    """Routes prompts to models based on complexity classification.

    Args:
        routes: Mapping of :class:`Complexity` level to :class:`BaseModel`.
        classifier: Optional custom complexity classifier. If *None*, a
            default :class:`ComplexityClassifier` is created.
    """

    def __init__(
        self,
        routes: dict[Complexity, BaseModel],
        classifier: ComplexityClassifier | None = None,
    ) -> None:
        self._routes = routes
        self._classifier = classifier or ComplexityClassifier()

    def route(self, prompt: str) -> BaseModel:
        """Select the appropriate model for the given prompt.

        Args:
            prompt: The input prompt.

        Returns:
            The selected model.

        Raises:
            KeyError: If no model is configured for the detected complexity.
        """
        result = self._classifier.classify(prompt)
        if result.complexity in self._routes:
            return self._routes[result.complexity]
        # Fallback: try MEDIUM, then any available model
        for level in (Complexity.MEDIUM, Complexity.SIMPLE, Complexity.HARD):
            if level in self._routes:
                return self._routes[level]
        raise KeyError(f"No model configured for complexity {result.complexity}")

    def generate(self, prompt: str, **kwargs: Any) -> CompletionResult:
        """Route and generate in one call.

        Args:
            prompt: The input prompt.
            **kwargs: Additional keyword arguments passed to the model's
                ``generate`` method.

        Returns:
            The completion result from the selected model.
        """
        model = self.route(prompt)
        return model.generate(prompt, **kwargs)


class CascadeRouter:
    """Tries models from cheapest to most expensive, escalating on failure.

    The *validator* function checks whether a completion is acceptable.
    If not, the next (more capable) model is tried.

    Args:
        models: Ordered list of models from cheapest to most expensive.
        validator: A callable that takes a :class:`CompletionResult` and
            returns *True* if it is acceptable.
    """

    def __init__(
        self,
        models: list[BaseModel],
        validator: Callable[[CompletionResult], bool] | None = None,
    ) -> None:
        if not models:
            raise ValueError("CascadeRouter requires at least one model")
        self._models = models
        self._validator = validator or (lambda _: True)

    def generate(self, prompt: str, **kwargs: Any) -> CompletionResult:
        """Try each model in order until one produces a valid response.

        Args:
            prompt: The input prompt.
            **kwargs: Additional keyword arguments passed to each model's
                ``generate`` method.

        Returns:
            The first valid :class:`CompletionResult`, or the result from
            the last model if none pass validation.
        """
        for model in self._models:
            result = model.generate(prompt, **kwargs)
            if self._validator(result):
                return result
        # Return the last result even if it didn't pass validation
        return result  # type: ignore[possibly-undefined]


class LatencyRouter:
    """Races multiple models and returns the first valid response.

    Note: In this synchronous implementation, models are tried sequentially
    (true racing would require async). If a model produces a result that
    passes validation within the timeout, it is returned immediately.

    Args:
        models: Models to race.
        validator: Optional validation function for responses.
    """

    def __init__(
        self,
        models: list[BaseModel],
        validator: Callable[[CompletionResult], bool] | None = None,
    ) -> None:
        if not models:
            raise ValueError("LatencyRouter requires at least one model")
        self._models = models
        self._validator = validator or (lambda _: True)

    def generate(self, prompt: str, **kwargs: Any) -> CompletionResult:
        """Try models sequentially, returning the first valid result.

        Args:
            prompt: The input prompt.
            **kwargs: Additional keyword arguments.

        Returns:
            The first valid :class:`CompletionResult`.
        """
        last_result: CompletionResult | None = None
        for model in self._models:
            try:
                result = model.generate(prompt, **kwargs)
                last_result = result
                if self._validator(result):
                    return result
            except Exception:
                continue
        if last_result is not None:
            return last_result
        raise RuntimeError("All models failed to produce a result")
