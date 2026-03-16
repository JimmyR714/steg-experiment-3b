"""Simple model registry for managing named model instances."""

from __future__ import annotations

from llm_agents.models.base import BaseModel


class ModelRegistry:
    """Registry that maps string names to BaseModel instances."""

    def __init__(self) -> None:
        self._models: dict[str, BaseModel] = {}

    def register_model(self, name: str, model_instance: BaseModel) -> None:
        """Register a model instance under the given name.

        Args:
            name: Unique identifier for the model.
            model_instance: A concrete BaseModel implementation.

        Raises:
            TypeError: If model_instance is not a BaseModel.
            ValueError: If a model with the given name is already registered.
        """
        if not isinstance(model_instance, BaseModel):
            raise TypeError(
                f"Expected a BaseModel instance, got {type(model_instance).__name__}"
            )
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered")
        self._models[name] = model_instance

    def get_model(self, name: str) -> BaseModel:
        """Retrieve a registered model by name.

        Args:
            name: The name the model was registered under.

        Returns:
            The registered BaseModel instance.

        Raises:
            KeyError: If no model is registered with the given name.
        """
        if name not in self._models:
            raise KeyError(f"No model registered with name '{name}'")
        return self._models[name]

    def list_models(self) -> list[str]:
        """Return a sorted list of all registered model names."""
        return sorted(self._models.keys())


# Global default registry instance.
_default_registry = ModelRegistry()


def register_model(name: str, model_instance: BaseModel) -> None:
    """Register a model in the default global registry."""
    _default_registry.register_model(name, model_instance)


def get_model(name: str) -> BaseModel:
    """Get a model from the default global registry."""
    return _default_registry.get_model(name)


def list_models() -> list[str]:
    """List models in the default global registry."""
    return _default_registry.list_models()
