"""Model interfaces and types."""

from llm_agents.models.types import CompletionResult, LogProbResult, TokenLogProb
from llm_agents.models.base import BaseModel
from llm_agents.models.registry import (
    ModelRegistry,
    register_model,
    get_model,
    list_models,
)

# Lazy imports for backends with heavy dependencies.
def __getattr__(name: str):
    if name == "OpenAIModel":
        from llm_agents.models.openai_model import OpenAIModel
        return OpenAIModel
    if name == "HFModel":
        from llm_agents.models.hf_model import HFModel
        return HFModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from llm_agents.models.streaming import (
    CollectStreamCallback,
    PrintStreamCallback,
    StreamCallback,
    StreamChunk,
    StreamingResult,
    simulate_stream,
    stream_with_callback,
)

__all__ = [
    "BaseModel",
    "CollectStreamCallback",
    "CompletionResult",
    "HFModel",
    "LogProbResult",
    "ModelRegistry",
    "OpenAIModel",
    "PrintStreamCallback",
    "StreamCallback",
    "StreamChunk",
    "StreamingResult",
    "TokenLogProb",
    "get_model",
    "list_models",
    "register_model",
    "simulate_stream",
    "stream_with_callback",
]
