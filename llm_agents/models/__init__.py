"""Model interfaces and types."""

from llm_agents.models.types import CompletionResult, LogProbResult, TokenLogProb
from llm_agents.models.base import BaseModel

__all__ = [
    "BaseModel",
    "CompletionResult",
    "LogProbResult",
    "TokenLogProb",
]
