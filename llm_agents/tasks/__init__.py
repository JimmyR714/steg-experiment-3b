"""Standard agent task library for common multi-agent workflows."""

from llm_agents.tasks.standard import (
    chain,
    classify,
    debate,
    map_reduce,
    qa,
    summarize,
)
from llm_agents.tasks.types import ClassifyResult, DebateResult

__all__ = [
    "ClassifyResult",
    "DebateResult",
    "chain",
    "classify",
    "debate",
    "map_reduce",
    "qa",
    "summarize",
]
