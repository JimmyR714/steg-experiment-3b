"""Agent components for autonomous LLM interaction."""

from llm_agents.agents.agent import Agent, AgentResponse, ToolCallRecord
from llm_agents.agents.cot import inject_cot_instruction, parse_thinking
from llm_agents.agents.message_bus import Message, MessageBus
from llm_agents.agents.multi_agent import MultiAgentSystem
from llm_agents.agents.reflection import (
    CritiqueResult,
    PeerCritique,
    ReflectiveAgent,
    SelfCritique,
)
from llm_agents.agents.roles import (
    ANALYST,
    BUILTIN_ROLES,
    CODER,
    COORDINATOR,
    CRITIC,
    FACT_CHECKER,
    RESEARCHER,
    WRITER,
    AgentRole,
    create_agent,
)
from llm_agents.agents.task import TaskResult
from llm_agents.agents.team import AgentTeam, code_review_team, debate_team, research_team
from llm_agents.agents.streaming_agent import StreamingAgent
from llm_agents.agents.fsm import (
    State,
    StateMachineAgent,
    Transition,
    fsm_to_mermaid,
)

__all__ = [
    "ANALYST",
    "Agent",
    "AgentResponse",
    "AgentRole",
    "AgentTeam",
    "BUILTIN_ROLES",
    "CODER",
    "COORDINATOR",
    "CRITIC",
    "CritiqueResult",
    "FACT_CHECKER",
    "Message",
    "MessageBus",
    "MultiAgentSystem",
    "PeerCritique",
    "RESEARCHER",
    "ReflectiveAgent",
    "SelfCritique",
    "TaskResult",
    "ToolCallRecord",
    "WRITER",
    "State",
    "StateMachineAgent",
    "StreamingAgent",
    "Transition",
    "code_review_team",
    "create_agent",
    "debate_team",
    "fsm_to_mermaid",
    "inject_cot_instruction",
    "parse_thinking",
    "research_team",
]
