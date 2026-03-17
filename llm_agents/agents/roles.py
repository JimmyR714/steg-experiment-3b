"""Pre-built agent roles and persona presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_agents.agents.agent import Agent
from llm_agents.tools.base import Tool

if TYPE_CHECKING:
    from llm_agents.models.base import BaseModel


@dataclass
class AgentRole:
    """A reusable agent persona with default configuration.

    Attributes:
        name: Short name for this role (e.g. ``"Researcher"``).
        system_prompt: The system prompt that defines the persona.
        default_tools: Tool names that should be enabled for this role.
        enable_cot: Whether chain-of-thought reasoning is enabled.
        temperature: Recommended sampling temperature.
        description: Human-readable description of the role's purpose.
    """

    name: str
    system_prompt: str
    default_tools: list[str] = field(default_factory=list)
    enable_cot: bool = True
    temperature: float = 0.7
    description: str = ""


# ---------------------------------------------------------------------------
# Built-in roles
# ---------------------------------------------------------------------------

RESEARCHER = AgentRole(
    name="Researcher",
    system_prompt=(
        "You are a meticulous research assistant. Your job is to find, "
        "synthesize, and present information accurately. Always cite your "
        "sources and distinguish between established facts and speculation. "
        "When uncertain, say so explicitly."
    ),
    default_tools=["web_search", "store_memory", "recall"],
    enable_cot=True,
    temperature=0.3,
    description="Web search, note-taking, citation-aware research agent.",
)

CODER = AgentRole(
    name="Coder",
    system_prompt=(
        "You are an expert software engineer. Write clean, well-tested code. "
        "Follow best practices for the language in use. When debugging, reason "
        "step-by-step about possible causes before proposing fixes. Always "
        "consider edge cases and error handling."
    ),
    default_tools=["execute_python", "execute_shell"],
    enable_cot=True,
    temperature=0.2,
    description="Code execution, debugging, and software development agent.",
)

ANALYST = AgentRole(
    name="Analyst",
    system_prompt=(
        "You are a data analyst. Process data carefully, show your "
        "calculations, and present results in structured formats (tables, "
        "lists, JSON). Use quantitative reasoning and statistical thinking. "
        "Always verify your calculations."
    ),
    default_tools=["execute_python"],
    enable_cot=True,
    temperature=0.2,
    description="Calculator, data processing, and structured output agent.",
)

WRITER = AgentRole(
    name="Writer",
    system_prompt=(
        "You are a skilled writer and editor. Produce clear, engaging prose "
        "tailored to the audience and purpose. Pay attention to structure, "
        "tone, grammar, and style. When editing, explain your changes."
    ),
    default_tools=[],
    enable_cot=False,
    temperature=0.8,
    description="Long-form generation, editing, and style-aware writing agent.",
)

CRITIC = AgentRole(
    name="Critic",
    system_prompt=(
        "You are an evaluator and critic. Assess work against clear criteria "
        "using a structured rubric. Be fair, specific, and constructive. "
        "Provide a numerical score alongside qualitative feedback. "
        "Identify both strengths and areas for improvement."
    ),
    default_tools=[],
    enable_cot=True,
    temperature=0.3,
    description="Evaluation-focused, rubric-based scoring agent.",
)

COORDINATOR = AgentRole(
    name="Coordinator",
    system_prompt=(
        "You are a project coordinator. Break complex tasks into subtasks, "
        "delegate to appropriate team members, synthesize results, and "
        "ensure overall quality. Track progress and identify blockers. "
        "Communicate clearly and concisely."
    ),
    default_tools=[],
    enable_cot=True,
    temperature=0.5,
    description="Delegation, synthesis, and project management agent.",
)

FACT_CHECKER = AgentRole(
    name="FactChecker",
    system_prompt=(
        "You are a fact-checker. Verify claims by cross-referencing multiple "
        "sources. Classify each claim as Supported, Refuted, or Unverifiable. "
        "Provide evidence for your determination. Be skeptical of "
        "unsubstantiated claims."
    ),
    default_tools=["web_search", "recall"],
    enable_cot=True,
    temperature=0.2,
    description="Verification, source cross-referencing agent.",
)

BUILTIN_ROLES: dict[str, AgentRole] = {
    "researcher": RESEARCHER,
    "coder": CODER,
    "analyst": ANALYST,
    "writer": WRITER,
    "critic": CRITIC,
    "coordinator": COORDINATOR,
    "fact_checker": FACT_CHECKER,
}
"""Registry of all built-in roles, keyed by lowercase name."""


def create_agent(
    role: AgentRole | str,
    model: BaseModel,
    *,
    tools: list[Tool] | None = None,
    **overrides: Any,
) -> Agent:
    """Create an agent from a role preset.

    Args:
        role: An :class:`AgentRole` instance or the name of a built-in role
            (e.g. ``"researcher"``, ``"coder"``).
        model: The LLM backend for the agent.
        tools: Additional tools to provide beyond the role defaults.
            Role-default tools are referenced by name but not automatically
            instantiated—pass concrete :class:`Tool` instances here.
        **overrides: Override any ``Agent`` constructor parameter (e.g.
            ``system_prompt``, ``enable_cot``, ``max_tool_rounds``).

    Returns:
        A configured :class:`Agent`.

    Raises:
        KeyError: If *role* is a string that does not match a built-in role.
    """
    if isinstance(role, str):
        key = role.lower().replace(" ", "_")
        if key not in BUILTIN_ROLES:
            raise KeyError(
                f"Unknown built-in role: {role!r}. "
                f"Available: {', '.join(BUILTIN_ROLES)}"
            )
        role = BUILTIN_ROLES[key]

    agent_kwargs: dict[str, Any] = {
        "name": role.name,
        "model": model,
        "system_prompt": role.system_prompt,
        "tools": tools or [],
        "enable_cot": role.enable_cot,
    }
    agent_kwargs.update(overrides)

    return Agent(**agent_kwargs)
