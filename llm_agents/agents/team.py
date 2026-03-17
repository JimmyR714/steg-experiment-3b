"""Pre-configured multi-agent team setups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llm_agents.agents.agent import Agent
from llm_agents.agents.roles import (
    ANALYST,
    CODER,
    COORDINATOR,
    CRITIC,
    RESEARCHER,
    WRITER,
    AgentRole,
    create_agent,
)

if TYPE_CHECKING:
    from llm_agents.models.base import BaseModel
    from llm_agents.tools.base import Tool


@dataclass
class AgentTeam:
    """A named group of agents configured to work together.

    Attributes:
        name: Human-readable team name.
        agents: The agents comprising this team.
        description: What this team is designed to accomplish.
    """

    name: str
    agents: list[Agent] = field(default_factory=list)
    description: str = ""

    def run_all(self, prompt: str) -> list[tuple[str, str]]:
        """Run all agents on the same prompt and collect responses.

        Args:
            prompt: The task or question for all agents.

        Returns:
            A list of ``(agent_name, response_content)`` tuples.
        """
        results: list[tuple[str, str]] = []
        for agent in self.agents:
            agent.reset()
            resp = agent.run(prompt)
            results.append((agent.name, resp.content))
        return results

    def run_sequential(self, prompt: str) -> str:
        """Run agents sequentially, passing each output to the next.

        The first agent receives the original *prompt*.  Each subsequent
        agent receives the previous agent's response as input.

        Args:
            prompt: The initial task or question.

        Returns:
            The final agent's response content.
        """
        current_input = prompt
        for agent in self.agents:
            agent.reset()
            resp = agent.run(current_input)
            current_input = resp.content
        return current_input


def research_team(
    model: BaseModel,
    tools: list[Tool] | None = None,
) -> AgentTeam:
    """Create a research team: researcher + analyst + writer.

    Args:
        model: The LLM backend shared by all agents.
        tools: Optional extra tools for the researcher.

    Returns:
        An :class:`AgentTeam` ready to use.
    """
    return AgentTeam(
        name="Research Team",
        agents=[
            create_agent(RESEARCHER, model, tools=tools),
            create_agent(ANALYST, model),
            create_agent(WRITER, model),
        ],
        description="Researcher gathers info, analyst processes data, writer produces output.",
    )


def code_review_team(
    model: BaseModel,
    tools: list[Tool] | None = None,
) -> AgentTeam:
    """Create a code review team: coder + critic + coordinator.

    Args:
        model: The LLM backend shared by all agents.
        tools: Optional extra tools for the coder.

    Returns:
        An :class:`AgentTeam` ready to use.
    """
    return AgentTeam(
        name="Code Review Team",
        agents=[
            create_agent(CODER, model, tools=tools),
            create_agent(CRITIC, model),
            create_agent(COORDINATOR, model),
        ],
        description="Coder writes code, critic reviews it, coordinator manages the process.",
    )


def debate_team(
    topic: str,
    model: BaseModel,
) -> AgentTeam:
    """Create a debate team: two debaters with opposing views + a judge.

    Args:
        topic: The debate topic used to customize system prompts.
        model: The LLM backend shared by all agents.

    Returns:
        An :class:`AgentTeam` with two debaters and a judge.
    """
    debater_for = AgentRole(
        name="Debater-For",
        system_prompt=(
            f"You are debating in FAVOR of the following position: {topic}\n\n"
            "Present strong arguments supporting this position. Use evidence, "
            "logic, and persuasive reasoning. Acknowledge counterarguments "
            "but refute them."
        ),
        enable_cot=True,
        temperature=0.7,
        description="Argues in favor of the topic.",
    )

    debater_against = AgentRole(
        name="Debater-Against",
        system_prompt=(
            f"You are debating AGAINST the following position: {topic}\n\n"
            "Present strong arguments opposing this position. Use evidence, "
            "logic, and persuasive reasoning. Acknowledge counterarguments "
            "but refute them."
        ),
        enable_cot=True,
        temperature=0.7,
        description="Argues against the topic.",
    )

    judge = AgentRole(
        name="Judge",
        system_prompt=(
            "You are an impartial debate judge. Evaluate arguments from both "
            "sides based on logical soundness, evidence quality, and "
            "persuasiveness. Declare a winner and explain your reasoning."
        ),
        enable_cot=True,
        temperature=0.3,
        description="Evaluates and judges the debate.",
    )

    return AgentTeam(
        name="Debate Team",
        agents=[
            create_agent(debater_for, model),
            create_agent(debater_against, model),
            create_agent(judge, model),
        ],
        description=f"Two debaters argue for/against '{topic}', judged by an impartial judge.",
    )
