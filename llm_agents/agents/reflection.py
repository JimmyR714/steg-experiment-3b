"""Agent self-reflection and critique for iterative output improvement."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_agents.logprobs.ops import entropy, perplexity

if TYPE_CHECKING:
    from llm_agents.agents.agent import AgentResponse
    from llm_agents.models.base import BaseModel
    from llm_agents.models.types import LogProbResult

from llm_agents.agents.agent import Agent


@dataclass
class CritiqueResult:
    """Result of a critique evaluation.

    Attributes:
        accept: Whether the output is acceptable.
        feedback: Textual feedback explaining what should be improved.
        score: A quality score from 0.0 to 1.0.
    """

    accept: bool
    feedback: str
    score: float


class SelfCritique:
    """Critique strategy where the agent critiques its own output.

    Uses a second LLM pass with a critique-focused system prompt to evaluate
    the agent's output.

    Args:
        model: The LLM model to use for critique.
        criteria: Description of the quality criteria to evaluate against.
    """

    def __init__(self, model: BaseModel, criteria: str = "") -> None:
        self._model = model
        self._criteria = criteria or "Is the response accurate, complete, and well-structured?"

    def critique(self, original_prompt: str, response_text: str) -> CritiqueResult:
        """Evaluate *response_text* as a response to *original_prompt*.

        Returns a :class:`CritiqueResult` with accept/reject and feedback.
        """
        critique_prompt = (
            "System: You are a critical reviewer. Evaluate the following response "
            "to the given prompt. Return ONLY a JSON object with keys: "
            '"accept" (bool), "feedback" (str), "score" (float 0-1).\n\n'
            f"Criteria: {self._criteria}\n\n"
            f"Original prompt: {original_prompt}\n\n"
            f"Response to evaluate: {response_text}\n\n"
            "Assistant:"
        )
        completion = self._model.generate(critique_prompt, max_tokens=512)
        return _parse_critique(completion.text)


class PeerCritique:
    """Critique strategy using a separate agent as the critic.

    Args:
        critic_agent: An :class:`Agent` configured as a critic.
    """

    def __init__(self, critic_agent: Agent) -> None:
        self._critic = critic_agent

    def critique(self, original_prompt: str, response_text: str) -> CritiqueResult:
        """Have the critic agent evaluate the response.

        Returns a :class:`CritiqueResult`.
        """
        message = (
            "Evaluate the following response to the given prompt. "
            "Return ONLY a JSON object with keys: "
            '"accept" (bool), "feedback" (str), "score" (float 0-1).\n\n'
            f"Original prompt: {original_prompt}\n\n"
            f"Response to evaluate: {response_text}"
        )
        result = self._critic.run(message)
        return _parse_critique(result.content)


def _parse_critique(text: str) -> CritiqueResult:
    """Parse a critique JSON response, with fallback defaults."""
    try:
        # Try to extract JSON from the text
        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return CritiqueResult(
                accept=bool(data.get("accept", False)),
                feedback=str(data.get("feedback", "")),
                score=float(data.get("score", 0.0)),
            )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: treat as rejected with the raw text as feedback
    return CritiqueResult(accept=False, feedback=text, score=0.0)


def _should_reflect(logprobs: LogProbResult | None, entropy_threshold: float = 2.0) -> bool:
    """Determine if reflection is needed based on logprob confidence.

    High average entropy across token positions indicates the model was
    uncertain, suggesting reflection would be beneficial.

    Args:
        logprobs: The log-probability result from the generation.
        entropy_threshold: Average entropy above which reflection is triggered.

    Returns:
        *True* if reflection is recommended.
    """
    if logprobs is None:
        return False
    if not logprobs.top_k_per_position:
        return False

    entropies = [entropy(pos_logprobs) for pos_logprobs in logprobs.top_k_per_position]
    if not entropies:
        return False

    avg_entropy = sum(entropies) / len(entropies)
    return avg_entropy > entropy_threshold


class ReflectiveAgent:
    """Wraps an agent with a reflection loop for iterative improvement.

    After each response, a critic evaluates the output. If rejected, feedback
    is provided and the agent retries. Optionally uses logprob-based entropy
    as an additional trigger for reflection.

    Args:
        agent: The underlying agent to wrap.
        critic: A critique strategy (``SelfCritique`` or ``PeerCritique``).
        max_rounds: Maximum number of reflection rounds.
        entropy_threshold: If the agent's response has average entropy above
            this threshold, reflection is automatically triggered even if the
            critic accepts.
    """

    def __init__(
        self,
        agent: Agent,
        critic: SelfCritique | PeerCritique,
        max_rounds: int = 2,
        entropy_threshold: float = 2.0,
    ) -> None:
        self.agent = agent
        self.critic = critic
        self.max_rounds = max_rounds
        self.entropy_threshold = entropy_threshold

    def run(self, user_message: str) -> AgentResponse:
        """Run the agent with reflection loop.

        Steps:
            1. Run the agent on the user message.
            2. Check if reflection is needed (via critic or entropy).
            3. If needed, provide feedback and re-run.
            4. Return the final response.
        """
        response = self.agent.run(user_message)

        for _ in range(self.max_rounds):
            # Check if entropy-based reflection is needed
            needs_entropy_reflection = _should_reflect(
                response.logprobs, self.entropy_threshold
            )

            # Get critique
            critique = self.critic.critique(user_message, response.content)

            if critique.accept and not needs_entropy_reflection:
                return response

            # Build feedback message
            feedback = f"Please improve your response.\nFeedback: {critique.feedback}"
            if needs_entropy_reflection:
                feedback += (
                    "\nNote: Your response showed high uncertainty. "
                    "Please be more precise and confident."
                )

            response = self.agent.run(feedback)

        return response
