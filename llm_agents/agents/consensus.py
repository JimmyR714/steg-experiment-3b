"""Consensus and voting mechanisms for aggregating multi-agent answers."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llm_agents.agents.agent import Agent, AgentResponse
from llm_agents.logprobs.ops import perplexity

if TYPE_CHECKING:
    from llm_agents.models.types import LogProbResult


@dataclass
class ConsensusResult:
    """Result of a consensus vote.

    Attributes:
        answer: The winning answer.
        vote_distribution: Mapping of answer text to vote count or weight.
        confidence: Overall confidence score for the winning answer.
        dissenting_views: List of answers that did not win.
    """

    answer: str
    vote_distribution: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    dissenting_views: list[str] = field(default_factory=list)


def _confidence_from_logprobs(logprobs: LogProbResult | None) -> float:
    """Derive a confidence score from logprobs using inverse perplexity.

    Returns a value in (0, 1] where 1 is most confident.  Falls back to
    a neutral 0.5 when logprobs are unavailable.
    """
    if logprobs is None or not logprobs.tokens:
        return 0.5
    try:
        ppl = perplexity(logprobs)
    except ValueError:
        return 0.5
    # Map perplexity to (0, 1]: confidence = 1 / perplexity
    return min(1.0, 1.0 / max(ppl, 1e-9))


def _collect_responses(
    agents: list[Agent], prompt: str
) -> list[tuple[AgentResponse, float]]:
    """Run each agent independently and return (response, confidence) pairs."""
    results: list[tuple[AgentResponse, float]] = []
    for agent in agents:
        agent.reset()
        resp = agent.run(prompt)
        conf = _confidence_from_logprobs(resp.logprobs)
        results.append((resp, conf))
    return results


def _build_result(
    votes: dict[str, float],
    responses: list[tuple[AgentResponse, float]],
) -> ConsensusResult:
    """Build a ConsensusResult from a vote distribution."""
    if not votes:
        return ConsensusResult(answer="", confidence=0.0)

    winner = max(votes, key=votes.get)  # type: ignore[arg-type]
    total = sum(votes.values())
    confidence = votes[winner] / total if total > 0 else 0.0

    dissenting = [ans for ans in votes if ans != winner]

    return ConsensusResult(
        answer=winner,
        vote_distribution=dict(votes),
        confidence=confidence,
        dissenting_views=dissenting,
    )


def majority_vote(agents: list[Agent], prompt: str) -> ConsensusResult:
    """Each agent answers independently; majority wins.

    Ties are broken by average logprob confidence of the tied answers.

    Args:
        agents: The agents to poll.
        prompt: The question or task prompt.

    Returns:
        A :class:`ConsensusResult` with the winning answer.
    """
    responses = _collect_responses(agents, prompt)

    # Count votes and track confidence per answer
    answer_counts: dict[str, int] = Counter()
    answer_confidence: dict[str, list[float]] = {}

    for resp, conf in responses:
        text = resp.content.strip()
        answer_counts[text] += 1
        answer_confidence.setdefault(text, []).append(conf)

    # Break ties using average confidence
    max_count = max(answer_counts.values())
    tied = [ans for ans, cnt in answer_counts.items() if cnt == max_count]

    if len(tied) == 1:
        winner = tied[0]
    else:
        winner = max(
            tied,
            key=lambda ans: sum(answer_confidence[ans]) / len(answer_confidence[ans]),
        )

    votes = {ans: float(cnt) for ans, cnt in answer_counts.items()}
    total = sum(votes.values())
    confidence = votes[winner] / total if total > 0 else 0.0
    dissenting = [ans for ans in votes if ans != winner]

    return ConsensusResult(
        answer=winner,
        vote_distribution=votes,
        confidence=confidence,
        dissenting_views=dissenting,
    )


def weighted_vote(
    agents: list[Agent], weights: list[float], prompt: str
) -> ConsensusResult:
    """Weighted voting by agent reliability or expertise scores.

    Args:
        agents: The agents to poll.
        weights: A weight for each agent (must match length of agents).
        prompt: The question or task prompt.

    Returns:
        A :class:`ConsensusResult` with the winning answer.

    Raises:
        ValueError: If agents and weights have different lengths.
    """
    if len(agents) != len(weights):
        raise ValueError("agents and weights must have the same length")

    responses = _collect_responses(agents, prompt)

    votes: dict[str, float] = {}
    for (resp, conf), weight in zip(responses, weights):
        text = resp.content.strip()
        votes[text] = votes.get(text, 0.0) + weight

    return _build_result(votes, responses)


def ranked_choice(
    agents: list[Agent], prompt: str, rounds: int = 3
) -> ConsensusResult:
    """Iterative elimination voting over multiple rounds.

    In each round, every agent is asked to pick from the remaining
    candidates.  The candidate with the fewest votes is eliminated.
    Continues until one candidate remains or *rounds* is exhausted.

    Args:
        agents: The agents to poll.
        prompt: The question or task prompt.
        rounds: Maximum number of elimination rounds.

    Returns:
        A :class:`ConsensusResult`.
    """
    # First round: collect all unique answers
    responses = _collect_responses(agents, prompt)
    candidates = list({resp.content.strip() for resp, _ in responses})

    if len(candidates) <= 1:
        answer = candidates[0] if candidates else ""
        return ConsensusResult(
            answer=answer,
            vote_distribution={answer: float(len(agents))} if answer else {},
            confidence=1.0 if answer else 0.0,
        )

    vote_counts: dict[str, float] = Counter()
    for resp, _ in responses:
        vote_counts[resp.content.strip()] += 1

    for _round in range(rounds):
        if len(candidates) <= 1:
            break

        # Eliminate the candidate with the fewest votes
        min_votes = min(vote_counts.get(c, 0) for c in candidates)
        eliminated = [c for c in candidates if vote_counts.get(c, 0) == min_votes]
        candidates = [c for c in candidates if c not in eliminated]

        if not candidates:
            # All tied — restore and stop
            candidates = eliminated
            break

        if len(candidates) <= 1:
            break

        # Re-vote among remaining candidates
        candidate_list = "\n".join(f"- {c}" for c in candidates)
        revote_prompt = (
            f"Given the original question: {prompt}\n\n"
            f"Choose the best answer from these remaining candidates:\n"
            f"{candidate_list}\n\n"
            f"Respond with ONLY one of the candidates above, exactly as written."
        )

        vote_counts = Counter()
        for agent in agents:
            agent.reset()
            resp = agent.run(revote_prompt)
            text = resp.content.strip()
            # Match to closest candidate
            best_match = text
            if text not in candidates:
                for c in candidates:
                    if c.lower() in text.lower() or text.lower() in c.lower():
                        best_match = c
                        break
                else:
                    best_match = candidates[0]
            vote_counts[best_match] += 1

    winner = candidates[0] if candidates else ""
    votes = {c: float(vote_counts.get(c, 0)) for c in candidates}

    total = sum(votes.values()) if votes else 0
    confidence = votes.get(winner, 0) / total if total > 0 else 0.0
    dissenting = [c for c in candidates if c != winner]

    return ConsensusResult(
        answer=winner,
        vote_distribution=votes,
        confidence=confidence,
        dissenting_views=dissenting,
    )


def debate_consensus(
    agents: list[Agent], prompt: str, max_rounds: int = 3
) -> ConsensusResult:
    """Agents debate until they converge or max rounds are exhausted.

    Each round, agents see all previous arguments and are asked to update
    their answer.  Convergence is detected when all agents give the same
    answer.

    Args:
        agents: The agents to poll (at least 2).
        prompt: The question or task prompt.
        max_rounds: Maximum number of debate rounds.

    Returns:
        A :class:`ConsensusResult`.

    Raises:
        ValueError: If fewer than 2 agents are provided.
    """
    if len(agents) < 2:
        raise ValueError("debate_consensus requires at least 2 agents")

    # Initial answers
    current_answers: list[str] = []
    for agent in agents:
        agent.reset()
        resp = agent.run(prompt)
        current_answers.append(resp.content.strip())

    for _round in range(max_rounds):
        # Check for convergence
        if len(set(current_answers)) == 1:
            answer = current_answers[0]
            return ConsensusResult(
                answer=answer,
                vote_distribution={answer: float(len(agents))},
                confidence=1.0,
            )

        # Build summary of all current positions
        positions = "\n".join(
            f"Agent {i+1}: {ans}" for i, ans in enumerate(current_answers)
        )
        debate_prompt = (
            f"Original question: {prompt}\n\n"
            f"Current positions from all agents:\n{positions}\n\n"
            f"Consider the other agents' answers and provide your updated "
            f"answer. If you agree with another agent, adopt their answer "
            f"exactly. Respond with ONLY your answer."
        )

        new_answers: list[str] = []
        for agent in agents:
            resp = agent.run(debate_prompt)
            new_answers.append(resp.content.strip())

        current_answers = new_answers

    # Didn't converge — use majority vote on final answers
    vote_counts: dict[str, int] = Counter(current_answers)
    winner = max(vote_counts, key=vote_counts.get)  # type: ignore[arg-type]
    votes = {ans: float(cnt) for ans, cnt in vote_counts.items()}
    total = sum(votes.values())

    return ConsensusResult(
        answer=winner,
        vote_distribution=votes,
        confidence=votes[winner] / total if total > 0 else 0.0,
        dissenting_views=[ans for ans in votes if ans != winner],
    )
