"""Standard agent task library — pre-built recipes for common workflows."""

from __future__ import annotations

import math
import re
from typing import Any

from llm_agents.agents.agent import Agent
from llm_agents.models.base import BaseModel
from llm_agents.tasks.types import ClassifyResult, DebateResult


def summarize(model: BaseModel, text: str) -> str:
    """Summarize the given text using a single agent.

    Args:
        model: The LLM backend to use.
        text: The text to summarize.

    Returns:
        A concise summary of the input text.
    """
    agent = Agent(
        name="summarizer",
        model=model,
        system_prompt="You are a concise summarizer. Provide a clear, brief summary of the given text.",
        enable_cot=False,
    )
    response = agent.run(f"Summarize the following text:\n\n{text}")
    return response.content


def qa(model: BaseModel, question: str, context: str) -> str:
    """Answer a question given a context passage.

    Args:
        model: The LLM backend to use.
        question: The question to answer.
        context: The context passage containing relevant information.

    Returns:
        The answer to the question.
    """
    agent = Agent(
        name="qa_agent",
        model=model,
        system_prompt=(
            "You are a question-answering assistant. Answer the question "
            "based only on the provided context. Be concise and accurate."
        ),
        enable_cot=False,
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = agent.run(prompt)
    return response.content


def classify(model: BaseModel, text: str, labels: list[str]) -> ClassifyResult:
    """Classify text into one of the given labels using log-probabilities.

    The model is asked to pick a label, and log-probabilities from the
    generation are used to estimate a probability distribution over labels.

    Args:
        model: The LLM backend to use.
        text: The text to classify.
        labels: The candidate labels.

    Returns:
        A :class:`ClassifyResult` with the predicted label and probability
        estimates.
    """
    labels_str = ", ".join(labels)
    agent = Agent(
        name="classifier",
        model=model,
        system_prompt=(
            f"You are a text classifier. Classify the given text into exactly "
            f"one of these labels: {labels_str}. Respond with ONLY the label, "
            f"nothing else."
        ),
        enable_cot=False,
    )
    response = agent.run(f"Classify this text:\n\n{text}")
    predicted = response.content.strip()

    # Build probability dict from logprobs if available.
    probabilities: dict[str, float] = {}
    if response.logprobs and response.logprobs.top_k_per_position:
        # Look at the first position's top-k tokens for label matches.
        top_k = response.logprobs.top_k_per_position[0]
        for token_lp in top_k:
            token_text = token_lp.token.strip().lower()
            for label in labels:
                if label.lower().startswith(token_text) or token_text.startswith(label.lower()):
                    probabilities[label] = math.exp(token_lp.logprob)
    if not probabilities:
        # Fallback: assign 1.0 to the predicted label.
        for label in labels:
            probabilities[label] = 1.0 if label.lower() == predicted.lower() else 0.0

    # Normalize probabilities.
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {k: v / total for k, v in probabilities.items()}

    return ClassifyResult(label=predicted, probabilities=probabilities)


def debate(
    models: list[BaseModel],
    topic: str,
    rounds: int = 2,
) -> DebateResult:
    """Run a debate between two agents with a judge.

    Two agents argue opposing sides of the topic for the specified number
    of rounds, then a judge agent picks the winner.

    Args:
        models: A list of 2 or 3 models. ``models[0]`` is used for the
            pro side, ``models[1]`` for the con side, and ``models[2]``
            (or ``models[0]`` if only 2 are provided) for the judge.
        topic: The debate topic.
        rounds: Number of argument rounds.

    Returns:
        A :class:`DebateResult` with rounds, winner, and judgment.

    Raises:
        ValueError: If fewer than 2 models are provided.
    """
    if len(models) < 2:
        raise ValueError("debate() requires at least 2 models")

    pro_model = models[0]
    con_model = models[1]
    judge_model = models[2] if len(models) > 2 else models[0]

    pro_agent = Agent(
        name="pro",
        model=pro_model,
        system_prompt=(
            f"You are arguing IN FAVOR of the following topic: {topic}. "
            f"Make compelling, concise arguments. You may respond to your "
            f"opponent's previous arguments."
        ),
        enable_cot=False,
    )
    con_agent = Agent(
        name="con",
        model=con_model,
        system_prompt=(
            f"You are arguing AGAINST the following topic: {topic}. "
            f"Make compelling, concise arguments. You may respond to your "
            f"opponent's previous arguments."
        ),
        enable_cot=False,
    )

    debate_rounds: list[tuple[str, str]] = []

    for i in range(rounds):
        if i == 0:
            pro_response = pro_agent.run(
                f"Present your opening argument for: {topic}"
            )
        else:
            pro_response = pro_agent.run(
                f"The opposing side argued: {con_arg}\n\nPresent your rebuttal."
            )
        pro_arg = pro_response.content

        con_response = con_agent.run(
            f"The pro side argued: {pro_arg}\n\nPresent your counter-argument."
            if i == 0
            else f"The pro side argued: {pro_arg}\n\nPresent your rebuttal."
        )
        con_arg = con_response.content

        debate_rounds.append((pro_arg, con_arg))

    # Judge evaluates the debate.
    transcript = ""
    for idx, (pro_arg, con_arg) in enumerate(debate_rounds, 1):
        transcript += f"\n--- Round {idx} ---\n"
        transcript += f"PRO: {pro_arg}\n"
        transcript += f"CON: {con_arg}\n"

    judge_agent = Agent(
        name="judge",
        model=judge_model,
        system_prompt=(
            "You are an impartial debate judge. Read the debate transcript "
            "and decide the winner. Respond with your reasoning followed by "
            "your verdict on a new line starting with 'WINNER: pro' or "
            "'WINNER: con'."
        ),
        enable_cot=False,
    )
    judge_response = judge_agent.run(
        f"Topic: {topic}\n\nDebate transcript:{transcript}\n\n"
        f"Who won the debate?"
    )

    judgment = judge_response.content
    winner = ""
    # Extract winner from judgment.
    winner_match = re.search(r"WINNER:\s*(pro|con)", judgment, re.IGNORECASE)
    if winner_match:
        winner = winner_match.group(1).lower()

    return DebateResult(
        topic=topic,
        rounds=debate_rounds,
        winner=winner,
        judgment=judgment,
    )


def chain(agents: list[Agent], input_text: str) -> str:
    """Run a sequential pipeline of agents.

    The output of agent N becomes the input to agent N+1.

    Args:
        agents: Ordered list of agents to run.
        input_text: Initial input for the first agent.

    Returns:
        The output of the last agent in the chain.

    Raises:
        ValueError: If the agents list is empty.
    """
    if not agents:
        raise ValueError("chain() requires at least one agent")

    current = input_text
    for agent in agents:
        response = agent.run(current)
        current = response.content
    return current


def map_reduce(
    agent: Agent,
    items: list[str],
    reduce_agent: Agent,
) -> str:
    """Map an agent over a list of items, then reduce the results.

    Each item is processed independently by the *agent* (which is reset
    between items to avoid cross-contamination). The results are then
    combined and fed to *reduce_agent* for synthesis.

    Args:
        agent: The agent applied to each item.
        items: The list of input items to process.
        reduce_agent: The agent that synthesizes the mapped results.

    Returns:
        The reduced output.

    Raises:
        ValueError: If items is empty.
    """
    if not items:
        raise ValueError("map_reduce() requires at least one item")

    mapped_results: list[str] = []
    for item in items:
        agent.reset()
        response = agent.run(item)
        mapped_results.append(response.content)

    # Feed all mapped results to the reducer.
    combined = "\n\n".join(
        f"Item {i+1} result:\n{result}"
        for i, result in enumerate(mapped_results)
    )
    reduce_agent.reset()
    response = reduce_agent.run(
        f"Synthesize the following results:\n\n{combined}"
    )
    return response.content
