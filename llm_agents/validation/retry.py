"""Retry logic for validated agent output."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from llm_agents.validation.schema import OutputSchema, ValidationResult

if TYPE_CHECKING:
    from llm_agents.agents.agent import Agent, AgentResponse


def with_retry(
    agent: Agent,
    user_message: str,
    schema: OutputSchema,
    max_attempts: int = 3,
    backoff: str = "immediate",
) -> AgentResponse:
    """Run *agent* with retries until output validates against *schema*.

    On validation failure, the error is fed back to the agent as a follow-up
    message so it can correct its output.

    Args:
        agent: The agent to run.
        user_message: The initial user message.
        schema: The output schema to validate against.
        max_attempts: Maximum number of attempts.
        backoff: Backoff strategy: ``"immediate"``, ``"linear"``, or
            ``"exponential"``.

    Returns:
        The :class:`AgentResponse` from the successful (or last) attempt.
    """
    last_response: AgentResponse | None = None
    last_result: ValidationResult | None = None

    for attempt in range(max_attempts):
        if attempt == 0:
            response = agent.run(user_message)
        else:
            # Build feedback message with validation errors
            error_msg = (
                "Your previous output did not match the required format.\n"
                f"Errors: {', '.join(last_result.errors)}\n"  # type: ignore[union-attr]
                f"Expected format: {schema.description or 'See schema.'}\n"
                "Please try again, outputting ONLY valid JSON matching the schema."
            )
            response = agent.run(error_msg)

        last_response = response
        result = schema.validate(response.content)
        last_result = result

        if result.valid:
            return response

        # Apply backoff before next attempt
        if attempt < max_attempts - 1:
            _backoff_sleep(attempt, backoff)

    # Return last response even if validation failed
    return last_response  # type: ignore[return-value]


def _backoff_sleep(attempt: int, strategy: str) -> None:
    """Sleep according to backoff strategy."""
    if strategy == "immediate":
        return
    elif strategy == "linear":
        time.sleep(attempt + 1)
    elif strategy == "exponential":
        time.sleep(2**attempt)
