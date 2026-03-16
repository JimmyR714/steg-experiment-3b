"""OpenAI-compatible API model backend.

Works with OpenAI, vLLM, llama.cpp server, and any endpoint that
implements the OpenAI chat/completions or completions API.
"""

from __future__ import annotations

from openai import OpenAI

from llm_agents.models.base import BaseModel
from llm_agents.models.types import (
    CompletionResult,
    LogProbResult,
    TokenLogProb,
)


class OpenAIModel(BaseModel):
    """Model backend that calls an OpenAI-compatible API.

    Args:
        model: Model identifier (e.g. "gpt-4", "gpt-3.5-turbo").
        api_key: API key.  Defaults to the OPENAI_API_KEY env var.
        base_url: Optional base URL for alternative endpoints (vLLM, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_logprobs(choice) -> LogProbResult | None:
        """Extract log-probability data from an API response choice."""
        logprobs_data = getattr(choice, "logprobs", None)
        if logprobs_data is None:
            return None

        content = getattr(logprobs_data, "content", None)
        if content is None:
            return None

        tokens: list[TokenLogProb] = []
        top_k_per_position: list[list[TokenLogProb]] = []

        for idx, token_info in enumerate(content):
            tokens.append(
                TokenLogProb(
                    token=token_info.token,
                    logprob=token_info.logprob,
                    rank=0,  # chosen token
                )
            )

            position_top_k: list[TokenLogProb] = []
            top_logprobs = getattr(token_info, "top_logprobs", None) or []
            for rank, alt in enumerate(top_logprobs):
                position_top_k.append(
                    TokenLogProb(
                        token=alt.token,
                        logprob=alt.logprob,
                        rank=rank,
                    )
                )
            top_k_per_position.append(position_top_k)

        return LogProbResult(
            prompt="",  # filled in by caller
            tokens=tokens,
            top_k_per_position=top_k_per_position,
        )

    def _call_api(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        stop: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ):
        """Make a chat completion API call and return the raw response."""
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            kwargs["stop"] = stop
        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs is not None:
                kwargs["top_logprobs"] = top_logprobs

        return self._client.chat.completions.create(**kwargs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        response = self._call_api(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        choice = response.choices[0]
        return CompletionResult(
            text=choice.message.content or "",
            logprob_result=None,
            finish_reason=choice.finish_reason or "stop",
        )

    def get_logprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        top_k: int = 5,
    ) -> LogProbResult:
        response = self._call_api(
            prompt,
            max_tokens=max_tokens,
            temperature=1.0,
            logprobs=True,
            top_logprobs=top_k,
        )
        choice = response.choices[0]
        result = self._parse_logprobs(choice)
        if result is None:
            return LogProbResult(prompt=prompt)

        # Frozen dataclass — rebuild with the actual prompt.
        return LogProbResult(
            prompt=prompt,
            tokens=result.tokens,
            top_k_per_position=result.top_k_per_position,
        )
