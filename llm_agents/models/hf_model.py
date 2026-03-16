"""HuggingFace Transformers model backend.

Loads a local AutoModelForCausalLM, runs a forward pass, and extracts
log-probabilities from the logits.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_agents.models.base import BaseModel
from llm_agents.models.types import (
    CompletionResult,
    LogProbResult,
    TokenLogProb,
)


class HFModel(BaseModel):
    """Model backend for local HuggingFace causal language models.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        device: Torch device string (e.g. "cpu", "cuda").
        torch_dtype: Dtype for model weights.  Defaults to float32.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        ).to(device)
        self._model.eval()

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> torch.Tensor:
        """Tokenize text and return input_ids on the correct device."""
        return self._tokenizer.encode(text, return_tensors="pt").to(self.device)

    def _logits_to_logprobs(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """Convert raw logits to log-probabilities via log-softmax."""
        return torch.log_softmax(logits, dim=-1)

    def _extract_logprob_result(
        self,
        prompt: str,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        top_k: int,
    ) -> LogProbResult:
        """Run a forward pass on the full sequence and extract log-probs
        for the generated (non-prompt) tokens."""
        with torch.no_grad():
            outputs = self._model(generated_ids)
        all_logprobs = self._logits_to_logprobs(outputs.logits[0])

        prompt_len = input_ids.shape[1]
        tokens: list[TokenLogProb] = []
        top_k_per_position: list[list[TokenLogProb]] = []

        for i in range(prompt_len - 1, generated_ids.shape[1] - 1):
            logprobs_at_pos = all_logprobs[i]
            next_token_id = generated_ids[0, i + 1].item()
            token_str = self._tokenizer.decode([next_token_id])
            token_lp = logprobs_at_pos[next_token_id].item()

            # Determine rank among all tokens at this position.
            sorted_indices = torch.argsort(logprobs_at_pos, descending=True)
            rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[0].item()

            tokens.append(TokenLogProb(token=token_str, logprob=token_lp, rank=rank))

            # Top-k alternatives.
            top_k_ids = sorted_indices[:top_k]
            position_top_k: list[TokenLogProb] = []
            for r, tid in enumerate(top_k_ids):
                tid_int = tid.item()
                position_top_k.append(
                    TokenLogProb(
                        token=self._tokenizer.decode([tid_int]),
                        logprob=logprobs_at_pos[tid_int].item(),
                        rank=r,
                    )
                )
            top_k_per_position.append(position_top_k)

        return LogProbResult(
            prompt=prompt,
            tokens=tokens,
            top_k_per_position=top_k_per_position,
        )

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
        input_ids = self._encode(prompt)
        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1e-7,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        # Decode only the newly generated tokens.
        new_ids = output_ids[0, input_ids.shape[1] :]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        # Apply stop sequences.
        finish_reason = "stop"
        if stop:
            for seq in stop:
                idx = text.find(seq)
                if idx != -1:
                    text = text[:idx]
                    break
            else:
                if len(new_ids) >= max_tokens:
                    finish_reason = "length"
        elif len(new_ids) >= max_tokens:
            finish_reason = "length"

        return CompletionResult(text=text, finish_reason=finish_reason)

    def get_logprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        top_k: int = 5,
    ) -> LogProbResult:
        input_ids = self._encode(prompt)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        return self._extract_logprob_result(prompt, input_ids, output_ids, top_k)
