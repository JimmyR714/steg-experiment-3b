# Steganography Between Agents: Security Assessment

## Short Answer

**No — the complete steganographic scheme is not wired to run between agents in the current codebase.** The encoding and decoding functions exist and are complete, but they are fully isolated from the multi-agent communication infrastructure with no integration bridge.

---

## What the Scheme Requires End-to-End

Zamir's scheme ("Undetectable Steganography for Language Models") needs all four of these elements operating together:

1. **A sender** that generates text using keyed interval-narrowing over the model's token distribution to embed a hidden message
2. **A channel** through which the stego-text travels between sender and receiver
3. **A receiver** that retraces the token selection path using the same key to recover the message
4. **A shared secret key** accessible to both sides at generation and analysis time

---

## What Exists in the Repo

`main.py` contains a complete, functional, standalone implementation:

| Component | Location | Status |
|---|---|---|
| Encoder (`calibrated_generate`) | `main.py:149` | Complete |
| Decoder (`response_analysis`) | `main.py:227` | Complete |
| Key derivation (SHA256 passphrase → key) | `main.py:65` | Complete |
| HMAC priority ordering (keyed token shuffle) | `main.py:50` | Complete |
| Interval arithmetic (bit → probability mapping) | `main.py:113` | Complete |
| Payload pack/unpack | `main.py:83` | Complete |
| CLI (`encode`/`decode` subcommands) | `main.py:333` | Complete |

The multi-agent framework also has:

- Inter-agent message passing via `MessageBus` (`llm_agents/agents/message_bus.py`)
- `MultiAgentSystem` that orchestrates agents via that bus (`llm_agents/agents/multi_agent.py`)
- A full logprobs stack (`llm_agents/logprobs/ops.py`, `llm_agents/logprobs/sampling.py`) that could theoretically support the scheme

---

## The Missing Integration

The critical question is whether these two halves are wired together. They are not. The gap is decisive.

**`main.py` is never imported by any agent code.** No agent file, tool, or workflow references `calibrated_generate`, `response_analysis`, or `main` at all.

The `Agent.run()` loop at `agent.py:238` calls:

```python
completion = self.model.generate(prompt, max_tokens=1024)
```

This is `BaseModel.generate()` — standard generation with no steganographic encoding. The output is plain text appended to history and sent over the bus as a bare string.

The `MessageBus.send()` at `message_bus.py:62` accepts a `Message(content: str)` — it transmits whatever string the agent produced verbatim, with no encode/decode pass.

For the scheme to run between agents, there would need to be code that:

1. Substitutes `calibrated_generate()` for `model.generate()` in the agent loop, passing a shared passphrase and a payload to embed in each agent's responses
2. Applies `response_analysis()` to messages *received* from the bus before the agent acts on them, using the same passphrase to extract the hidden payload

Neither exists. The two halves of the system — the steganographic functions and the multi-agent infrastructure — have no call path connecting them. `main.py` is a standalone CLI tool that loads its own HuggingFace model directly and is architecturally separate from the `llm_agents` framework.

---

## Summary

The repository contains all the individual components required for this steganographic scheme, and `main.py` is a production-ready standalone implementation of it. However, **there is no code through which the entire scheme could currently be running between agents** — the encoder/decoder is not integrated into agent message generation or reception at any point in the call graph.
