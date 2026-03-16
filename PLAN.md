# Multi-Agent LLM System with Log-Probabilities — Implementation Plan

This document breaks the system into small, self-contained phases. Each phase
produces working, testable code. The system is implemented in Python.

---

## Phase 1: Project Scaffolding & Model Interface

**Goal:** Set up the project structure and define the abstract interface for
loading LLM models and obtaining log-probabilities from completions.

**Deliverables:**
- `pyproject.toml` with dependencies (numpy, openai, transformers, torch, httpx)
- Package directory: `llm_agents/`
- `llm_agents/models/base.py` — abstract `BaseModel` class with methods:
  - `generate(prompt, max_tokens, temperature, top_k, …) -> CompletionResult`
  - `get_logprobs(prompt, max_tokens, top_k) -> LogProbResult`
- `llm_agents/models/types.py` — dataclasses:
  - `TokenLogProb(token, logprob, rank)`
  - `LogProbResult(prompt, tokens: list[TokenLogProb], top_k_per_position: list[list[TokenLogProb]])`
  - `CompletionResult(text, logprob_result, finish_reason)`
- Unit tests in `tests/test_models_types.py` validating dataclass construction.

**Lines of code estimate:** ~200

---

## Phase 2: Concrete Model Backends

**Goal:** Implement two concrete model backends so the system can actually call
models and retrieve log-probabilities.

**Deliverables:**
- `llm_agents/models/openai_model.py` — wraps the OpenAI-compatible API
  (works with OpenAI, vLLM, llama.cpp server, etc.). Extracts `top_logprobs`
  from the response and populates `LogProbResult`.
- `llm_agents/models/hf_model.py` — wraps a local HuggingFace
  `AutoModelForCausalLM`. Runs forward pass, extracts logits, converts to
  log-probs, returns `LogProbResult`.
- `llm_agents/models/registry.py` — simple model registry:
  - `register_model(name, model_instance)`
  - `get_model(name) -> BaseModel`
  - `list_models() -> list[str]`
- Tests with mocked API responses in `tests/test_openai_model.py`.

**Lines of code estimate:** ~300

---

## Phase 3: Log-Probability Operations

**Goal:** Provide utility functions for common operations on log-probabilities.

**Deliverables:**
- `llm_agents/logprobs/ops.py` — functions operating on `LogProbResult`:
  - `entropy(logprobs_at_position)` — Shannon entropy from log-probs
  - `perplexity(logprob_result)` — sequence-level perplexity
  - `top_k_tokens(logprobs_at_position, k)` — return top-k tokens
  - `normalize(logprobs)` — re-normalize a set of log-probs (log-softmax)
  - `kl_divergence(p_logprobs, q_logprobs)` — KL divergence between two distributions
  - `surprise(logprob)` — pointwise surprise (negative log-prob)
  - `marginal_prob(logprobs, token_set)` — sum prob mass over a set of tokens
- Tests in `tests/test_logprob_ops.py` with hand-computed expected values.

**Lines of code estimate:** ~200

---

## Phase 4: Probability Tree Exploration

**Goal:** Given a prompt, build a tree of the most probable continuations by
repeatedly querying the model.

**Deliverables:**
- `llm_agents/logprobs/tree.py`:
  - `TreeNode(token, logprob, children: list[TreeNode], cumulative_logprob)`
  - `build_prob_tree(model, prompt, branch_factor=3, depth=3) -> TreeNode`
    — at each node, take the top `branch_factor` tokens, extend the prompt
    with each, and recurse up to `depth`.
  - `print_tree(node, indent=0)` — pretty-print the tree to stdout.
  - `tree_to_dict(node)` — serialize tree to a nested dict (JSON-friendly).
  - `best_path(node)` — return the highest cumulative-probability path.
  - `all_paths(node)` — yield all root-to-leaf paths with cumulative logprobs.
- Tests in `tests/test_tree.py` using a mock model.

**Lines of code estimate:** ~250

---

## Phase 5: Tool System

**Goal:** Give agents the ability to call simple tools (functions).

**Deliverables:**
- `llm_agents/tools/base.py`:
  - `@tool` decorator that registers a callable with a name, description, and
    JSON-schema for parameters.
  - `Tool(name, description, parameters_schema, fn)` dataclass.
- `llm_agents/tools/registry.py`:
  - `ToolRegistry` — stores tools, formats them into the system prompt or
    tool-call JSON that models expect, parses tool-call responses.
- `llm_agents/tools/builtin.py` — a few built-in tools:
  - `calculator(expression: str) -> str`
  - `web_search(query: str) -> str` (stub / mock)
  - `read_file(path: str) -> str`
  - `write_file(path: str, content: str) -> str`
- `llm_agents/tools/executor.py`:
  - `execute_tool_call(tool_registry, tool_call_json) -> str` — validates
    args against schema and calls the function.
- Tests in `tests/test_tools.py`.

**Lines of code estimate:** ~300

---

## Phase 6: Single Agent Core

**Goal:** Implement a single autonomous agent that can hold a conversation,
use tools, and optionally include chain-of-thought.

**Deliverables:**
- `llm_agents/agents/agent.py`:
  - `Agent(name, model, system_prompt, tools, enable_cot=True)`
  - `agent.run(user_message) -> AgentResponse` — runs the agent loop:
    1. Build messages (system + history + user).
    2. If `enable_cot`, wrap generation to allow `<think>...</think>` tags
       in the output; strip them from the final user-visible response but
       keep them internally.
    3. Check if the model output contains a tool call; if so, execute it
       and feed the result back into the conversation, then re-generate.
    4. Return final response + metadata (tool calls made, thinking, logprobs).
  - `AgentResponse(content, thinking, tool_calls, logprobs)`
- `llm_agents/agents/cot.py`:
  - `inject_cot_instruction(system_prompt) -> str` — appends CoT instructions.
  - `parse_thinking(text) -> tuple[str, str]` — separates `<think>` blocks.
  - Toggle support: agent constructor flag `enable_cot`.
- Tests in `tests/test_agent.py` with mocked model.

**Lines of code estimate:** ~300

---

## Phase 7: Multi-Agent Communication

**Goal:** Allow multiple agents to communicate with each other to complete
tasks collaboratively.

**Deliverables:**
- `llm_agents/agents/message_bus.py`:
  - `Message(sender, recipient, content, metadata)` dataclass.
  - `MessageBus` — in-memory pub/sub. Agents subscribe by name or to
    broadcast. Supports `send(msg)` and `receive(agent_name) -> list[Message]`.
- `llm_agents/agents/multi_agent.py`:
  - `MultiAgentSystem(agents: list[Agent], message_bus: MessageBus)`
  - `run_task(task: str, coordinator: str) -> TaskResult` — the coordinator
    agent receives the task, can delegate sub-tasks to other agents via
    tool calls (`send_message`, `wait_for_reply`), and synthesizes a final
    answer.
  - Inter-agent communication exposed as tools automatically injected into
    each agent:
    - `send_message(to: str, content: str)`
    - `broadcast(content: str)`
    - `wait_for_reply(from_agent: str, timeout: int)`
    - `list_agents() -> list[str]`
- `llm_agents/agents/task.py`:
  - `TaskResult(result, agent_trace: list[Message], logprobs)`
- Tests in `tests/test_multi_agent.py` with mocked models.

**Lines of code estimate:** ~350

---

## Phase 8: Standard Agent Task Library

**Goal:** Provide pre-built "recipes" for common multi-agent patterns.

**Deliverables:**
- `llm_agents/tasks/standard.py` — functions that set up and run common
  agent workflows:
  - `summarize(model, text) -> str` — single-agent summarization.
  - `qa(model, question, context) -> str` — question-answering.
  - `classify(model, text, labels) -> dict` — classification with probs.
  - `debate(models, topic, rounds) -> DebateResult` — two agents argue
    opposing sides, a judge agent picks the winner.
  - `chain(agents, input) -> str` — sequential pipeline: output of agent N
    is input to agent N+1.
  - `map_reduce(agent, items, reduce_agent) -> str` — map an agent over a
    list, then reduce.
- `llm_agents/tasks/types.py` — result dataclasses for the above.
- Tests in `tests/test_standard_tasks.py`.

**Lines of code estimate:** ~300

---

## Phase 9: CLI & Integration Demo

**Goal:** Provide a runnable CLI entry point and a demo script that ties
everything together.

**Deliverables:**
- `llm_agents/cli.py` — simple CLI (argparse or click):
  - `python -m llm_agents run --model <name> --prompt "..."` — single completion.
  - `python -m llm_agents tree --model <name> --prompt "..." --depth 3 --branch 3` — prob tree.
  - `python -m llm_agents agent --model <name> --tools --cot` — interactive agent REPL.
  - `python -m llm_agents multi --config <yaml>` — run a multi-agent task from config.
- `llm_agents/__main__.py` — entry point.
- `examples/demo.py` — end-to-end script demonstrating:
  - Loading two models.
  - Building a probability tree.
  - Running log-prob operations.
  - Single agent with tools and CoT.
  - Multi-agent debate task.
- `README.md` with usage instructions.

**Lines of code estimate:** ~300

---

## Dependency Graph

```
Phase 1 (scaffolding + types)
  └─> Phase 2 (model backends)
        ├─> Phase 3 (logprob ops)
        │     └─> Phase 4 (prob tree)
        └─> Phase 5 (tools)
              └─> Phase 6 (single agent + CoT)
                    └─> Phase 7 (multi-agent)
                          └─> Phase 8 (task library)
                                └─> Phase 9 (CLI + demo)
```

Phases 3/4 and 5/6 can be developed in parallel since they are independent
until Phase 7 merges them.

---

## Summary

| Phase | Description                    | Est. LoC |
|-------|--------------------------------|----------|
| 1     | Scaffolding & model interface  | ~200     |
| 2     | Concrete model backends        | ~300     |
| 3     | Log-probability operations     | ~200     |
| 4     | Probability tree exploration   | ~250     |
| 5     | Tool system                    | ~300     |
| 6     | Single agent core + CoT toggle | ~300     |
| 7     | Multi-agent communication      | ~350     |
| 8     | Standard task library          | ~300     |
| 9     | CLI & integration demo         | ~300     |
| **Total** |                            | **~2,500** |
