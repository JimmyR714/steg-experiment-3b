# Repository Structure Guide

This guide explains how the **llm-agents** repository is organized so you can find your way around, add new features, and maintain the codebase with confidence.

## Quick Overview

**llm-agents** is a Python framework for building multi-agent LLM systems with log-probability support. It provides model backends, tool use, multi-agent orchestration, safety guardrails, caching, tracing, and more.

- **Language:** Python 3.10+
- **Build system:** setuptools via `pyproject.toml`
- **Test framework:** pytest / pytest-asyncio
- **Dependencies:** numpy, openai, transformers, torch, httpx

```
steg-experiment-3b/
├── pyproject.toml          # Project metadata, dependencies, build config
├── llm_agents/             # All source code lives here
│   ├── models/             # LLM backends (OpenAI, HuggingFace)
│   ├── agents/             # Single-agent and multi-agent orchestration
│   ├── tools/              # Tool system (decorator, registry, execution)
│   ├── logprobs/           # Log-probability utilities and uncertainty
│   ├── memory/             # RAG and persistent memory stores
│   ├── validation/         # Output schema enforcement and retry
│   ├── tracing/            # Execution tracing, export, cost tracking
│   ├── safety/             # Input/output filters and guardrails
│   ├── routing/            # Adaptive model selection and dispatch
│   ├── ratelimit/          # Token bucket rate limiting and budgets
│   ├── cache/              # Exact and semantic caching layers
│   ├── workflows/          # Declarative YAML/JSON pipeline DSL
│   ├── tasks/              # Pre-built task recipes (summarize, QA, etc.)
│   ├── prompts/            # Prompt templates and composition
│   ├── eval/               # Evaluation datasets, metrics, and runners
│   └── plugins/            # Dynamic tool loading and plugin manifests
└── tests/                  # One test file per module (22 files)
```

---

## Package-by-Package Breakdown

### `llm_agents/models/` — LLM Backends

The foundation layer. Every model implements a common interface so the rest of the system doesn't care which provider you use.

| File | What it does |
|------|-------------|
| `base.py` | Abstract `BaseModel` class with `generate()` and `get_logprobs()` |
| `types.py` | Shared data types: `TokenLogProb`, `LogProbResult`, `CompletionResult` |
| `openai_model.py` | OpenAI-compatible API wrapper (also works with vLLM, llama.cpp) |
| `hf_model.py` | Local HuggingFace transformers wrapper |
| `registry.py` | Named model registry — register and retrieve models by string key |
| `streaming.py` | Streaming result types and async iterators |

**Tests:** `test_models_types.py`, `test_openai_model.py`, `test_streaming.py`

---

### `llm_agents/agents/` — Agent Orchestration

The largest module. Contains everything related to running agents — from a single agent with tools to multi-agent teams.

| File | What it does |
|------|-------------|
| `agent.py` | Core `Agent` class — runs an LLM with tools and chain-of-thought |
| `cot.py` | Chain-of-thought parsing and prompt injection |
| `multi_agent.py` | `MultiAgentSystem` — coordinates multiple agents on a task |
| `message_bus.py` | Inter-agent message passing |
| `task.py` | Task result data types |
| `roles.py` | Pre-built agent personas (Researcher, Coder, Analyst, Writer, Critic) |
| `team.py` | Pre-configured agent teams |
| `reflection.py` | Self-critique and iterative refinement loops |
| `consensus.py` | Voting and consensus mechanisms (majority, weighted, debate) |
| `fsm.py` | Finite state machine agents with state transitions |
| `streaming_agent.py` | Token-by-token streaming agent |

**Tests:** `test_agent.py`, `test_multi_agent.py`, `test_reflection.py`, `test_consensus.py`, `test_fsm.py`

---

### `llm_agents/tools/` — Tool System

Lets agents call functions. Tools are defined with a `@tool` decorator that auto-generates schemas from type hints.

| File | What it does |
|------|-------------|
| `base.py` | `Tool` dataclass and `@tool` decorator |
| `registry.py` | `ToolRegistry` — stores tools and formats them for LLM prompts |
| `executor.py` | Parses tool calls from LLM output and executes them |
| `builtin.py` | Built-in tools: calculator, web_search, read_file, write_file |
| `sandbox.py` | Sandboxed code execution environment |
| `sandbox_manager.py` | Sandbox lifecycle and resource management |

**Tests:** `test_tools.py`, `test_sandbox.py`

---

### `llm_agents/logprobs/` — Log-Probability Operations

Utilities for working with token-level log-probabilities — useful for uncertainty estimation, calibration, and tree search.

| File | What it does |
|------|-------------|
| `ops.py` | Core functions: entropy, perplexity, KL divergence, top-k extraction |
| `tree.py` | `TreeNode` and `build_prob_tree()` for exploring token continuations |
| `sampling.py` | Diverse sampling, self-consistency, conformal prediction |
| `uncertainty.py` | Confidence scoring and hallucination detection |

**Tests:** `test_logprob_ops.py`, `test_tree.py`, `test_uncertainty.py`

---

### `llm_agents/memory/` — RAG and Persistent Memory

Gives agents long-term memory via vector similarity search over stored documents.

| File | What it does |
|------|-------------|
| `store.py` | `MemoryStore` (abstract), `InMemoryStore`, `PersistentStore` (SQLite) |
| `embeddings.py` | `Embedder` protocol with `HFEmbedder` and `OpenAIEmbedder` |
| `chunker.py` | Text chunking (sliding window, separator-based) |

**Tests:** `test_memory.py`

---

### `llm_agents/validation/` — Output Validation

Constrains agent output to a specific format (JSON, CSV, YAML, etc.) and retries on failure.

| File | What it does |
|------|-------------|
| `schema.py` | `OutputSchema`, `ValidationResult`, `extract_json()` |
| `retry.py` | Retry wrapper that re-prompts the model on validation failure |
| `formats.py` | Pre-built schemas for JSON, CSV, YAML, markdown tables |

**Tests:** `test_validation.py`

---

### `llm_agents/tracing/` — Observability

Records execution traces for debugging, performance analysis, and cost tracking.

| File | What it does |
|------|-------------|
| `tracer.py` | `Tracer` context manager, `TraceEvent`, `Span` dataclasses |
| `export.py` | Export to JSON, Chrome Tracing format, OpenTelemetry |
| `cost.py` | `TokenCounter`, `CostEstimator`, `BudgetGuard` |

**Tests:** `test_tracing.py`

---

### `llm_agents/safety/` — Safety and Guardrails

Input/output filtering to prevent prompt injection, policy violations, and unsafe content.

| File | What it does |
|------|-------------|
| `input_filter.py` | Prompt injection detection with configurable threat levels |
| `output_filter.py` | Output policy violation detection and classification |
| `guardrails.py` | Composable `Guardrail` chains and `GuardedAgent` wrapper |

**Tests:** `test_safety.py`

---

### `llm_agents/routing/` — Model Routing

Automatically picks the right model for a query based on complexity, latency, or budget.

| File | What it does |
|------|-------------|
| `classifier.py` | `ComplexityClassifier` — analyzes prompt difficulty |
| `router.py` | `ModelRouter`, `CascadeRouter` (escalate on failure), `LatencyRouter` |
| `budget.py` | Token budget-aware routing |

**Tests:** `test_routing.py`

---

### `llm_agents/ratelimit/` — Rate Limiting

Prevents API rate limit violations and controls token spend.

| File | What it does |
|------|-------------|
| `limiter.py` | Token bucket algorithm with adaptive rate limiting |
| `budget.py` | `TokenBudget` tracking and allocation |
| `middleware.py` | `RateLimitedModel` wrapper with automatic retry and backoff |

**Tests:** `test_ratelimit.py`

---

### `llm_agents/cache/` — Caching

Saves tokens and latency by caching model responses.

| File | What it does |
|------|-------------|
| `exact.py` | Hash-based exact match cache (in-memory, SQLite, Redis) |
| `semantic.py` | Embedding-based semantic similarity cache |
| `middleware.py` | `CachedModel` wrapper and cache statistics |

**Tests:** (covered in integration tests)

---

### `llm_agents/workflows/` — Declarative Pipelines

Define multi-step, multi-agent pipelines in YAML or JSON without writing Python.

| File | What it does |
|------|-------------|
| `schema.py` | `Workflow`, `AgentDefinition`, `Step` dataclasses with variable interpolation |
| `engine.py` | `WorkflowEngine` — executes workflows step by step |
| `loader.py` | Loads and validates workflows from YAML/JSON files |

**Tests:** `test_workflows.py`

---

### `llm_agents/tasks/` — Standard Task Library

Ready-made recipes for common LLM tasks.

| File | What it does |
|------|-------------|
| `standard.py` | `summarize()`, `qa()`, `classify()`, `debate()`, `chain()`, `map_reduce()` |
| `types.py` | Result dataclasses for each task type |

**Tests:** `test_standard_tasks.py`

---

### `llm_agents/prompts/` — Prompt Templates

Reusable, composable prompt templates with variable substitution.

| File | What it does |
|------|-------------|
| `template.py` | `PromptTemplate` with Jinja2-style variable rendering |
| `library.py` | Pre-built templates: chain-of-thought, few-shot, persona, etc. |
| `composer.py` | `PromptComposer` for layering and combining templates |

**Tests:** (covered via agent and task tests)

---

### `llm_agents/eval/` — Evaluation Framework

Systematic benchmarking and regression testing for agents.

| File | What it does |
|------|-------------|
| `dataset.py` | `EvalDataset` loaders (JSON, CSV, JSONL, HuggingFace) |
| `metrics.py` | Metrics: `exact_match`, `fuzzy_match`, `llm_judge`, `factual_consistency` |
| `runner.py` | `EvalRunner` with parallel execution |
| `compare.py` | Side-by-side comparison with statistical significance testing |

**Tests:** (covered via integration tests)

---

### `llm_agents/plugins/` — Plugin System

Load third-party tools dynamically at runtime.

| File | What it does |
|------|-------------|
| `loader.py` | Load tools from Python files, modules, or PyPI packages |
| `manifest.py` | Plugin manifest format (YAML-based) |
| `sandbox.py` | Sandboxed plugin execution with permission controls |

**Tests:** `test_plugins.py`

---

## Design Patterns Used

These patterns appear throughout the codebase. Understanding them will help you write code that fits in naturally.

| Pattern | Where it appears | How it works |
|---------|-----------------|-------------|
| **Abstract base class** | `models/base.py`, `memory/store.py` | Define an interface; concrete classes implement it |
| **Decorator** | `tools/base.py` (`@tool`) | Turn a plain function into a `Tool` with auto-generated schema |
| **Registry** | `models/registry.py`, `tools/registry.py` | Store named instances for lookup by string key |
| **Middleware / Wrapper** | `cache/middleware.py`, `ratelimit/middleware.py` | Wrap a model to add cross-cutting behavior transparently |
| **Chain of responsibility** | `safety/guardrails.py` | Run input through a sequence of filters; any can reject |
| **Strategy** | `routing/router.py` | Swap routing algorithms without changing caller code |
| **Frozen dataclasses** | `models/types.py`, `tracing/tracer.py` | Immutable value objects with `@dataclass(frozen=True)` |

---

## How to Add a New Feature

### Adding a new tool

1. Create a function in `llm_agents/tools/builtin.py` (or a new file if it's complex).
2. Decorate it with `@tool`:
   ```python
   from llm_agents.tools.base import tool

   @tool
   def my_tool(query: str, limit: int = 10) -> str:
       """Short description shown to the LLM."""
       return "result"
   ```
3. Pass it to an agent: `Agent(model=model, tools=[my_tool])`.
4. Add a test in `tests/test_tools.py`.

### Adding a new model backend

1. Create a new file in `llm_agents/models/`, e.g. `anthropic_model.py`.
2. Subclass `BaseModel` from `models/base.py` and implement `generate()` and `get_logprobs()`.
3. Add a test in `tests/test_<name>_model.py`.

### Adding a new subpackage

1. Create a directory under `llm_agents/` with an `__init__.py`.
2. Follow the existing pattern: put data types in `types.py`, core logic in descriptively named files, and any middleware wrappers in `middleware.py`.
3. Add corresponding `tests/test_<name>.py`.

### Adding a new agent role

1. Open `llm_agents/agents/roles.py`.
2. Add a new role definition following the existing pattern (name, system prompt, default tools).
3. Optionally add it to a team in `llm_agents/agents/team.py`.

---

## How to Run Tests

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_agent.py -v

# Run tests matching a keyword
pytest -k "test_generation"
```

Each test file corresponds to one or more source modules. Tests use mock models with configurable responses so they run without API keys.

---

## Test File to Source Module Mapping

| Test file | Tests code in |
|-----------|--------------|
| `test_models_types.py` | `models/types.py` |
| `test_openai_model.py` | `models/openai_model.py` |
| `test_streaming.py` | `models/streaming.py`, `agents/streaming_agent.py` |
| `test_agent.py` | `agents/agent.py`, `agents/cot.py` |
| `test_multi_agent.py` | `agents/multi_agent.py`, `agents/message_bus.py` |
| `test_reflection.py` | `agents/reflection.py` |
| `test_consensus.py` | `agents/consensus.py` |
| `test_fsm.py` | `agents/fsm.py` |
| `test_tools.py` | `tools/base.py`, `tools/registry.py`, `tools/executor.py` |
| `test_sandbox.py` | `tools/sandbox.py`, `tools/sandbox_manager.py` |
| `test_logprob_ops.py` | `logprobs/ops.py` |
| `test_tree.py` | `logprobs/tree.py` |
| `test_uncertainty.py` | `logprobs/uncertainty.py` |
| `test_memory.py` | `memory/store.py`, `memory/embeddings.py`, `memory/chunker.py` |
| `test_validation.py` | `validation/schema.py`, `validation/retry.py` |
| `test_tracing.py` | `tracing/tracer.py`, `tracing/export.py` |
| `test_safety.py` | `safety/input_filter.py`, `safety/output_filter.py`, `safety/guardrails.py` |
| `test_routing.py` | `routing/classifier.py`, `routing/router.py` |
| `test_ratelimit.py` | `ratelimit/limiter.py`, `ratelimit/budget.py`, `ratelimit/middleware.py` |
| `test_standard_tasks.py` | `tasks/standard.py` |
| `test_workflows.py` | `workflows/schema.py`, `workflows/engine.py`, `workflows/loader.py` |
| `test_plugins.py` | `plugins/loader.py`, `plugins/manifest.py` |

---

## Dependency Flow

Understanding which packages depend on which helps you know what might break when you change something.

```
models  (no internal deps — safe to change in isolation)
  └─► tools  (depends on models for execution context)
  └─► logprobs  (depends on models for token probabilities)
  └─► agents  (depends on models + tools + logprobs)
        └─► tasks  (depends on agents)
        └─► workflows  (depends on agents + tools)
        └─► eval  (depends on agents + tasks)

memory, validation, prompts  (depend on models only)
tracing, safety, routing, ratelimit, cache  (middleware — wrap models)
plugins  (depends on tools)
```

Changes to `models/base.py` or `models/types.py` have the widest blast radius. Changes to leaf packages like `tasks/`, `eval/`, or `plugins/` are the safest.

---

## Maintenance Checklist

When making changes to this repo:

1. **Run `pytest tests/`** before and after your changes.
2. **Use type hints** — the codebase uses `from __future__ import annotations` and full type annotations everywhere.
3. **Use frozen dataclasses** for value objects (data that shouldn't be mutated after creation).
4. **Follow the existing module layout** — types in `types.py`, core logic in named files, middleware in `middleware.py`.
5. **One test file per module** — add tests alongside your code.
6. **Keep mock models in tests** — don't require real API keys to run the test suite.
