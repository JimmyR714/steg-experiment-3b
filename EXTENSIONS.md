# Multi-Agent System — Extension Proposals

This document proposes extensions to the existing multi-agent LLM system.
Each extension is self-contained and includes motivation, design sketch,
affected modules, and estimated scope.

---

## 1. Agent Memory & Retrieval (RAG)

**Motivation:** Agents currently have no memory beyond their conversation
history within a single `run()` call. Long-running or multi-session agents
need persistent memory, and agents working with large documents need
retrieval-augmented generation.

**Design:**
- `llm_agents/memory/store.py`:
  - `MemoryStore` abstract base with `add(text, metadata)`, `search(query, k)`,
    `clear()`.
  - `InMemoryStore` — uses numpy cosine similarity over embeddings.
  - `PersistentStore` — wraps SQLite + numpy for disk-backed storage.
- `llm_agents/memory/embeddings.py`:
  - `Embedder` protocol with `embed(texts: list[str]) -> np.ndarray`.
  - `HFEmbedder` — wraps sentence-transformers locally.
  - `OpenAIEmbedder` — wraps the OpenAI embeddings API.
- `llm_agents/memory/chunker.py`:
  - `chunk_text(text, chunk_size, overlap)` — sliding-window text splitter.
  - `chunk_by_separator(text, sep)` — split on paragraphs/sections.
- Integration: `Agent` gains an optional `memory: MemoryStore` parameter.
  Before each LLM call, relevant memories are retrieved and injected into the
  system prompt as context. A `remember` tool is auto-injected so agents can
  explicitly store information.
- New built-in tools: `store_memory(content)`, `recall(query, k)`.

**Affected modules:** `agents/agent.py`, new `memory/` package.
**Estimated LoC:** ~400

---

## 2. Structured Output & Validation

**Motivation:** Many tasks require output in a specific format (JSON, YAML,
lists with particular schemas). Currently there is no mechanism to enforce
output structure or retry on malformed results.

**Design:**
- `llm_agents/validation/schema.py`:
  - `OutputSchema` — wraps a JSON Schema or Pydantic model.
  - `validate(text, schema) -> ValidationResult` — attempts to parse and
    validate the agent's output against the schema.
  - `extract_json(text) -> dict` — robust JSON extraction from mixed text
    (handles markdown code fences, trailing commas, etc.).
- `llm_agents/validation/retry.py`:
  - `with_retry(agent, schema, max_attempts=3) -> AgentResponse` — runs
    agent, validates output; on failure, feeds the validation error back
    as a user message and retries.
  - Configurable backoff strategy (immediate, linear, exponential).
- `llm_agents/validation/formats.py`:
  - Pre-built schemas for common formats: `json_object`, `json_array`,
    `csv_row`, `markdown_table`, `yaml_document`.
  - `constrained_choice(options)` — ensures output is one of N choices.

**Affected modules:** new `validation/` package, `agents/agent.py` (optional
schema parameter on `run()`).
**Estimated LoC:** ~350

---

## 3. Agent Self-Reflection & Critique

**Motivation:** Single-pass generation often produces suboptimal results.
A reflection loop where the agent reviews and improves its own output can
significantly improve quality without needing multiple agents.

**Design:**
- `llm_agents/agents/reflection.py`:
  - `ReflectiveAgent(agent, critic_prompt, max_rounds=2)` — wraps an agent.
    After each response, the same (or a different) model evaluates the
    output against criteria and either accepts it or provides feedback.
  - `SelfCritique` — the agent critiques its own output using a second pass
    with a critique-focused system prompt.
  - `PeerCritique` — a separate agent provides the critique.
- Critique protocol:
  - Critic returns `{"accept": bool, "feedback": str, "score": float}`.
  - If not accepted, feedback is appended to conversation and agent retries.
- Integration with logprobs: use model confidence (entropy/perplexity) as
  an additional signal for whether reflection is needed. High-entropy
  responses automatically trigger reflection.

**Affected modules:** new file in `agents/`, `logprobs/ops.py` for
confidence thresholds.
**Estimated LoC:** ~250

---

## 4. Execution Tracing & Observability

**Motivation:** Debugging multi-agent systems is difficult. A structured
tracing system makes it possible to replay, visualize, and analyze agent
interactions.

**Design:**
- `llm_agents/tracing/tracer.py`:
  - `Tracer` — context manager that records all events in a trace.
  - `TraceEvent(timestamp, agent_name, event_type, data)` — covers:
    model calls, tool executions, messages sent/received, thinking blocks.
  - `Span` — groups related events (e.g., one agent turn = one span).
- `llm_agents/tracing/export.py`:
  - `to_json(trace)` — serialize full trace to JSON.
  - `to_chrome_trace(trace)` — export to Chrome Tracing format for
    visualization in `chrome://tracing`.
  - `to_opentelemetry(trace)` — convert to OTLP spans for integration
    with Jaeger/Zipkin/Grafana.
- `llm_agents/tracing/cost.py`:
  - `TokenCounter` — tracks prompt/completion tokens per model call.
  - `CostEstimator(price_per_1k_tokens)` — estimates dollar cost.
  - `BudgetGuard(max_tokens)` — raises `BudgetExceededError` when limit hit.
- Integration: `Agent.run()` and `MultiAgentSystem.run_task()` accept an
  optional `tracer` parameter. All internal operations emit trace events.

**Affected modules:** new `tracing/` package, `agents/agent.py`,
`agents/multi_agent.py`.
**Estimated LoC:** ~400

---

## 5. Consensus & Voting Mechanisms

**Motivation:** When multiple agents produce answers, the system needs
principled ways to aggregate them beyond simple coordinator selection.

**Design:**
- `llm_agents/agents/consensus.py`:
  - `MajorityVote(agents, prompt) -> ConsensusResult` — each agent answers
    independently, majority wins. Ties broken by average logprob confidence.
  - `WeightedVote(agents, weights, prompt)` — weighted by agent reliability
    or domain expertise scores.
  - `RankedChoice(agents, prompt, rounds)` — iterative elimination voting.
  - `DebateConsensus(agents, prompt, max_rounds)` — agents debate until
    they converge on an answer or max rounds exhausted.
- `ConsensusResult(answer, vote_distribution, confidence, dissenting_views)`.
- Logprob integration: per-agent confidence scores derived from sequence
  perplexity weight the votes automatically.

**Affected modules:** new file in `agents/`, uses `logprobs/ops.py`.
**Estimated LoC:** ~300

---

## 6. Workflow DSL & Declarative Pipelines

**Motivation:** Complex multi-agent workflows currently require writing
Python code. A declarative YAML/JSON-based workflow definition makes it
easier to compose, share, and modify pipelines.

**Design:**
- `llm_agents/workflows/schema.py`:
  - YAML schema for workflows:
    ```yaml
    workflow:
      name: research_and_summarize
      agents:
        researcher:
          model: gpt-4
          tools: [web_search, read_file]
          system_prompt: "You are a research assistant."
        summarizer:
          model: gpt-3.5-turbo
          system_prompt: "You summarize research findings."
      steps:
        - agent: researcher
          input: "{{task}}"
          output: research_results
        - agent: summarizer
          input: "Summarize: {{research_results}}"
          output: final_summary
      output: "{{final_summary}}"
    ```
- `llm_agents/workflows/engine.py`:
  - `WorkflowEngine` — parses YAML, instantiates agents, executes steps.
  - Variable interpolation with `{{variable}}` syntax.
  - Control flow: `parallel` (fan-out), `conditional` (if/else on output),
    `loop` (repeat until condition).
- `llm_agents/workflows/loader.py`:
  - `load_workflow(path) -> Workflow`
  - `validate_workflow(workflow) -> list[str]` — returns validation errors.

**Affected modules:** new `workflows/` package, integrates with `agents/`
and `tasks/`.
**Estimated LoC:** ~450

---

## 7. Adaptive Routing & Model Selection

**Motivation:** Not all queries need the most expensive model. An intelligent
router can reduce cost and latency by dispatching to the right model based
on task complexity.

**Design:**
- `llm_agents/routing/classifier.py`:
  - `ComplexityClassifier` — classifies incoming prompts as simple/medium/hard
    using a lightweight model or heuristic features (length, keyword presence,
    required reasoning depth).
  - Features: token count, question type, domain detection, presence of
    code/math, multi-step indicators.
- `llm_agents/routing/router.py`:
  - `ModelRouter(routes: dict[str, BaseModel])` — maps complexity levels to
    models. E.g., simple -> gpt-3.5-turbo, hard -> gpt-4.
  - `CascadeRouter(models: list[BaseModel], validator)` — tries cheapest
    model first; if output fails validation, escalates to next model.
  - `LatencyRouter(models, timeout)` — races multiple models, returns first
    valid response.
- `llm_agents/routing/budget.py`:
  - `BudgetRouter(models, token_budget)` — selects model to stay within
    a cumulative token budget across a session.

**Affected modules:** new `routing/` package, integrates with `models/`.
**Estimated LoC:** ~350

---

## 8. Sandboxed Code Execution Tool

**Motivation:** Agents that can write and execute code are dramatically more
capable for data analysis, math, and programming tasks. This requires a
safe execution environment.

**Design:**
- `llm_agents/tools/sandbox.py`:
  - `PythonSandbox` — executes Python code in a restricted subprocess with:
    - Timeout enforcement (default 30s).
    - Memory limit (default 256MB).
    - No network access, no filesystem writes outside a temp directory.
    - Captured stdout/stderr returned as string.
  - `DockerSandbox` — optional; runs code in an ephemeral Docker container
    for stronger isolation.
- New built-in tools:
  - `execute_python(code: str) -> str` — runs code, returns output.
  - `execute_shell(command: str) -> str` — runs shell command in sandbox.
- `llm_agents/tools/sandbox_manager.py`:
  - Manages sandbox lifecycle, temp directory cleanup, result caching.
  - Tracks resource usage per agent for observability.

**Affected modules:** new files in `tools/`, `tools/builtin.py` updated.
**Estimated LoC:** ~300

---

## 9. Prompt Template Library

**Motivation:** System prompts, tool descriptions, and task instructions
contain repetitive patterns. A template system reduces duplication and
makes prompt engineering more systematic.

**Design:**
- `llm_agents/prompts/template.py`:
  - `PromptTemplate(template_str, variables)` — Jinja2-style templating
    with `{{ variable }}` substitution.
  - `render(template, **kwargs) -> str` — renders template with values.
  - `ChatTemplate(system, user, assistant)` — structured multi-turn template.
- `llm_agents/prompts/library.py`:
  - Pre-built templates for common patterns:
    - `CHAIN_OF_THOUGHT` — standard CoT elicitation.
    - `FEW_SHOT` — formats examples into few-shot prompt.
    - `PERSONA` — role-play system prompt with configurable traits.
    - `STRUCTURED_OUTPUT` — instructs model to output specific format.
    - `SELF_CRITIQUE` — reflection-prompting template.
    - `TOOL_USE` — standardized tool-use instructions.
- `llm_agents/prompts/composer.py`:
  - `PromptComposer` — builds complex prompts by layering templates.
  - Supports conditional sections, loops over examples, and includes.

**Affected modules:** new `prompts/` package, used by `agents/` and `tasks/`.
**Estimated LoC:** ~300

---

## 10. Agent Roles & Persona Presets

**Motivation:** Configuring agents from scratch is tedious. Pre-built
personas with appropriate system prompts, tools, and behavioral settings
accelerate development and encourage best practices.

**Design:**
- `llm_agents/agents/roles.py`:
  - `AgentRole` dataclass: `name, system_prompt, default_tools, enable_cot,
    temperature, description`.
  - Built-in roles:
    - `Researcher` — web search, note-taking, citation-aware.
    - `Coder` — code execution, file read/write, debugging focus.
    - `Analyst` — calculator, data processing, structured output.
    - `Writer` — long-form generation, editing, style-aware.
    - `Critic` — evaluation-focused, rubric-based scoring.
    - `Coordinator` — delegation, synthesis, project management.
    - `FactChecker` — verification, source cross-referencing.
  - `create_agent(role, model, **overrides) -> Agent` — factory function.
- `llm_agents/agents/team.py`:
  - `AgentTeam` — pre-configured multi-agent setups:
    - `research_team()` — researcher + analyst + writer.
    - `code_review_team()` — coder + critic + coordinator.
    - `debate_team(topic)` — two debaters + judge.

**Affected modules:** new files in `agents/`.
**Estimated LoC:** ~300

---

## 11. Caching & Memoization Layer

**Motivation:** Repeated or similar queries waste tokens and latency.
A caching layer with exact and semantic matching can dramatically reduce
costs during development and production.

**Design:**
- `llm_agents/cache/exact.py`:
  - `ExactCache` — hash-based cache keyed on (model, prompt, parameters).
  - Backends: in-memory dict, SQLite, Redis.
  - TTL support and max-size eviction (LRU).
- `llm_agents/cache/semantic.py`:
  - `SemanticCache(embedder, threshold=0.95)` — caches responses and
    returns cached result if a new query's embedding is within cosine
    similarity threshold of a cached query.
  - Useful for paraphrased queries hitting the same intent.
- `llm_agents/cache/middleware.py`:
  - `CachedModel(model, cache)` — wraps any `BaseModel` with transparent
    caching. `generate()` and `get_logprobs()` check cache first.
  - Cache statistics: hit rate, token savings, latency savings.

**Affected modules:** new `cache/` package, wraps `models/`.
**Estimated LoC:** ~300

---

## 12. Evaluation & Benchmarking Framework

**Motivation:** Measuring agent quality requires systematic evaluation.
An eval framework enables regression testing, model comparison, and
prompt optimization.

**Design:**
- `llm_agents/eval/dataset.py`:
  - `EvalDataset` — list of `EvalExample(input, expected, metadata)`.
  - Loaders: from JSON, CSV, JSONL, HuggingFace datasets.
- `llm_agents/eval/metrics.py`:
  - `exact_match(predicted, expected) -> float`
  - `fuzzy_match(predicted, expected, threshold) -> float`
  - `llm_judge(predicted, expected, judge_model) -> Score` — uses an LLM
    to grade quality on a rubric.
  - `factual_consistency(predicted, context) -> float` — checks if
    response is supported by the context.
  - `Composite(*metrics, weights)` — weighted combination.
- `llm_agents/eval/runner.py`:
  - `EvalRunner(agent, dataset, metrics) -> EvalReport`
  - Parallel execution with configurable concurrency.
  - `EvalReport` — per-example scores, aggregates, confidence intervals.
- `llm_agents/eval/compare.py`:
  - `compare(reports: list[EvalReport])` — side-by-side comparison table.
  - Statistical significance testing (bootstrap).

**Affected modules:** new `eval/` package.
**Estimated LoC:** ~450

---

## 13. Streaming & Real-Time Output

**Motivation:** Long generations leave users waiting with no feedback.
Streaming token-by-token output improves perceived latency and enables
real-time monitoring.

**Design:**
- `llm_agents/models/streaming.py`:
  - `StreamingResult` — async iterator yielding `StreamChunk(token, logprob,
    finish_reason)`.
  - `OpenAIModel.generate_stream()` — SSE streaming from OpenAI API.
  - `HFModel.generate_stream()` — token-by-token from local model.
- `llm_agents/agents/streaming_agent.py`:
  - `StreamingAgent` — extends `Agent` with `run_stream()` that yields
    partial results.
  - Tool-call detection mid-stream: buffers until tool-call JSON is
    complete, then executes and resumes streaming.
  - Thinking block detection: yields thinking tokens to a separate
    callback while streaming visible content.
- Callback protocol:
  - `on_token(token, logprob)`
  - `on_thinking(text)`
  - `on_tool_call(name, args)`
  - `on_tool_result(name, result)`
  - `on_complete(response)`

**Affected modules:** `models/openai_model.py`, `models/hf_model.py`,
new files in `agents/`.
**Estimated LoC:** ~350

---

## 14. Rate Limiting & Token Budgets

**Motivation:** API rate limits and cost control are critical for
production deployments. The system needs built-in mechanisms to respect
limits and enforce budgets.

**Design:**
- `llm_agents/ratelimit/limiter.py`:
  - `RateLimiter(requests_per_minute, tokens_per_minute)` — token bucket
    algorithm. `acquire(estimated_tokens)` blocks or raises if over limit.
  - `AdaptiveRateLimiter` — adjusts rate based on 429 responses.
- `llm_agents/ratelimit/budget.py`:
  - `TokenBudget(max_prompt_tokens, max_completion_tokens, max_total)`.
  - `BudgetTracker` — accumulates usage across agent runs.
  - `BudgetExceededError` — raised when budget is exhausted.
  - Budget allocation: divide budget across agents in a multi-agent run.
- `llm_agents/ratelimit/middleware.py`:
  - `RateLimitedModel(model, limiter, budget)` — wraps any model.
  - Automatic retry with backoff on rate-limit errors.
  - Usage reporting: tokens used, remaining budget, estimated cost.

**Affected modules:** new `ratelimit/` package, wraps `models/`.
**Estimated LoC:** ~300

---

## 15. Adversarial Robustness & Safety Filters

**Motivation:** Agents exposed to untrusted input (tool results, user
messages, other agents) need protection against prompt injection and
harmful outputs.

**Design:**
- `llm_agents/safety/input_filter.py`:
  - `InputFilter` — scans incoming text for prompt injection patterns.
  - `detect_injection(text) -> InjectionResult` — regex + heuristic
    detection of common injection patterns (ignore previous instructions,
    system prompt leaks, delimiter attacks).
  - `sanitize(text) -> str` — escapes or removes suspicious patterns.
- `llm_agents/safety/output_filter.py`:
  - `OutputFilter` — scans agent output for policy violations.
  - `ContentClassifier(model)` — uses a lightweight model to classify
    output safety.
  - Configurable block/warn/log behavior.
- `llm_agents/safety/guardrails.py`:
  - `Guardrail(input_filters, output_filters)` — composable filter chain.
  - `GuardedAgent(agent, guardrails)` — wraps an agent with safety checks
    on both input and output.
  - Audit log: all filter triggers are logged with context.

**Affected modules:** new `safety/` package, wraps `agents/`.
**Estimated LoC:** ~350

---

## 16. Agent State Machines

**Motivation:** Complex agent behaviors (multi-step forms, approval
workflows, conditional branching) are hard to express as free-form
conversation. A state machine provides structure.

**Design:**
- `llm_agents/agents/fsm.py`:
  - `State(name, prompt, tools, transitions)` — defines what the agent
    does in each state and what triggers transitions.
  - `Transition(condition, target_state)` — condition can be a regex match
    on output, a tool call name, or a Python callable.
  - `StateMachineAgent(states, initial_state, model)` — runs the agent
    in a loop, transitioning between states based on outputs.
- Example states for a customer-service agent:
  - `greeting` -> `identify_issue` -> `troubleshoot` | `escalate` -> `resolve`
- Visualization: `fsm_to_mermaid(agent) -> str` — generates Mermaid diagram
  of the state machine.

**Affected modules:** new file in `agents/`.
**Estimated LoC:** ~300

---

## 17. Logprob-Based Uncertainty Quantification

**Motivation:** The system already has logprob infrastructure but doesn't
use it for higher-level uncertainty reasoning. Agents should be able to
express calibrated confidence and flag uncertain outputs.

**Design:**
- `llm_agents/logprobs/uncertainty.py`:
  - `confidence_score(logprob_result) -> float` — aggregate confidence
    from sequence logprobs (normalized perplexity).
  - `token_uncertainty_map(logprob_result) -> list[tuple[str, float]]` —
    per-token uncertainty, useful for highlighting uncertain spans.
  - `is_hallucination_risk(logprob_result, threshold) -> bool` — flags
    responses where large spans have high uncertainty.
  - `calibration_curve(predictions, actuals)` — measures how well model
    confidence tracks actual accuracy.
- `llm_agents/logprobs/sampling.py`:
  - `diverse_sample(model, prompt, n, temperature_schedule)` — generates
    n completions with varying temperatures to explore the output space.
  - `self_consistency(model, prompt, n) -> ConsistencyResult` — generates
    n samples and measures agreement (majority answer + confidence).
  - `conformal_prediction(model, prompt, calibration_data, alpha)` —
    returns a prediction set guaranteed to contain the correct answer
    with probability 1-alpha (conformal inference).

**Affected modules:** new files in `logprobs/`, uses `logprobs/ops.py`.
**Estimated LoC:** ~350

---

## 18. Plugin System & Dynamic Tool Loading

**Motivation:** The current tool system requires code changes to add tools.
A plugin system allows loading tools from external packages, directories,
or URLs at runtime.

**Design:**
- `llm_agents/plugins/loader.py`:
  - `load_plugin(path_or_module) -> list[Tool]` — discovers and loads
    `@tool`-decorated functions from a Python file or module.
  - `PluginDirectory(path)` — watches a directory and auto-loads new
    tool files.
  - `load_from_pypi(package_name)` — installs and loads a pip package
    that exposes tools via entry points.
- `llm_agents/plugins/manifest.py`:
  - `PluginManifest` — YAML file describing a plugin:
    ```yaml
    name: weather-tools
    version: 1.0.0
    tools:
      - name: get_weather
        module: weather_tools.api
        function: get_current_weather
    dependencies:
      - requests>=2.28
    ```
- `llm_agents/plugins/sandbox.py`:
  - `SandboxedPlugin` — runs plugin tools in a subprocess for isolation.
  - Permission system: plugins declare required permissions (network,
    filesystem, etc.) and the user must approve.

**Affected modules:** new `plugins/` package, integrates with `tools/`.
**Estimated LoC:** ~350

---

## Implementation Priority Matrix

| Extension                        | Impact | Complexity | Dependencies    |
|----------------------------------|--------|------------|-----------------|
| 4. Tracing & Observability       | High   | Medium     | None            |
| 2. Structured Output             | High   | Low        | None            |
| 3. Self-Reflection               | High   | Low        | None            |
| 17. Uncertainty Quantification   | High   | Medium     | logprobs/       |
| 1. Memory & RAG                  | High   | High       | embeddings      |
| 12. Evaluation Framework         | High   | Medium     | None            |
| 5. Consensus Mechanisms          | Medium | Low        | logprobs/       |
| 7. Adaptive Routing              | Medium | Medium     | models/         |
| 14. Rate Limiting                | Medium | Low        | models/         |
| 6. Workflow DSL                  | Medium | High       | agents/, tasks/ |
| 8. Code Execution Sandbox        | Medium | Medium     | tools/          |
| 13. Streaming                    | Medium | Medium     | models/         |
| 11. Caching                      | Medium | Low        | models/         |
| 9. Prompt Templates              | Low    | Low        | None            |
| 10. Agent Roles                  | Low    | Low        | agents/         |
| 15. Safety Filters               | High   | Medium     | agents/         |
| 16. State Machines               | Medium | Medium     | agents/         |
| 18. Plugin System                | Medium | High       | tools/          |

**Recommended implementation order** (balancing impact and dependencies):
1. Structured Output & Validation (unblocks reliable downstream use)
2. Tracing & Observability (essential for debugging everything after)
3. Self-Reflection & Critique (high quality improvement, low effort)
4. Uncertainty Quantification (leverages existing logprob infrastructure)
5. Consensus Mechanisms (enhances multi-agent quality)
6. Evaluation Framework (enables measuring all improvements)
7. Rate Limiting & Token Budgets (production readiness)
8. Safety Filters (production readiness)
9. Memory & RAG (unlocks new use cases)
10. Everything else based on need

---

## Total Estimated Scope

| Extensions 1-18 combined | ~5,950 LoC |
|--------------------------|------------|

This would roughly triple the codebase from ~2,100 to ~8,000 lines,
transforming it from a capable prototype into a production-grade framework.
