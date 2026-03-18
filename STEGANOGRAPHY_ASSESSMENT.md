# Steganography Risk Assessment

## Scope

Assessment of whether the inter-agent communication system in this repository
could be used to execute the steganographic scheme described in Or Zamir's
"Undetectable Steganography for Language Models" between cooperating agents.

## Background: Zamir's Scheme

The scheme requires four tightly coupled components operating end-to-end:

1. **Shared secret key** between sender and receiver agents (shared randomness).
2. **Sender-side coupling function** that biases token selection from the model's
   probability distribution to encode hidden bits, while preserving the marginal
   output distribution (making it computationally undetectable).
3. **Receiver-side decoding** that recomputes the sender model's probability
   distribution over the observed tokens and applies the inverse coupling with
   the shared key to extract the hidden message.
4. **Coupling/arithmetic-coding primitives** that map between the probability
   space and the hidden bit stream.

All four must be present *and connected* for the scheme to function.

## Assessment

### Component 1: Shared Secret Key — NOT PRESENT

- `MessageBus` (`agents/message_bus.py`) delivers messages with a `metadata`
  dict, but no code populates it with cryptographic keys or shared randomness.
- No key exchange, seed sharing, or pre-shared secret mechanism exists between
  any agents.

### Component 2: Sender-Side Token-Level Control — NOT POSSIBLE

- Agents generate output by calling `model.generate()` (`agents/agent.py:238`),
  which returns a complete string (`CompletionResult.text`).
- The `HFModel` delegates to `transformers` `model.generate()` internally;
  the `OpenAIModel` delegates to the OpenAI API. In both cases, the token
  sampling loop is **opaque to the agent**.
- There is no hook, callback, or override point where a steganographic coupling
  function could intercept individual token selection decisions.
- The agent loop (`agents/agent.py:230–291`) treats model output as an opaque
  string and only inspects it for tool-call JSON patterns.

### Component 3: Receiver-Side Decoding — NOT PRESENT

- When a receiving agent gets a message via the `MessageBus`
  (`agents/multi_agent.py:204`), it processes it as a plain-text prompt:
  `agent.run(msg.content)`.
- There is no code path that:
  - Re-tokenizes the received message.
  - Recomputes the sender model's probability distribution over those tokens.
  - Applies an inverse coupling function with a shared key to extract bits.
- The `logprobs` module (`logprobs/ops.py`, `logprobs/tree.py`,
  `logprobs/uncertainty.py`) provides analytical utilities (entropy, perplexity,
  KL divergence, probability trees), but none of these are invoked on received
  inter-agent messages.

### Component 4: Coupling / Arithmetic Coding Primitives — NOT PRESENT

- No arithmetic coding, interval-based coupling, or bit-to-token mapping
  functions exist anywhere in the codebase.
- The `logprobs/sampling.py` module implements `diverse_sample` and
  `self_consistency`, which are standard ML techniques unrelated to
  steganographic encoding.

## Individual Building Blocks (Not Connected)

The codebase does contain individual building blocks that are *prerequisites*
for steganography, but they are not wired together into a steganographic
pipeline:

| Building Block | Location | Connected to Steg Pipeline? |
|---|---|---|
| Inter-agent messaging | `agents/message_bus.py` | No |
| Log-probability access | `models/base.py`, `models/hf_model.py`, `models/openai_model.py` | No |
| Probability tree exploration | `logprobs/tree.py` | No |
| Entropy / KL divergence | `logprobs/ops.py` | No |
| Top-k token retrieval | `logprobs/ops.py` | No |
| Token uncertainty analysis | `logprobs/uncertainty.py` | No |

## Conclusion

**The entire Zamir steganographic scheme cannot be running between agents in the
current version of this repository.** The critical architectural gap is that
agents have no control over individual token selection during generation — the
sampling loop is fully encapsulated inside the model backends. Without the
ability to apply a coupling function during token sampling, the encoding half
of the scheme is impossible. Additionally, no shared-key infrastructure or
decoding logic exists on the receiving side.
