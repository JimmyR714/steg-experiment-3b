# Critical Evaluation: Steganography Implementation

**Evaluator:** Claude Sonnet 4.6  
**Date:** 2026-04-24  
**Commit evaluated:** `b04ed8f` ("Implemented")  
**Paper:** Zamir, "Undetectable Steganography for Language Models", TMLR 2024  
**Test environment:** GPT-2, PyTorch 2.10, transformers 5.5.4

---

## Executive Summary

The implementation does not implement the scheme described in the paper. It instead implements a different cryptographic steganography method — **arithmetic coding (interval narrowing) with a keyed token permutation** — which is a valid steganography technique but not the one specified. Three of the paper's four fundamental properties are violated: the algorithm, the undetectability guarantee, and the retriever's input interface all diverge from the specification. Despite this, the implementation achieves functional round-trip correctness for short messages under favourable keys, making it a partially working but fundamentally mismatch solution.

---

## 1. Algorithm Compliance

### 1.1 Paper's Algorithm (Algorithms 3 & 4, §6)

The paper builds on the CGZ watermarking scheme extended with a Dynamic ECC with feedback. In brief:

1. **Token space** is reduced to binary `{0,1}` (all LLMs are treated as binary token generators via CGZ's reduction).
2. **Generation:** For each token position `i`, compute `p_i = Pr[token = 1]`. Select `x_i = 1 if F_k(i, next) ≤ p_i`, where `next ∈ {0, 1, ←}` is the current ECC symbol being transmitted and `F_k` is a PRF keyed on the secret.
3. **Score:** After each token, compute three scores `score_σ = Σ s(x_i, F_k(i, σ))` for `σ ∈ {0,1,←}`, where `s(x, k) = ln(1/k)` if `x=1`, else `ln(1/(1-k))`.
4. **ECC commitment:** When `(score_σ − score_len) / √score_len > t`, commit `σ` as the received ECC symbol. Use the Dynamic ECC's backspace mechanism to correct bit errors, restart scores, and transmit the next ECC symbol.
5. **Retrieval:** The decoder (Algorithm 4) takes only `(response, key)` — **not the prompt** — and independently runs the same score accumulation.

### 1.2 What the Implementation Does

The implementation uses **arithmetic coding (AC)** over the LLM's token distribution:

1. Packs the payload into a bit string using a UTF-8 encoding with a 16-bit length header (not the paper's 5-bit alphabet for A–Z, space, and punctuation).
2. Converts the bit string into a single integer `target` placed in the upper bits of `[0, 2^2048)`.
3. For each generation step, takes the top-K candidates, applies a **keyed shuffle** of their order, splits the interval proportionally by probability, and selects the token whose sub-interval contains `target`.
4. The decoder retokenises the response (using the prompt for context), replays the interval narrowing, then calls `_recover_target` to extract the point in `[lo, hi)` with the most trailing zeros.

### 1.3 Verdict

| Property | Paper requires | Implementation | Status |
|---|---|---|---|
| Core algorithm | CGZ score function + Dynamic ECC | Arithmetic coding + keyed permutation | **WRONG** |
| Token space reduction | Binary via CGZ reduction | Top-K with renormalized softmax | **WRONG** |
| ECC with feedback | Yes — ternary `{0,1,←}` with backspace | Absent | **MISSING** |
| Decoder inputs | `(response, key)` only | `(prompt, response, key)` | **WRONG** |
| Output distribution | Unchanged (provably undetectable) | Renormalized over top-K | **WRONG** |
| Payload alphabet | 5-bit codes (A–Z, space, punctuation) | UTF-8 with 16-bit header | **DIFFERENT** |

---

## 2. Undetectability Violation (Critical)

The paper's central security claim (Definition 2.6) is that the encoder's output is **computationally indistinguishable** from the original model's output.

The implementation violates this by renormalising the probability distribution over the top-50 candidates only:

```python
vals, idx = torch.topk(logits, k)
probs = torch.softmax(vals, dim=-1).tolist()   # ← renormalized over top-K only
```

Empirically (across five test prompts with GPT-2):

- **Average excluded probability mass:** 31.2% (tokens outside top-50 are zeroed out)
- **Maximum excluded mass:** 45.6%
- **Minimum excluded mass:** 15.6%

This means every token selected by the encoder has higher probability under the renormalised distribution than under the true model. An adversary with model access can compute the expected log-probability gain per token and detect the encoding with high confidence after a few dozen tokens. The paper's theorem explicitly relies on using the *actual* model probabilities `p_i`; renormalisation to top-K breaks this guarantee entirely.

---

## 3. Correctness: Round-Trip Reliability

Despite not implementing the paper's algorithm, the arithmetic coding approach is **internally consistent** and achieves correct round-trips in many cases. Below are empirical results across 24 (message, key) pairs at `max_tokens=200` (prompt: "Write about technology"):

| Message | Bits | Keys tested | Success rate |
|---|---|---|---|
| "A" | 32 | k1, k2, k3 | 3/3 (100%) |
| "AB"–"ABCDEF" | 40–64 | k1, k2, k3 | 18/18 (100%) |
| "HELLO" | 56 | k1, k2, k3 | 3/3 (100%) |
| "HELLO WORLD" | 104 | k1, k2, k3 | 3/3 (100%) |

However, reliability collapses with unlucky keys:

| Message | Key | Tokens generated | EOS early? | Correct? |
|---|---|---|---|---|
| "HELLO WORLD" | cap_key | 3 | Yes | No — garbled |
| "secret message" | cap_key | 4 | Yes | No — garbled |
| "secret message" | testkey | 6 | Yes | No — garbled |
| "secret message" | key3 | 12 | Yes | No — partial |
| "ABC" | cap_key | 8 | Yes | No — 'AD\x00' |
| "secret message" | longkey | 400 | No | Yes |

**Root cause:** The arithmetic coding assigns each candidate token a sub-interval proportional to its probability. When the `target` point happens to fall in the EOS token's sub-interval at an early step, generation terminates silently before the interval has narrowed enough for correct recovery. The required interval width for a payload of `n` bits is `< 2^(2048 − n)`. With only 3–6 tokens generated (≈15–30 bits of entropy), this threshold is never reached for 100+ bit payloads.

The encoder provides no indication of failure — it silently returns an unusable ciphertext. The user has no way to detect that decoding will fail.

### 3.1 The `_recover_target` Mechanism

`_recover_target` finds the point in `[lo, hi)` with the most trailing zero bits. This is correct when the interval has been narrowed below `2^(PREC − n_bits)`, ensuring the target is the unique such point. When the interval remains too wide (early EOS case), a spurious point with more trailing zeros can appear inside `[lo, hi)`, producing garbage output. This is the exact mechanism producing `'\\x00\\x00...'` output in the failing cases.

---

## 4. Additional Bugs Found

### 4.1 `_interval_counts` Integer Overflow Bug (Latent)

When `total_range < len(candidates)`, the `max(1, ...)` floor applied to each count causes `sum(counts) > total_range`, which the rounding-residual adjustment cannot fix:

```python
# With total_range=25, 50 candidates of equal probability:
# Each gets max(1, 0) = 1 → sum = 50 ≠ 25
# diff = 25 - 50 = -25
# counts[best] = max(1, 1 + (-25)) = max(1, -24) = 1  (still 50 total)
```

Test results confirm: with 50 equal-probability candidates, `_interval_counts` returns sum=50 for any `total_range ≤ 49`. This means the cumulative distribution overflows the intended interval, and `sel` (the chosen token index) will always be set to `len(o_probs) - 1` (the last token in the permuted order), regardless of the target.

With `PREC=2048` and typical generation lengths (<400 tokens), the interval width stays well above 50, so this bug does not manifest in practice. However, it is a latent correctness defect.

### 4.2 Decoder Requires Original Prompt

The paper's Algorithm 4 takes only `(x_1, ..., x_L, key)`. The implementation's `response_analysis` requires the original prompt to reconstruct the model's context at each step. This is a fundamental interface deviation: anyone wishing to use this retriever must have stored the original prompt alongside the stego-text. This breaks the paper's stated use case where a third party can verify a response without knowing the original conversation.

### 4.3 Retokenization Path Brittleness

When token IDs are not available (text-only decode path), the implementation uses:

```python
full_ids = tokenizer.encode(prompt + response)
resp_ids = full_ids[len(prompt_ids):]
```

This subtracts the prompt token count from the concatenated tokenization. For GPT-2 (BPE without special separator tokens) the tested cases all matched exactly. However, this assumption breaks with tokenisers that insert special tokens between prompt and continuation, or where a multi-byte character spans the prompt/response boundary. The correct approach is `tokenizer.encode(response)` after establishing the continuation boundary via the original prompt.

### 4.4 CLI Incompatibility with Provided Test Suite

The provided `test_suite.py` (spec: `decode --text T --key K`) is entirely incompatible with this implementation's CLI (`decode --prompt P --stego T --key K`). Additional missing flags: `--raw-bits`, `--message-bits`. None of the test suite's decode tests would pass without modification.

---

## 5. Tests That Passed

Despite the architectural deviations, the implementation is internally coherent:

| Category | Tests | Status |
|---|---|---|
| Payload codec (pack/unpack UTF-8) | 11/11 | PASS |
| Quantisation & interval arithmetic | 7/7 | PASS |
| Key derivation & priority ordering | 6/6 | PASS |
| `_recover_target` correctness | 3/3 | PASS |
| Encode determinism | 4/4 | PASS |
| Round-trip (short messages, good keys) | 10/10 | PASS |
| Key sensitivity (wrong key gives wrong output) | 1/1 | PASS |
| Retokenisation consistency (GPT-2) | 3/3 | PASS |
| Interval narrowing (target stays in interval) | 3/3 | PASS |
| CLI argument validation (missing required args) | 5/5 | PASS |

---

## 6. Summary of Failures

| # | Finding | Severity |
|---|---|---|
| F1 | Implements arithmetic coding, not the paper's CGZ+ECC algorithm | Critical — wrong scheme |
| F2 | Renormalises over top-K only; breaks undetectability guarantee | Critical — security failure |
| F3 | Decoder requires original prompt; paper's Algorithm 4 does not | Critical — interface deviation |
| F4 | No Dynamic ECC: no backspace, no bit-error correction | Critical — missing component |
| F5 | Silent EOS-early-termination failure for unlucky keys/messages | Major — reliability failure |
| F6 | `_interval_counts` sum bug when `total_range < num_candidates` | Minor — latent, not triggered |
| F7 | Different payload alphabet (UTF-8 vs paper's 5-bit alphabet) | Minor — incompatibility |
| F8 | CLI incompatible with provided test_suite.py | Minor — tooling issue |

---

## 7. Conclusion

The implementation is **not a correct implementation of Zamir (2024)**. It implements a different steganographic scheme — standard arithmetic coding steganography — which is a well-known technique (similar to Meteor, Ding et al. 2023) rather than the CGZ-based score/ECC approach described in the paper. The three critical deviations (wrong algorithm, violated undetectability, prompt-dependent decoder) mean it cannot satisfy the paper's main theorem.

Within its own design space, the arithmetic coding implementation is mostly correct and achieves reliable round-trips for short messages (≤8 bytes) across varied keys. The main correctness risk is the key-dependent premature EOS failure, which causes silent encoding failures for longer messages with unlucky keys. This would require either a retry mechanism, a mechanism to avoid selecting EOS before sufficient entropy has been accumulated, or choosing sufficiently long `max_tokens`.
