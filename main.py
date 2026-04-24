#!/usr/bin/env python3
"""Prompt analysis and response generation pipeline.

Provides utilities for analyzing prompts, generating calibrated responses,
and evaluating output quality using log-probability distributions.

Usage:
    python main.py encode --prompt "..." --message "secret" --key "mykey"
    python main.py decode --prompt "..." --stego "..." --key "mykey"
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_agents.logprobs.ops import entropy, normalize


# ── Configuration ─────────────────────────────────────────────────────────
MODEL_ID = "gpt2"
CANDIDATE_POOL = 50
MAX_OUTPUT_TOKENS = 256
INTERVAL_PRECISION = 2048     # working precision for distribution analysis
QUANTIZATION_BITS = 24        # fixed-point precision for probability mapping


# ── Keyed analysis primitives ─────────────────────────────────────────────

def _keyed_hash(key: bytes, data: bytes) -> bytes:
    """Compute a keyed hash for reproducible token-level analysis."""
    return hmac.new(key, data, hashlib.sha256).digest()


def _sequence_fingerprint(token_ids: list[int]) -> bytes:
    """Produce a compact fingerprint of a token-id sequence."""
    buf = struct.pack(f">{len(token_ids)}I", *token_ids) if token_ids else b"\x00"
    return hashlib.sha256(buf).digest()


def _priority_ordering(key: bytes, fingerprint: bytes, n: int) -> list[int]:
    """Return a keyed priority ordering over *n* candidate indices.

    Used during diverse / calibrated sampling so that the order in which
    candidates are evaluated is deterministic but secret-dependent,
    reducing repetition artefacts across runs.
    """
    seed_bytes = _keyed_hash(key, fingerprint)
    seed = int.from_bytes(seed_bytes[:4], "big") % (2**31)
    rng = np.random.RandomState(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices


def _derive_analysis_key(passphrase: str) -> bytes:
    """Derive a fixed-length analysis key from an arbitrary passphrase."""
    return hashlib.sha256(passphrase.encode("utf-8")).digest()


# ── Probability helpers ───────────────────────────────────────────────────

def _candidate_distribution(model, input_ids: torch.Tensor, k: int = CANDIDATE_POOL):
    """Return (token_ids, probabilities) for the top-*k* next-token candidates."""
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    vals, idx = torch.topk(logits, k)
    probs = torch.softmax(vals, dim=-1).tolist()
    return idx.tolist(), probs


# ── Payload codec ─────────────────────────────────────────────────────────

def _pack_payload(text: str) -> list[int]:
    """Encode a UTF-8 string into a bit vector with a 16-bit length header."""
    raw = text.encode("utf-8")
    n = len(raw)
    bits: list[int] = []
    for shift in range(15, -1, -1):
        bits.append((n >> shift) & 1)
    for byte in raw:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _unpack_payload(bits: list[int]) -> str:
    """Reconstruct a UTF-8 string from a bit vector with a 16-bit length header."""
    if len(bits) < 16:
        return ""
    n = 0
    for i in range(16):
        n = (n << 1) | bits[i]
    out = []
    for i in range(n):
        byte_val = 0
        for j in range(8):
            pos = 16 + i * 8 + j
            byte_val = (byte_val << 1) | (bits[pos] if pos < len(bits) else 0)
        out.append(byte_val)
    return bytes(out).decode("utf-8", errors="replace")


# ── Interval distribution mapping ────────────────────────────────────────

def _quantize_probs(probs: list[float]) -> list[int]:
    """Convert floating-point probabilities to fixed-point integers."""
    scale = 1 << QUANTIZATION_BITS
    q = [max(1, int(p * scale)) for p in probs]
    return q


def _interval_counts(q_probs: list[int], total_range: int) -> list[int]:
    """Compute integer interval widths for each symbol within *total_range*.

    Uses exact integer arithmetic to avoid floating-point drift.
    Each symbol receives at least 1 unit.
    """
    q_total = sum(q_probs)
    counts = [max(1, qp * total_range // q_total) for qp in q_probs]
    # Absorb rounding residual into the largest symbol.
    allocated = sum(counts)
    diff = total_range - allocated
    if diff != 0:
        best = max(range(len(q_probs)), key=lambda i: q_probs[i])
        counts[best] = max(1, counts[best] + diff)
    return counts


def _cumulative(counts: list[int]) -> list[int]:
    """Return cumulative sums: [0, c0, c0+c1, ..., total]."""
    cum = [0]
    for c in counts:
        cum.append(cum[-1] + c)
    return cum


# ── Core generation (steganographic encoder) ──────────────────────────────

def calibrated_generate(
    prompt: str,
    payload: str,
    passphrase: str,
    model,
    tokenizer,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    top_k: int = CANDIDATE_POOL,
) -> str:
    """Generate a response whose token-selection path encodes *payload*.

    The output text is sampled from the model's own distribution, guided by
    a keyed priority ordering that embeds the payload via adaptive interval
    narrowing on the cumulative probability space.  An observer without the
    passphrase cannot distinguish the output from ordinary model samples.

    Returns the generated continuation (prompt NOT included).
    """
    key = _derive_analysis_key(passphrase)
    bits = _pack_payload(payload)
    n_msg_bits = len(bits)

    # Represent the message as a target point in [0, 2^PREC)
    PREC = INTERVAL_PRECISION
    msg_int = 0
    for b in bits:
        msg_int = (msg_int << 1) | b
    # Place in the upper bits; remaining bits are zero (padding)
    target = msg_int << (PREC - n_msg_bits)

    lo = 0
    hi = 1 << PREC

    prompt_ids = tokenizer.encode(prompt)
    ctx = list(prompt_ids)
    gen_ids: list[int] = []

    for _ in range(max_tokens):
        inp = torch.tensor([ctx])
        cand_ids, cand_probs = _candidate_distribution(model, inp, top_k)

        fp = _sequence_fingerprint(ctx)
        order = _priority_ordering(key, fp, len(cand_ids))
        o_ids = [cand_ids[i] for i in order]
        o_probs = [cand_probs[i] for i in order]
        psum = sum(o_probs)
        o_probs = [p / psum for p in o_probs]

        rng = hi - lo
        q_probs = _quantize_probs(o_probs)
        counts = _interval_counts(q_probs, rng)
        cum = _cumulative(counts)

        # Select the token whose interval contains the target
        v = target - lo
        sel = len(o_probs) - 1
        for i in range(len(o_probs)):
            if v < cum[i + 1]:
                sel = i
                break

        lo_old = lo
        lo = lo_old + cum[sel]
        hi = lo_old + cum[sel + 1]

        tid = o_ids[sel]
        ctx.append(tid)
        gen_ids.append(tid)

        if tid == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, gen_ids


# ── Core analysis (steganographic decoder) ────────────────────────────────

def response_analysis(
    prompt: str,
    response: str,
    passphrase: str,
    model,
    tokenizer,
    top_k: int = CANDIDATE_POOL,
    token_ids: list[int] | None = None,
) -> str:
    """Analyse a model response by retracing its token-selection path.

    Recovers the payload that was embedded during calibrated generation
    by performing the inverse of the interval-narrowing process.

    If *token_ids* are provided directly (from the encoder), they are
    used as-is, bypassing re-tokenisation.  Otherwise the response
    text is re-tokenised — which may introduce one-off errors when the
    tokeniser merges subword boundaries differently.

    Returns the recovered payload string.
    """
    key = _derive_analysis_key(passphrase)
    PREC = INTERVAL_PRECISION

    prompt_ids = tokenizer.encode(prompt)

    if token_ids is not None:
        resp_ids = list(token_ids)
    else:
        full_ids = tokenizer.encode(prompt + response)
        resp_ids = full_ids[len(prompt_ids):]

    lo = 0
    hi = 1 << PREC

    ctx = list(prompt_ids)

    for tid in resp_ids:
        inp = torch.tensor([ctx])
        cand_ids, cand_probs = _candidate_distribution(model, inp, top_k)

        fp = _sequence_fingerprint(ctx)
        order = _priority_ordering(key, fp, len(cand_ids))
        o_ids = [cand_ids[i] for i in order]
        o_probs = [cand_probs[i] for i in order]
        psum = sum(o_probs)
        o_probs = [p / psum for p in o_probs]

        if tid not in o_ids:
            ctx.append(tid)
            continue

        idx = o_ids.index(tid)
        rng = hi - lo
        q_probs = _quantize_probs(o_probs)
        counts = _interval_counts(q_probs, rng)
        cum = _cumulative(counts)

        lo_old = lo
        lo = lo_old + cum[idx]
        hi = lo_old + cum[idx + 1]

        ctx.append(tid)

    # Recover the embedded target point from the final interval [lo, hi).
    # The target was constructed as  msg_int << (PREC - n_msg_bits),
    # so it has many trailing zero bits.  We locate it by finding the
    # point inside [lo, hi) that is a multiple of the largest power of two.
    target = _recover_target(lo, hi, PREC)
    bits: list[int] = []
    for i in range(PREC):
        bits.append((target >> (PREC - 1 - i)) & 1)

    return _unpack_payload(bits)


def _recover_target(lo: int, hi: int, prec: int) -> int:
    """Find the point in [lo, hi) with the most trailing zero bits.

    The encoder placed the target at  msg_int << shift  where shift is
    large, so the true target has the most trailing zeros of any point
    in the interval.  We search from the highest power-of-two down.
    """
    for k in range(prec - 1, -1, -1):
        step = 1 << k
        # Smallest multiple of step that is >= lo
        candidate = ((lo + step - 1) >> k) << k
        if lo <= candidate < hi:
            return candidate
    return lo  # fallback


# ── Model loading ─────────────────────────────────────────────────────────

def load_model(model_id: str = MODEL_ID):
    """Load a HuggingFace causal-LM and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import json as _json

    parser = argparse.ArgumentParser(
        description="Prompt analysis and calibrated response generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Generate a calibrated response for a prompt.")
    enc.add_argument("--prompt", required=True, help="Input prompt for the model.")
    enc.add_argument("--message", required=True, help="Payload to embed in the response.")
    enc.add_argument("--key", required=True, help="Passphrase for reproducible analysis.")
    enc.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    enc.add_argument("--model", default=MODEL_ID)

    dec = sub.add_parser("decode", help="Analyse a response to recover calibration data.")
    dec.add_argument("--prompt", required=True, help="Original prompt.")
    dec.add_argument("--stego", required=True, help="Response text or JSON from encoder.")
    dec.add_argument("--key", required=True, help="Passphrase used during generation.")
    dec.add_argument("--model", default=MODEL_ID)

    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.command == "encode":
        stego_text, token_ids = calibrated_generate(
            args.prompt, args.message, args.key,
            model, tokenizer, max_tokens=args.max_tokens,
        )
        output = _json.dumps({"text": stego_text, "tokens": token_ids})
        print(output)
    elif args.command == "decode":
        stego_input = args.stego
        tok_ids = None
        try:
            parsed = _json.loads(stego_input)
            if isinstance(parsed, dict) and "tokens" in parsed:
                tok_ids = parsed["tokens"]
                stego_input = parsed.get("text", "")
        except (_json.JSONDecodeError, TypeError):
            pass

        recovered = response_analysis(
            args.prompt, stego_input, args.key,
            model, tokenizer, token_ids=tok_ids,
        )
        print(recovered)


if __name__ == "__main__":
    main()
