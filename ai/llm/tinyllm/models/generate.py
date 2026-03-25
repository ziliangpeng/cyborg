"""Text generation utilities for TinyLLM."""

from typing import TYPE_CHECKING

from tinygrad import Tensor
from tinygrad.uop.ops import UOp

if TYPE_CHECKING:
    from .base import BaseModel
    from ..kv_cache import KVCache


def generate(
    model,
    input_ids: Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k=None,
    do_sample: bool = False,
    kv_cache=None,
) -> Tensor:
    if kv_cache is not None:
        return _generate_with_cache(
            model, input_ids, max_new_tokens, temperature, top_k, do_sample, kv_cache
        )

    for _ in range(max_new_tokens):
        seq_len = input_ids.shape[1]
        if seq_len > model.config.n_positions:
            input_ids = input_ids[:, -model.config.n_positions :]

        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if do_sample:
            if top_k is not None:
                next_token_logits = _top_k_filtering(next_token_logits, top_k)
            probs = next_token_logits.softmax(axis=-1)
            next_token = _multinomial_sample(probs)
        else:
            next_token = next_token_logits.argmax(axis=-1, keepdim=True)

        input_ids = Tensor.cat(input_ids, next_token, dim=1)

    return input_ids


def _generate_with_cache(model, input_ids, max_new_tokens, temperature, top_k, do_sample, kv_cache):
    from ..kv_cache import VariableKVCache

    def _sample(logits: Tensor) -> Tensor:
        if temperature != 1.0:
            logits = logits / temperature
        if do_sample:
            if top_k is not None:
                logits = _top_k_filtering(logits, top_k)
            return _multinomial_sample(logits.softmax(axis=-1))
        return logits.argmax(axis=-1, keepdim=True)

    if isinstance(kv_cache, VariableKVCache):
        return _generate_variable_jit(model, input_ids, max_new_tokens, _sample, kv_cache)
    else:
        return _generate_eager(model, input_ids, max_new_tokens, _sample, kv_cache)


def _generate_eager(model, input_ids, max_new_tokens, _sample, kv_cache):
    logits = model(input_ids, kv_cache=kv_cache)
    next_token = _sample(logits[:, -1, :])
    output = Tensor.cat(input_ids, next_token, dim=1)

    for _ in range(max_new_tokens - 1):
        logits = model(next_token, kv_cache=kv_cache)
        next_token = _sample(logits[:, -1, :])
        output = Tensor.cat(output, next_token, dim=1)
    return output


def _generate_variable_jit(model, input_ids, max_new_tokens, _sample, kv_cache):
    max_ctx = kv_cache.max_seq_len
    T_total = input_ids.shape[1]

    # Persist the Variable UOp on the model so the same object is reused across calls.
    # This allows TinyJit to replay the compiled graph without retracing.
    if not hasattr(model, "_start_pos_var") or model._start_pos_var.arg[2] != max_ctx - 1:
        model._start_pos_var = UOp.variable("start_pos", 1, max_ctx - 1)
    v = model._start_pos_var

    model._active_cache = kv_cache
    # Only reset JIT if not yet compiled (cnt < 2) or if the cache changed.
    # Resetting forces a recompile which is expensive for large models.
    if model._jit.cnt < 2:
        model._jit.reset()

    # --- CHUNKED PREFILL ---
    # If model._chunk_size > 0, split prefill into fixed-size chunks.
    # Benefits: smaller attention matrices per chunk (chunk_size^2 vs T_total^2),
    # and all KV assigns are batched into one GPU sync via flush().
    # We use integer start_pos for each chunk (not symbolic) so that the causal
    # mask stays concrete and correctness is maintained.
    chunk_size = getattr(model, "_chunk_size", 0)

    kv_cache._eager = False  # lazy mode: batch all KV assigns until flush()
    start_pos = 0

    if chunk_size > 0 and T_total > chunk_size:
        # Process full prompt in fixed-size chunks using eager integer start_pos.
        # Attention matrix per chunk = (chunk_size, start_pos + chunk_size)
        # instead of (T_total, T_total) — lower peak memory and compute.
        for chunk_start in range(0, T_total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T_total)
            chunk = input_ids[:, chunk_start:chunk_end]
            chunk_len = chunk.shape[1]
            logits = model._forward(chunk, kv_cache, start_pos=start_pos)
            start_pos += chunk_len

        kv_cache.flush()  # one GPU sync after all prefill chunks
    else:
        # No chunking: process full prompt in one forward pass (original behavior)
        logits = model(input_ids, kv_cache=kv_cache, start_pos=0)
        kv_cache.flush()           # one GPU sync to realize all accumulated assigns
        start_pos = T_total

    next_token = _sample(logits[:, -1, :])
    output = Tensor.cat(input_ids, next_token, dim=1)

    for _ in range(max_new_tokens - 1):
        sp = v.bind(start_pos)
        logits = model(next_token, kv_cache=kv_cache, start_pos=sp)
        next_token = _sample(logits[:, -1, :])
        output = Tensor.cat(output, next_token, dim=1)
        start_pos += 1

    model._active_cache = None
    return output
def _top_k_filtering(logits: Tensor, k: int) -> Tensor:
    top_vals, _ = logits.topk(k)
    kth_val = top_vals[:, -1:]
    mask = logits < kth_val
    return logits * (1 - mask.float()) + mask.float() * -1e9


def _multinomial_sample(probs: Tensor) -> Tensor:
    return probs.multinomial(num_samples=1)
