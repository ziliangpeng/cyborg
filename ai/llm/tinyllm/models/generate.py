"""Text generation utilities for TinyLLM."""

from typing import TYPE_CHECKING

from tinygrad import Tensor
from tinygrad.uop.ops import UOp

if TYPE_CHECKING:
    pass


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
        return _generate_with_cache(model, input_ids, max_new_tokens, temperature, top_k, do_sample, kv_cache)
    return _generate_without_cache(model, input_ids, max_new_tokens, temperature, top_k, do_sample)


def _generate_without_cache(model, input_ids, max_new_tokens, temperature, top_k, do_sample):
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

    kv_cache._eager = False  # lazy mode: skip per-layer realize() during prefill
    logits = model(input_ids, kv_cache=kv_cache, start_pos=0)
    kv_cache.flush()  # one GPU sync to realize all accumulated assigns
    next_token = _sample(logits[:, -1, :])
    output = Tensor.cat(input_ids, next_token, dim=1)
    start_pos = input_ids.shape[1]

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
