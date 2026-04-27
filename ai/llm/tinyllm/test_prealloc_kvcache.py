"""Test PreallocKVCache correctness and speedup vs SimpleKVCache."""

import time
from tinygrad import Tensor
from ai.llm.tinyllm import GPT2, SimpleKVCache, PreallocKVCache, generate
from ai.llm.tinyllm.utils import Tokenizer

model = GPT2.from_pretrained("gpt2")
tokenizer = Tokenizer.for_model("gpt2")

cfg = model.config
base = "The quick brown fox jumps over the lazy dog. "
long_text = base * 100

generate_tokens = 30
print(f"GPT-2 — {generate_tokens} new tokens, varying prefix length\n")
print(f"{'prefix':>8} | {'no_cache':>10} | {'simple':>10} | {'prealloc':>10} | {'speedup':>8} | ok")
print("-" * 70)

for target_prefix in [50, 200, 500, 900]:
    tokens = tokenizer.encode(long_text)[:target_prefix]
    input_ids = Tensor([tokens])

    # No cache
    t0 = time.perf_counter()
    out_no = generate(model, input_ids, max_new_tokens=generate_tokens)
    t_no = time.perf_counter() - t0

    # SimpleKVCache
    t0 = time.perf_counter()
    out_simple = generate(model, input_ids, max_new_tokens=generate_tokens, kv_cache=SimpleKVCache())
    t_simple = time.perf_counter() - t0

    # PreallocKVCache
    cache = PreallocKVCache(
        max_seq_len=cfg.n_positions,
        n_layers=cfg.n_layer,
        num_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
    )
    t0 = time.perf_counter()
    out_pre = generate(model, input_ids, max_new_tokens=generate_tokens, kv_cache=cache)
    t_pre = time.perf_counter() - t0

    ok = out_no[0].numpy().tolist() == out_pre[0].numpy().tolist()
    speedup = t_no / t_pre
    print(
        f"{len(tokens):>8} | {generate_tokens / t_no:>9.1f}t/s | {generate_tokens / t_simple:>9.1f}t/s | {generate_tokens / t_pre:>9.1f}t/s | {speedup:>7.2f}x | {ok}"
    )
