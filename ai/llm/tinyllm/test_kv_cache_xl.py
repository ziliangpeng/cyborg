import time

from tinygrad import Tensor

from ai.llm.tinyllm import GPT2, SimpleKVCache, generate
from ai.llm.tinyllm.utils import Tokenizer

model = GPT2.from_pretrained("gpt2-xl")
tokenizer = Tokenizer.for_model("gpt2-xl")

base = "The quick brown fox jumps over the lazy dog. "
long_text = base * 100

generate_tokens = 30
print(f"gpt2-xl — Generating {generate_tokens} new tokens from varying prefix lengths:\n")
print(f"{'prefix_len':>12} | {'no_cache tok/s':>16} | {'cache tok/s':>14} | {'speedup':>8} | correct")
print("-" * 75)

for target_prefix in [50, 200, 500, 900]:
    tokens = tokenizer.encode(long_text)[:target_prefix]
    input_ids = Tensor([tokens])

    _ = generate(model, input_ids, max_new_tokens=3)
    t0 = time.perf_counter()
    out_no = generate(model, input_ids, max_new_tokens=generate_tokens)
    t_no = time.perf_counter() - t0

    _ = generate(model, input_ids, max_new_tokens=3, kv_cache=SimpleKVCache())
    t0 = time.perf_counter()
    out_cache = generate(model, input_ids, max_new_tokens=generate_tokens, kv_cache=SimpleKVCache())
    t_cache = time.perf_counter() - t0

    match = out_no[0].numpy().tolist() == out_cache[0].numpy().tolist()
    print(
        f"{len(tokens):>12} | {generate_tokens / t_no:>16.1f} | {generate_tokens / t_cache:>14.1f} | {t_no / t_cache:>8.2f}x | {match}"
    )
