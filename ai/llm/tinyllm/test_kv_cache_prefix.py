import time

from tinygrad import Tensor

from ai.llm.tinyllm import GPT2, SimpleKVCache, generate
from ai.llm.tinyllm.utils import Tokenizer

model = GPT2.from_pretrained("gpt2")
tokenizer = Tokenizer.for_model("gpt2")

# Build a long prefix by repeating text until we hit target token count
base = "The quick brown fox jumps over the lazy dog. "
long_text = base * 100  # repeat to get plenty of tokens

# Test different prefix lengths with fixed short generation
generate_tokens = 30
print(f"Generating {generate_tokens} new tokens from varying prefix lengths:\n")
print(f"{'prefix_len':>12} | {'no_cache tok/s':>16} | {'cache tok/s':>14} | {'speedup':>8} | correct")
print("-" * 75)

for target_prefix in [50, 100, 200, 400, 700, 950]:
    tokens = tokenizer.encode(long_text)[:target_prefix]
    actual_len = len(tokens)
    input_ids = Tensor([tokens])

    # Warm up no-cache kernels first
    _ = generate(model, input_ids, max_new_tokens=3)

    t0 = time.perf_counter()
    out_no = generate(model, input_ids, max_new_tokens=generate_tokens)
    t_no = time.perf_counter() - t0

    cache = SimpleKVCache()
    # Warm up cache kernels
    _ = generate(model, input_ids, max_new_tokens=3, kv_cache=SimpleKVCache())

    t0 = time.perf_counter()
    out_cache = generate(model, input_ids, max_new_tokens=generate_tokens, kv_cache=cache)
    t_cache = time.perf_counter() - t0

    match = out_no[0].numpy().tolist() == out_cache[0].numpy().tolist()
    speedup = t_no / t_cache
    print(
        f"{actual_len:>12} | {generate_tokens / t_no:>16.1f} | {generate_tokens / t_cache:>14.1f} | {speedup:>8.2f}x | {match}"
    )
