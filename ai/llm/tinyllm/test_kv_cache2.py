import time

from tinygrad import Tensor

from ai.llm.tinyllm import GPT2, SimpleKVCache, generate
from ai.llm.tinyllm.utils import Tokenizer

prompt = "The meaning of life is"
max_tokens = 40

print("Loading model...")
model = GPT2.from_pretrained("gpt2")
tokenizer = Tokenizer.for_model("gpt2")
input_ids = Tensor([tokenizer.encode(prompt)])

for run in range(2):
    print(f"\n--- Run {run + 1} ---")

    t0 = time.perf_counter()
    out_no_cache = generate(model, input_ids, max_new_tokens=max_tokens)
    t1 = time.perf_counter()
    print(f"No cache:   {t1 - t0:.3f}s  {max_tokens / (t1 - t0):.1f} tok/s")

    cache = SimpleKVCache()
    t0 = time.perf_counter()
    out_with_cache = generate(model, input_ids, max_new_tokens=max_tokens, kv_cache=cache)
    t1 = time.perf_counter()
    print(f"With cache: {t1 - t0:.3f}s  {max_tokens / (t1 - t0):.1f} tok/s")

    match = out_no_cache[0].numpy().tolist() == out_with_cache[0].numpy().tolist()
    print(f"Outputs match: {match}")
