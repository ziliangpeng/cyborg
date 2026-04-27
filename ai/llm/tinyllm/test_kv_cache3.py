import time

from tinygrad import Tensor

from ai.llm.tinyllm import GPT2, SimpleKVCache, generate
from ai.llm.tinyllm.utils import Tokenizer

model = GPT2.from_pretrained("gpt2")
tokenizer = Tokenizer.for_model("gpt2")
input_ids = Tensor([tokenizer.encode("The meaning of life is")])

for max_tokens in [20, 50, 100, 200]:
    t0 = time.perf_counter()
    out = generate(model, input_ids, max_new_tokens=max_tokens)
    t_no = time.perf_counter() - t0

    cache = SimpleKVCache()
    t0 = time.perf_counter()
    out_c = generate(model, input_ids, max_new_tokens=max_tokens, kv_cache=cache)
    t_cache = time.perf_counter() - t0

    match = out[0].numpy().tolist() == out_c[0].numpy().tolist()
    print(
        f"tokens={max_tokens:3d}: no_cache={t_no:.2f}s ({max_tokens / t_no:.1f} tok/s)  cache={t_cache:.2f}s ({max_tokens / t_cache:.1f} tok/s)  speedup={t_no / t_cache:.2f}x  match={match}"
    )
