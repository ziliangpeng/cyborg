"""Test KV cache correctness and speedup for GPT-2."""
import time
from tinygrad import Tensor
from ai.llm.tinyllm import GPT2, SimpleKVCache, generate
from ai.llm.tinyllm.utils import Tokenizer

def test_kv_cache_correctness_and_speedup():
    prompt = "The meaning of life is"
    max_tokens = 40

    print("Loading model...")
    model = GPT2.from_pretrained("gpt2")
    tokenizer = Tokenizer.for_model("gpt2")
    input_ids = Tensor([tokenizer.encode(prompt)])

    # Without KV cache
    print("\n=== WITHOUT KV cache ===")
    t0 = time.perf_counter()
    out_no_cache = generate(model, input_ids, max_new_tokens=max_tokens)
    time_no_cache = time.perf_counter() - t0
    text_no_cache = tokenizer.decode(out_no_cache[0].numpy().tolist())
    print(text_no_cache)
    print(f"Time: {time_no_cache:.3f}s  |  Tok/s: {max_tokens / time_no_cache:.1f}")

    # With KV cache
    print("\n=== WITH KV cache ===")
    cache = SimpleKVCache()
    t0 = time.perf_counter()
    out_with_cache = generate(model, input_ids, max_new_tokens=max_tokens, kv_cache=cache)
    time_with_cache = time.perf_counter() - t0
    text_with_cache = tokenizer.decode(out_with_cache[0].numpy().tolist())
    print(text_with_cache)
    print(f"Time: {time_with_cache:.3f}s  |  Tok/s: {max_tokens / time_with_cache:.1f}")

    # Correctness
    print("\n=== CORRECTNESS ===")
    match = out_no_cache[0].numpy().tolist() == out_with_cache[0].numpy().tolist()
    print(f"Outputs match: {match}")
    assert match, "KV cache output does not match non-cached output!"

    print(f"\nSpeedup: {time_no_cache / time_with_cache:.2f}x")

if __name__ == "__main__":
    test_kv_cache_correctness_and_speedup()
