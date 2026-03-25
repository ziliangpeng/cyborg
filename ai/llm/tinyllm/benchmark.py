"""TinyLLM inference benchmark.

Runs warmup passes to prime the JIT, then measures inference performance
across a configurable grid of input/output lengths. Reports distribution
statistics (mean, median, p95, min, max) for tok/s, TTFT, and TPOT.

Usage:
    bazel run //ai/llm/tinyllm:benchmark -- \
        --model gpt2 \
        --warmup 3 \
        --runs 5 \
        --input-lens 32,128,512 \
        --output-lens 32,128
"""

import statistics
import time

import click
from tinygrad import Tensor

from ai.llm.tinyllm import generate, SimpleKVCache
from ai.llm.tinyllm.kv_cache import VariableKVCache
from ai.llm.tinyllm.utils import Tokenizer
from ai.llm.tinyllm.cli import load_model, format_param_count

_BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells sea shells by the sea shore. "
    "How much wood would a woodchuck chuck. "
)


def _make_prompt(tokenizer, target_len: int) -> Tensor:
    tokens = (tokenizer.encode(_BASE_TEXT) * (target_len // 10 + 2))[:target_len]
    return Tensor([tokens])


def _make_kv_cache(model, mode: str):
    """Factory: create a KV cache instance based on mode string."""
    if mode == "none":
        return None
    if mode == "simple":
        return SimpleKVCache()
    if mode == "variable":
        cfg = model.config
        head_dim = cfg.n_embd // cfg.n_head
        return VariableKVCache(
            max_seq_len=cfg.n_positions,
            n_layers=cfg.n_layer,
            num_heads=cfg.n_head,
            head_dim=head_dim,
        )
    raise ValueError(f"Unknown kv_cache mode: {mode}")


def _run_once(model, input_ids: Tensor, output_len: int, temperature: float, top_k, use_kv_cache: str = "none") -> dict:
    if input_ids.shape[1] > model.config.n_positions:
        input_ids = input_ids[:, -model.config.n_positions:]

    # Run 1: measure TTFT only (1 token, fresh cache)
    kv_cache = _make_kv_cache(model, use_kv_cache)
    t0 = time.perf_counter()
    generate(model, input_ids, max_new_tokens=1,
             temperature=temperature, top_k=top_k, kv_cache=kv_cache)
    ttft = time.perf_counter() - t0

    if output_len <= 1:
        return {"ttft": ttft, "tpot": 0.0, "total": ttft, "output_tokens": 1}

    # Run 2: measure total time for full output (fresh cache each time)
    kv_cache2 = _make_kv_cache(model, use_kv_cache)
    t1 = time.perf_counter()
    generate(model, input_ids, max_new_tokens=output_len,
             temperature=temperature, top_k=top_k, kv_cache=kv_cache2)
    total = time.perf_counter() - t1

    tpot = (total - ttft) / (output_len - 1)
    return {"ttft": ttft, "tpot": tpot, "total": total, "output_tokens": output_len}


def _stats(values):
    s = sorted(values)
    n = len(s)
    return {
        "mean":   statistics.mean(s),
        "median": statistics.median(s),
        "min":    s[0],
        "max":    s[-1],
        "p95":    s[max(0, int(n * 0.95) - 1)],
    }


def _fmt(label, values, unit):
    st = _stats(values)
    return (f"  {label:<8}"
            f"  mean={st['mean']:7.1f}"
            f"  median={st['median']:7.1f}"
            f"  p95={st['p95']:7.1f}"
            f"  min={st['min']:7.1f}"
            f"  max={st['max']:7.1f}"
            f"  {unit}")


@click.command()
@click.option("--model", default="gpt2", help="Model name")
@click.option("--warmup", default=3, help="Warmup runs per configuration")
@click.option("--runs", default=5, help="Timed runs per configuration")
@click.option("--input-lens", default="32,128,512", help="Comma-separated input lengths")
@click.option("--output-lens", default="32,128", help="Comma-separated output lengths")
@click.option("--temperature", default=1.0, help="Sampling temperature")
@click.option("--top-k", default=None, type=int, help="Top-k (None = greedy)")
@click.option("--kv-cache", type=click.Choice(["none", "simple", "variable"]), default="none", help="KV cache implementation: none, simple, variable")
def main(model, warmup, runs, input_lens, output_lens, temperature, top_k, kv_cache):
    """TinyLLM inference benchmark with JIT warmup and distribution stats."""
    in_lens  = [int(x) for x in input_lens.split(",")]
    out_lens = [int(x) for x in output_lens.split(",")]

    click.echo(f"Loading model: {model}")
    t0 = time.perf_counter()
    llm = load_model(model)
    tokenizer = Tokenizer.for_model(model)
    click.echo(f"Model loaded in {time.perf_counter()-t0:.2f}s "
               f"({format_param_count(llm.param_count())} params)")
    click.echo(f"warmup={warmup}  runs={runs}  kv_cache={kv_cache}  "
               f"input_lens={in_lens}  output_lens={out_lens}")
    click.echo()

    configs = [(i, o) for i in in_lens for o in out_lens
               if i + o <= llm.config.n_positions]

    # VariableKVCache uses TinyJit: first call = compile, second = verify.
    # Enforce at least 2 warmup passes so JIT is fully compiled before timing.
    effective_warmup = warmup
    if kv_cache == "variable" and warmup < 2:
        effective_warmup = 2
        click.echo(f"Note: bumping warmup from {warmup} to {effective_warmup} (VariableKVCache JIT needs >=2)")

    if effective_warmup > 0:
        jit_note = " (priming TinyJit)" if kv_cache == "variable" else ""
        click.echo(f"Warming up ({effective_warmup} pass(es) x {len(configs)} configs){jit_note}...")
        # For JIT warmup, only need a few decode steps to compile the kernel.
        # Using full out_len would waste time; 5 tokens is enough to prime JIT.
        for _ in range(effective_warmup):
            for in_len, out_len in configs:
                warmup_tokens = 5 if kv_cache == "variable" else out_len
                ids = _make_prompt(tokenizer, in_len)
                generate(llm, ids, max_new_tokens=warmup_tokens,
                         temperature=temperature, top_k=top_k,
                         kv_cache=_make_kv_cache(llm, kv_cache))
        click.echo("Warmup complete.\n")
    click.echo(f"  {'input':>6}  {'output':>6}  metric    "
               f"{'mean':>8}  {'median':>8}  {'p95':>8}  {'min':>8}  {'max':>8}  unit")
    click.echo("-" * 80)

    for in_len, out_len in configs:
        ids = _make_prompt(tokenizer, in_len)
        actual_in = ids.shape[1]

        results = [_run_once(llm, ids, out_len, temperature, top_k, use_kv_cache=kv_cache)
                   for _ in range(runs)]

        ttfts  = [r["ttft"] * 1000 for r in results]
        tpots  = [r["tpot"] * 1000 for r in results]
        tokpss = [r["output_tokens"] / r["total"] for r in results]

        click.echo(f"  input={actual_in:4d}  output={out_len:4d}:")
        click.echo(_fmt("tok/s",  tokpss, "tok/s"))
        click.echo(_fmt("TTFT",   ttfts,  "ms"))
        if out_len > 1:
            click.echo(_fmt("TPOT",   tpots,  "ms/tok"))
        click.echo()


if __name__ == "__main__":
    main()
