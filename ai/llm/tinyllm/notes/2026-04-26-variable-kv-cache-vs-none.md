# VariableKVCache vs No-Cache Benchmark

Date: 2026-04-26
Hardware: 1x H100 (gcp5-h100-0-43), tinygrad CUDA backend
Model: gpt2 (124M params)
Method: bazel run //ai/llm/tinyllm:benchmark, warmup=2, runs=3
Input length: 500 tokens (fixed). Output lengths: 100 / 200 / 400.

## Headline

VariableKVCache delivers ~6.5-7.7x speedup on TPOT (per-decoded-token cost) at
input=500, with stable ~10 ms/tok throughput regardless of output length.
Without a KV cache, every decode step re-runs the full 500+ context prefill,
so TPOT grows with sequence length and the model is bottlenecked at ~14 tok/s.

## Results

input=500, output=100
                  none             variable        speedup
  tok/s            15.4             96.9            6.3x
  TTFT (ms)        57.9             72.9            -
  TPOT (ms/tok)    65.2              9.7            6.7x

input=500, output=200
                  none             variable        speedup
  tok/s            13.7            101.0            7.4x
  TTFT (ms)        54.9             94.3 (med 76)   -
  TPOT (ms/tok)    73.0              9.5            7.7x

input=500, output=400
                  none             variable        speedup
  tok/s            13.7             90.4            6.6x
  TTFT (ms)        52.3             94.5 (med 74)   -
  TPOT (ms/tok)    73.4             10.9            6.7x

## Reading

- TPOT is the right metric here — it isolates per-step decode cost.
- variable TPOT is nearly flat at 9.5-10.9 ms even as the cached context grows
  from 600 to 900 tokens, confirming the symbolic-length JIT replay works:
  the attention kernel compiles once and reruns with a different bound
  start_pos each step. Slight upward drift comes from the actual K/V slice
  growing inside the (still-fixed) max_seq_len buffer.
- variable TTFT (~73 ms median) is slightly higher than none (~55 ms). Extra
  cost vs no-cache prefill: writing K/V into the preallocated buffer + one
  flush() to realize all assigns. Acceptable since prefill happens once.
- Mean TTFT for variable shows occasional outliers (135-142 ms) — likely
  one-shot JIT verification/finalization cost on the very first
  post-warmup invocation. Median is the cleaner number.
- none-mode TPOT (~65-73 ms) is far worse than its TTFT (~55 ms) because
  each decode step re-does the full prefill compute AND tinygrad has to
  re-trace at a new sequence length each step (no JIT reuse without a cache).

## Reproduce

  ssh gcp5
  salloc --partition=general --nodes=1 --gres=gpu:8 --exclusive \
    --nodelist=gcp5-h100-0-43 --time=6:00:00 -J tinyllm
  srun --jobid=<id> --gres=gpu:1 bash -lc '
    export PATH=/usr/local/cuda/bin:$PATH; export CUDA=1
    cd ~/cyborg
    bazel run //ai/llm/tinyllm:benchmark -- \
      --model gpt2 --kv-cache variable --warmup 2 --runs 3 \
      --input-lens 500 --output-lens 100,200,400'

  Repeat with --kv-cache none for the comparison run.

## Notes / pitfalls

- Avoid node gcp5-h100-0-17: CUDA error 802 "system not yet initialized"
  (fabricmanager not running). Node 43 was healthy.
- nvidia-smi is not on PATH inside srun shells; use /usr/local/cuda/bin/nvcc
  for tools and rely on tinygrad's CUDA backend (CUDA=1 env).
- VariableKVCache JIT compiles on call 1 and verifies on call 2 — benchmark
  harness already enforces warmup>=2 for this mode.

## Next

- Repeat on gpt2-xl (1.5B) and llama-3.2-1B to see absolute throughput at
  larger model scale where the per-step compute dominates.
- Sweep input length (128, 500, 1024, 2048) at fixed output to characterize
  TPOT-vs-context-length curve.
- Consider also benchmarking "simple" mode for completeness (cat-based,
  expected to be much worse than variable due to per-step recompile).
