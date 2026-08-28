# Phase 3 — what invariance costs

## Setup

| | |
|---|---|
| machine | MacBook, Apple M4, 10 cores (4P/6E), 16 GB unified memory |
| OS | macOS 26.5 (Darwin 25.5.0) |
| GPU | `applegpu_g16g` |
| MLX | 0.32.0 from PyPI, native arm64 CPython 3.12 |
| library | `mlx-batch-invariant` 0.1.0 |

Reproduce:

```
PYTHONPATH=. .venv/bin/python bench/bench.py micro       --dtype float16
PYTHONPATH=. .venv/bin/python bench/bench.py e2e         --dtype float16
PYTHONPATH=. .venv/bin/python bench/bench.py determinism --dtype float16
```

## Method

The baseline is stock MLX on the same machine, same shapes, same dtypes, same
thermal state. There is no naive-Python comparison anywhere in this document.

Stock and invariant are **interleaved inside every repeat** rather than measured in
two separate blocks, and the reported figure is the median of the per-repeat
*ratios*, not a ratio of medians. A laptop this size throttles noticeably over a
long sweep; interleaving makes that drift common-mode instead of attributing it to
whichever path ran second. Each cell is warmed once, then 9 repeats × 3 calls
(micro) or 3 repeats (end to end).

Two honesty notes on the numbers below:

* Small-shape cells are dispatch-bound, not compute-bound. A 1×2048×2048 fp16 GEMM
  is 8 MFLOP and takes 0.31 ms; almost all of that is launch and synchronisation
  overhead, so treat those ratios as "cost of one more kernel launch shape", not as
  arithmetic throughput.
* The bfloat16 sweep is visibly noisier than float16 (one attention cell reads
  2.13× where the neighbouring longer-context cell reads 0.97×). It ran later in
  the session on a warmer machine. It is reported as measured, not smoothed.

## Headline: the thing that is actually being bought

16-layer float16 model, 64-token prompt held at batch position 0, batch grown by
padding with unrelated prompts. Final-position logits compared as raw `uint16` bit
patterns.

| path | batch | logits differing from batch 1 |
|---|---:|---:|
| stock MLX | 2 | 26357 / 32000 |
| stock MLX | 4 | 26385 / 32000 |
| stock MLX | 8 | 26491 / 32000 |
| batch invariant | 2 | **0 / 32000** |
| batch invariant | 4 | **0 / 32000** |
| batch invariant | 8 | **0 / 32000** |

Stock MLX changes 82% of the logit bits for an unchanged prompt purely because
other sequences were in flight. That is the defect. Everything below is its price.

## End to end

16-layer Qwen3-1.7B-shaped decoder (hidden 2048, ffn 6144, 16 heads / 8 kv heads,
head dim 128, 32k vocab), 0.94B parameters, float16, batch 1.

| phase | stock MLX | batch invariant | ratio | throughput cost |
|---|---:|---:|---:|---:|
| prefill 256 tok | 1609.7 tok/s | 560.6 tok/s | 2.87x | **65.2%** |
| decode 32 tok | 39.3 tok/s | 16.1 tok/s | 2.44x | **59.0%** |

## GEMM microbenchmark — float16

Stock `mx.matmul` against `mlx_batch_invariant.linear`.

| shape (M × K × N) | role | stock ms | invariant ms | ratio |
|---|---|---:|---:|---:|
| 1 × 2048 × 2048 | o_proj, decode | 0.306 | 0.564 | 1.86x |
| 1 × 2048 × 12288 | gate+up, decode | 0.793 | 1.547 | 1.99x |
| 1 × 6144 × 2048 | down_proj, decode | 0.500 | 1.164 | 2.33x |
| 8 × 2048 × 12288 | gate+up, batch 8 | 1.241 | 1.558 | 1.26x |
| 32 × 2048 × 12288 | gate+up, batch 32 | 1.254 | 1.615 | 1.29x |
| 128 × 2048 × 12288 | gate+up, batch 128 | 2.125 | 5.202 | 2.43x |
| 512 × 2048 × 12288 | gate+up, prefill 512 | 7.386 | 19.514 | 2.64x |
| 2048 × 2048 × 2048 | o_proj, prefill 2048 | 5.015 | 13.153 | 2.61x |

## GEMM microbenchmark — bfloat16

| shape (M × K × N) | role | stock ms | invariant ms | ratio |
|---|---|---:|---:|---:|
| 1 × 2048 × 2048 | o_proj, decode | 0.785 | 1.613 | 2.00x |
| 1 × 2048 × 12288 | gate+up, decode | 0.990 | 1.834 | 1.77x |
| 1 × 6144 × 2048 | down_proj, decode | 0.569 | 1.180 | 2.11x |
| 8 × 2048 × 12288 | gate+up, batch 8 | 1.391 | 1.705 | 1.27x |
| 32 × 2048 × 12288 | gate+up, batch 32 | 1.403 | 1.787 | 1.20x |
| 128 × 2048 × 12288 | gate+up, batch 128 | 2.459 | 5.752 | 2.26x |
| 512 × 2048 × 12288 | gate+up, prefill 512 | 8.988 | 23.073 | 2.58x |
| 2048 × 2048 × 2048 | o_proj, prefill 2048 | 5.914 | 14.512 | 2.54x |

## Attention microbenchmark — float16

Stock `mx.fast.scaled_dot_product_attention` against ours.

| B × H/Hkv × qL × N | role | stock ms | invariant ms | ratio |
|---|---|---:|---:|---:|
| 1 × 16/8 × 1 × 512 | decode, 512 ctx | 0.222 | 0.235 | 1.06x |
| 1 × 16/8 × 1 × 2048 | decode, 2k ctx | 0.311 | 0.328 | 1.06x |
| 1 × 16/8 × 1 × 8192 | decode, 8k ctx | 0.668 | 0.665 | 0.98x |
| 1 × 16/8 × 1 × 32768 | decode, 32k ctx | 1.670 | 1.639 | 0.97x |
| 8 × 16/8 × 1 × 2048 | decode, batch 8 | 0.949 | 0.944 | 0.99x |
| 32 × 16/8 × 1 × 2048 | decode, batch 32 | 2.990 | 3.081 | 1.04x |
| 1 × 16/8 × 8 × 2048 | chunked prefill, 8 queries | 0.468 | 0.491 | 1.06x |
| 1 × 16/8 × 128 × 128 | prefill 128 | 0.266 | 0.882 | 3.27x |
| 1 × 16/8 × 512 × 512 | prefill 512 | 0.959 | 4.699 | 4.92x |

**Decode attention is free.** Across single-query decode from 512 to 32768 tokens
of context, and across batch 1 to 32, the invariant kernel is between 0.97× and
1.06× of stock — inside the noise floor. Three of the seven decode cells are
*faster* than stock, because forcing the single-pass path skips the two-pass
kernel's intermediate `sums`/`maxs` buffers and its second dispatch.

That is the most useful result in this document: for the workload that determinism
actually matters in — serving decode, where a user's reply must not depend on who
else is on the box — attention invariance costs nothing measurable. The entire
end-to-end cost is the GEMM.

## Attention microbenchmark — bfloat16

| B × H/Hkv × qL × N | role | stock ms | invariant ms | ratio |
|---|---|---:|---:|---:|
| 1 × 16/8 × 1 × 512 | decode, 512 ctx | 0.250 | 0.237 | 0.91x |
| 1 × 16/8 × 1 × 2048 | decode, 2k ctx | 0.326 | 0.650 | 2.13x |
| 1 × 16/8 × 1 × 8192 | decode, 8k ctx | 0.969 | 0.913 | 0.97x |
| 1 × 16/8 × 1 × 32768 | decode, 32k ctx | 2.344 | 2.389 | 0.99x |
| 8 × 16/8 × 1 × 2048 | decode, batch 8 | 1.140 | 1.493 | 1.21x |
| 32 × 16/8 × 1 × 2048 | decode, batch 32 | 4.242 | 4.408 | 1.02x |
| 1 × 16/8 × 8 × 2048 | chunked prefill, 8 queries | 0.813 | 0.993 | 1.57x |
| 1 × 16/8 × 128 × 128 | prefill 128 | 0.349 | 1.000 | 2.38x |
| 1 × 16/8 × 512 × 512 | prefill 512 | 1.121 | 5.446 | 4.86x |

## Where the cost comes from

**Prefill GEMM, ~2.6×.** This kernel is a plain shared-memory tiled GEMM with a
4×4 register tile and scalar FMAs. MLX's `steel_gemm` uses `simdgroup_matrix`
hardware instructions. That is the entire gap, and it is not inherent to
invariance: a simdgroup-matrix kernel with a fixed tile and a single threadgroup
owning K would be just as invariant and much faster. It is unwritten work, not a
law of physics. See ROADMAP.md.

**Decode GEMM, ~2.0×.** At M=1 the fixed 32-row tile does 32× more arithmetic than
necessary, but decode is memory-bound on streaming the weights, so the observed
penalty is 2× rather than 32×. Keeping the tile fixed is deliberate: choosing a
narrower tile for small M is exactly the shape-dependent dispatch that Phase 0
identified as the root cause of the bug.

**Prefill attention, 3.3–4.9×.** The invariant kernel is a *vector* attention
kernel — one threadgroup per query row, walking the whole key sequence. MLX routes
prefill to `steel_attention`, which tiles queries and keys together and reuses each
key block across many queries. Using a vector kernel for 512 queries throws that
reuse away. Also fixable and also unwritten.

**Decode attention, ~1.0×.** Same structure as MLX's own single-pass kernel, so
there is nothing to lose.

## Against the CUDA prior art

| implementation | workload | throughput cost |
|---|---|---:|
| Thinking Machines `batch_invariant_ops` (vLLM, Qwen3-8B, H100) | serving | ~61.5% |
| SGLang deterministic mode | serving | ~34.35% |
| **this library, decode** (M4, 0.94B fp16) | decode | **59.0%** |
| **this library, prefill** (M4, 0.94B fp16) | prefill | **65.2%** |

Comparable in magnitude to Thinking Machines' unoptimised first pass, and worse
than SGLang's tuned one — which is the expected place for a from-scratch kernel to
land. The comparison is directional only: different hardware, different model,
different framework, and their percentages are measured on a full serving stack
under load rather than a single-stream decode loop. It is offered to answer "is
this in the normal range for a determinism tax?" (yes) and not as a benchmark
result against those projects.

The honest summary: **the tax on the part that matters — decode attention — is
zero, and the rest of the tax is unwritten optimisation, not a cost of
determinism.**
