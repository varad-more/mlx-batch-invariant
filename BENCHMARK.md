# Phase 3 — what invariance costs

## Setup

| | |
|---|---|
| machine | MacBook, Apple M4, 10 cores (4P/6E), 16 GB unified memory |
| OS | macOS 26.5 (Darwin 25.5.0) |
| GPU | `applegpu_g16g` |
| MLX | 0.32.0 from PyPI, native arm64 CPython 3.12 |
| library | `mlx-batch-invariant` 0.2.0 |

Reproduce:

```
.venv/bin/python bench/bench.py all   --dtype float16  --json bench/all_f16.json
.venv/bin/python bench/bench.py micro --dtype bfloat16 --json bench/micro_bf16.json
.venv/bin/python bench/real_model.py
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

One honesty note: small-shape cells are dispatch-bound, not compute-bound. A
1×2048×2048 fp16 GEMM is 8 MFLOP and takes 0.32 ms; almost all of that is launch
and synchronisation overhead, so treat those ratios as "cost of one more kernel
launch shape", not as arithmetic throughput.

## Headline: the thing that is actually being bought

16-layer float16 model, 64-token prompt held at batch position 0, batch grown by
padding with unrelated prompts. Final-position logits compared as raw `uint16` bit
patterns.

| path | batch | logits differing from batch 1 |
|---|---:|---:|
| stock MLX | 2 | 25812 / 32000 |
| stock MLX | 4 | 26399 / 32000 |
| stock MLX | 8 | 26629 / 32000 |
| batch invariant | 2 | **0 / 32000** |
| batch invariant | 4 | **0 / 32000** |
| batch invariant | 8 | **0 / 32000** |

Stock MLX changes 82% of the logit bits for an unchanged prompt purely because
other sequences were in flight. That is the defect. Everything below is its price.

The same measurement on a real 4-bit checkpoint rather than a synthetic one
(`bench/real_model.py`, mlx-community/Qwen1.5-0.5B-Chat-4bit, 7-token prompt):

| path | batch 2 | batch 4 | batch 8 |
|---|---:|---:|---:|
| stock MLX | 113360 / 151936 | 121209 / 151936 | 117380 / 151936 |
| batch invariant | **0 / 151936** | **0 / 151936** | **0 / 151936** |

Under `strict=True` nothing in that model raised, and generation from the invariant
path is coherent — so the kernels are correct on trained weights, not just
invariant on random ones.

## End to end

16-layer Qwen3-1.7B-shaped decoder (hidden 2048, ffn 6144, 16 heads / 8 kv heads,
head dim 128, 32k vocab), 0.94B parameters, float16, batch 1.

| phase | stock MLX | batch invariant | ratio | throughput cost |
|---|---:|---:|---:|---:|
| prefill 256 tok | 1982.4 tok/s | 913.6 tok/s | 2.17x | **53.9%** |
| decode 32 tok | 53.6 tok/s | 27.0 tok/s | 1.99x | **49.7%** |

v0.1, on the same harness, cost 65.2% of prefill and 59.0% of decode. The
difference is the `simdgroup_matrix` GEMM (ROADMAP item 1) and the fused quantized
kernel (item 3), both landed in v0.2.

## GEMM microbenchmark — float16

Stock `mx.matmul` against `mlx_batch_invariant.linear`.

| shape (M × K × N) | role | stock ms | invariant ms | ratio | v0.1 |
|---|---|---:|---:|---:|---:|
| 1 × 2048 × 2048 | o_proj, decode | 0.322 | 0.659 | 1.83x | 1.86x |
| 1 × 2048 × 12288 | gate+up, decode | 0.809 | 1.294 | 1.60x | 1.99x |
| 1 × 6144 × 2048 | down_proj, decode | 0.496 | 0.884 | 1.73x | 2.33x |
| 8 × 2048 × 12288 | gate+up, batch 8 | 1.253 | 1.243 | **0.99x** | 1.26x |
| 32 × 2048 × 12288 | gate+up, batch 32 | 1.300 | 1.305 | **1.00x** | 1.29x |
| 128 × 2048 × 12288 | gate+up, batch 128 | 2.142 | 3.908 | 1.82x | 2.43x |
| 512 × 2048 × 12288 | gate+up, prefill 512 | 7.361 | 14.516 | 1.97x | 2.64x |
| 2048 × 2048 × 2048 | o_proj, prefill 2048 | 5.018 | 9.875 | 1.97x | 2.61x |

At batch 8 and 32 the invariant GEMM is now **at parity with stock MLX** — the
small-batch region where MLX's own dispatch is deciding between `gemv` and
`steel_gemm` is precisely where a single fixed kernel gives up nothing.

## GEMM microbenchmark — bfloat16

| shape (M × K × N) | role | stock ms | invariant ms | ratio |
|---|---|---:|---:|---:|
| 1 × 2048 × 2048 | o_proj, decode | 0.478 | 1.095 | 2.34x |
| 1 × 2048 × 12288 | gate+up, decode | 0.883 | 1.753 | 1.90x |
| 1 × 6144 × 2048 | down_proj, decode | 0.495 | 0.968 | 1.99x |
| 8 × 2048 × 12288 | gate+up, batch 8 | 1.278 | 1.228 | 0.97x |
| 32 × 2048 × 12288 | gate+up, batch 32 | 1.277 | 1.297 | 1.01x |
| 128 × 2048 × 12288 | gate+up, batch 128 | 2.135 | 3.839 | 1.80x |
| 512 × 2048 × 12288 | gate+up, prefill 512 | 7.355 | 14.172 | 1.93x |
| 2048 × 2048 × 2048 | o_proj, prefill 2048 | 5.017 | 9.634 | 1.92x |

## Quantized linear — float16 activations, int4 weights

Stock fused `quantized_matmul` against the fused invariant quantized kernel.

| shape (M × K × N) | role | stock ms | invariant ms | ratio | v0.1 |
|---|---|---:|---:|---:|---:|
| 1 × 2048 × 2048 | o_proj, decode | 0.225 | 0.467 | 2.06x | 3.33x |
| 1 × 2048 × 12288 | gate+up, decode | 0.382 | 1.504 | 3.90x | 6.89x |
| 1 × 6144 × 2048 | down_proj, decode | 0.298 | 1.230 | 4.12x | 5.40x |
| 256 × 2048 × 2048 | o_proj, prefill | 0.933 | 1.949 | 2.12x | 2.70x |
| 256 × 2048 × 12288 | gate+up, prefill | 3.906 | 9.796 | 2.51x | 2.79x |

v0.1 dequantised the whole weight to fp16 and then ran the float GEMM over it, so a
single decode row paid to materialise an entire matrix. v0.2 dequantises inside the
B-tile load, in the K loop. The invariance argument is identical either way — a
group's scale and bias are per-weight, not per-batch — so this was always a memory
cost rather than a cost of determinism, and it is now roughly halved.

What remains is the fixed 32-row tile: at M=1 the kernel dequantises a 32-row-wide
slab of weight to use one row of it. That is deliberate, and it is the same
trade-off as the decode GEMM below.

## Attention microbenchmark — float16

Stock `mx.fast.scaled_dot_product_attention` against ours.

| B × H/Hkv × qL × N | role | stock ms | invariant ms | ratio |
|---|---|---:|---:|---:|
| 1 × 16/8 × 1 × 512 | decode, 512 ctx | 0.235 | 0.245 | 1.04x |
| 1 × 16/8 × 1 × 2048 | decode, 2k ctx | 0.306 | 0.321 | 1.05x |
| 1 × 16/8 × 1 × 8192 | decode, 8k ctx | 0.642 | 0.635 | 1.02x |
| 1 × 16/8 × 1 × 32768 | decode, 32k ctx | 1.619 | 1.623 | 0.99x |
| 8 × 16/8 × 1 × 2048 | decode, batch 8 | 0.926 | 0.938 | 1.01x |
| 32 × 16/8 × 1 × 2048 | decode, batch 32 | 2.760 | 2.770 | 1.00x |
| 1 × 16/8 × 8 × 2048 | chunked prefill, 8 queries | 0.465 | 0.483 | 1.05x |
| 1 × 16/8 × 128 × 128 | prefill 128 | 0.284 | 0.845 | 2.95x |
| 1 × 16/8 × 512 × 512 | prefill 512 | 0.990 | 4.655 | 4.71x |

**Decode attention is free.** Across single-query decode from 512 to 32768 tokens
of context, and across batch 1 to 32, the invariant kernel is between 0.99× and
1.05× of stock — inside the noise floor.

That is the most useful result in this document: for the workload that determinism
actually matters in — serving decode, where a user's reply must not depend on who
else is on the box — attention invariance costs nothing measurable.

Attention sinks are supported as of v0.2 and are bitwise identical to stock MLX's
single-pass path, so the numbers above apply unchanged to sink models.

## Attention microbenchmark — bfloat16

| B × H/Hkv × qL × N | role | stock ms | invariant ms | ratio |
|---|---|---:|---:|---:|
| 1 × 16/8 × 1 × 512 | decode, 512 ctx | 0.233 | 0.243 | 1.05x |
| 1 × 16/8 × 1 × 2048 | decode, 2k ctx | 0.297 | 0.314 | 1.05x |
| 1 × 16/8 × 1 × 8192 | decode, 8k ctx | 0.663 | 0.623 | 0.93x |
| 1 × 16/8 × 1 × 32768 | decode, 32k ctx | 1.639 | 1.635 | 1.00x |
| 8 × 16/8 × 1 × 2048 | decode, batch 8 | 0.945 | 0.939 | 1.01x |
| 32 × 16/8 × 1 × 2048 | decode, batch 32 | 2.757 | 2.774 | 1.01x |
| 1 × 16/8 × 8 × 2048 | chunked prefill, 8 queries | 0.466 | 0.485 | 1.04x |
| 1 × 16/8 × 128 × 128 | prefill 128 | 0.276 | 0.839 | 3.04x |
| 1 × 16/8 × 512 × 512 | prefill 512 | 0.980 | 4.657 | 4.78x |

The bfloat16 sweep now tracks float16 to within a few percent everywhere except the
smallest decode cell. v0.1's bfloat16 table had a visibly noisy outlier (2.13× in a
cell whose neighbours read 0.97×); that was thermal, and re-running on a cold
machine removed it.

## Where the cost comes from

**Prefill GEMM, ~1.95×.** The kernel is now `simdgroup_multiply_accumulate` over
8×8 fragments with a fixed 32×32×16 tile, four simdgroups to a threadgroup. At
512×4096×4096 fp16 that is 1.7 TFLOP/s against stock's 3.5.

The remaining gap is *not* tile size, which was the obvious suspect. Sweeping the
tile as a fixed compile-time constant (so every candidate is equally invariant):

| tile | decode 1×4096×4096 | prefill 512×4096×4096 |
|---|---:|---:|
| 32×32×16, 2×2 simdgroups | **1.66x** | **2.01x** |
| 64×32×16 | 2.47x | — |
| 64×64×16 | 3.02x | 2.50x |
| 64×64×32 | 3.85x | — |

32×32×16 wins on both ends, so there is no prefill/decode tension to resolve here
and nothing to gain by dispatching on M. What is left is that the tile loop has no
double buffering: every threadgroup stalls on its global loads before each MMA
block instead of prefetching the next tile behind the current one. That is the next
lever, and it is unwritten. See ROADMAP.md.

One measurement worth keeping: the `unroll(full)` pragmas on the fragment loops are
load-bearing, not decoration. Without them the compiler spills the 8×8 accumulators
to thread memory and the 64×64 tile goes from 2.86× to 11× — worse than the scalar
FMA kernel it replaced.

**Decode GEMM, 1.6–1.8×.** At M=1 the fixed 32-row tile does 32× more arithmetic
than necessary, but decode is memory-bound on streaming the weights, so the
observed penalty is under 2× rather than 32×.

**Prefill attention, 3.0–4.8×.** The invariant kernel is a *vector* attention
kernel: one threadgroup per query row, walking the whole key sequence. It therefore
re-reads K and V once per query row. Measured, on 1×32/8×512×4096 fp16:

| | K/V bytes issued | time | effective load rate |
|---|---:|---:|---:|
| this library | 34.4 GB | 53.5 ms | 642 GB/s |
| `steel_attention` | 0.017 GB | 10.6 ms | — |

We are already saturating the cache hierarchy at 642 GB/s; the 5× gap is entirely
redundant traffic, not slow arithmetic. See ROADMAP.md for why the obvious fix
(blocking query rows together) was measured and rejected.

**Decode attention, ~1.0×.** Same structure as MLX's own single-pass kernel, so
there is nothing to lose.

## Against the CUDA prior art

| implementation | workload | throughput cost |
|---|---|---:|
| Thinking Machines `batch_invariant_ops` (vLLM, Qwen3-8B, H100) | serving | ~61.5% |
| SGLang deterministic mode | serving | ~34.35% |
| **this library, decode** (M4, 0.94B fp16) | decode | **49.7%** |
| **this library, prefill** (M4, 0.94B fp16) | prefill | **53.9%** |

Now between Thinking Machines' unoptimised first pass and SGLang's tuned one, where
v0.1 was worse than both. The comparison is directional only: different hardware,
different model, different framework, and their percentages are measured on a full
serving stack under load rather than a single-stream decode loop. It is offered to
answer "is this in the normal range for a determinism tax?" (yes) and not as a
benchmark result against those projects.

The honest summary: **the tax on the part that matters — decode attention — is
zero, small-batch GEMM is now free too, and what remains is the price of refusing
to pick a kernel based on the shape of the batch.**
