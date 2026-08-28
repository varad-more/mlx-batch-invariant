# Phase 0 — Kill test

**Verdict: the project survives, but it is much narrower than assumed. Matmul is the whole story.**

## Setup

| | |
|---|---|
| Machine | MacBook Pro, Apple M4 (10-core, 4P/6E), 16 GB unified memory |
| GPU architecture | `applegpu_g16g` — `devc == 'g'`, `arch_gen == 16` |
| OS | macOS 26.5 (25F71) |
| MLX | 0.32.0 (PyPI wheel), Python 3.12.12 arm64 |
| Neural Accelerator | `is_nax_available() == false` (requires `arch_gen >= 17`). Every `*_nax` kernel path is unreachable on this machine. |

Reproduce:

```
uv venv --python 3.12 && uv pip install mlx==0.32.0 numpy
.venv/bin/python probe/probe.py selftest                 # comparator positive control
.venv/bin/python probe/probe.py run                      # full sweep -> probe/results.jsonl
.venv/bin/python probe/probe.py summarize                # the matrix below
.venv/bin/python probe/probe.py splitkv                  # split-KV probe
.venv/bin/python probe/probe.py epilogue                 # addmm epilogue diagnostic
```

## Verdict

On this M4, MLX is batch-variant in exactly one place: `mx.matmul` and `mx.addmm`. Divergence
appears at **batch 2** — the smallest possible step — in every dtype and at every shape tested,
because batch 1 routes to a `gemv` kernel and batch 2 routes to a `steel_gemm` or
`steel_gemm_splitk` kernel with a completely different reduction structure. Further step changes
occur as the batch crosses the split-K partition-count boundaries, which are an explicit function
of M. In float32 this moves 92–99% of the output bits with a median of 5–22 ULP; in float16 and
bfloat16 the output rounding masks most of it, leaving 0.1–1% of bits differing by 1 ULP — small
per operation, but non-zero, which is all that matters for token divergence over a long generation.
Everything else measured — `mx.fast.rms_norm`, `mx.fast.layer_norm`, `mx.softmax`, `mx.sum`,
`mx.mean`, and `mx.fast.scaled_dot_product_attention` — is **bitwise batch-invariant across the
full sweep in all three dtypes**, including across kernel-strategy switches that do change which
Metal kernel runs. That is a genuine negative result and it removes RMSNorm from Phase 1 and,
as scoped, most of Phase 2. Two hazards adjacent to batch invariance did show up and are
documented below: a double-rounding difference in the fused `addmm` epilogue, and split-KV
attention whose result depends on the KV split count (a function of context length, not batch).

## Method

Fixed reference row, generated once per case from a fixed key and reused unchanged. For each batch
size in `1, 2, 3, 5, 8, 16, 31, 32, 33, 64, 128` the reference row is placed at index 0 and at
index `B//2` inside a batch of otherwise-random rows; the op runs; that row's output is sliced,
made contiguous, reinterpreted with `mx.view` as `uint32`/`uint16`, and compared bit-for-bit
against the batch-1 result. `allclose`, `mx.allclose` and float `==` appear nowhere in the
comparison path. Divergence magnitude is IEEE754 ULP distance computed on the integer ordinals,
reported as median and max over the differing elements.

Each case runs in three evaluation modes — `eager` (`mx.eval` immediately), `graph` (the op left
inside a larger graph at eval time), and `compiled` (`mx.compile`) — each compared against its own
batch-1 baseline, with the baselines also cross-compared.

Axes: dtype `float32 / float16 / bfloat16` swept independently; matmul `K ∈ {512, 1024, 2048,
4096, 11008} × N ∈ {1024, 4096}`; norms and softmax `D ∈ {1024, 4096, 8192}` (crossing the 4096
looped-kernel limit); reductions `R ∈ {64, 128, 1024, 4096}` (crossing the 64-element small-row
limit and the 32-row kernel-switch limit); attention `L ∈ {128, 512, 2048, 4096, 8192}` at `D=128`
for both MHA (`H=Hkv=8`) and GQA (`H=8, Hkv=2`).

The `predicted kernels` column is not measured — it is the dispatch condition transcribed from
mlx 0.32.0 C++ and evaluated in Python for this device. Where it changes, the measured divergence
changes with it, which is the main evidence that the source reading is correct.

**Comparator validation.** `probe.py selftest` asserts the pipeline reports exactly 1 differing
element at exactly 1 ULP for a 1-ULP perturbation in each dtype, that ±0.0 compare as 0 ULP, and
that a known-variant matmul is reported as variant. It passes.

**Skipped cells.** 5 of 8858 cells were skipped by the 3 GB input-memory guard, all MHA attention
at `L ≥ 4096` with `B ≥ 64` in float32/float16/bfloat16. They are marked `skipped` in the data,
not silently dropped.

## Matrix

| op | dtype | shape | verdict | first div B | bits differ | med ULP | max ULP | predicted kernels |
|---|---|---|---|---|---|---|---|---|
| addmm | bfloat16 | K=1024 N=1024 | **VARIANT** | 2 | 26.5% | 1 | 8 | gemv, splitk[2] |
| addmm | bfloat16 | K=1024 N=4096 | **VARIANT** | 2 | 26.3% | 1 | 15 | gemv, steel |
| addmm | bfloat16 | K=11008 N=1024 | **VARIANT** | 2 | 28.0% | 1 | 3 | gemv, splitk[16], splitk[32], splitk[8] |
| addmm | bfloat16 | K=11008 N=4096 | **VARIANT** | 2 | 25.5% | 1 | 7 | gemv, splitk[2], splitk[8], steel |
| addmm | bfloat16 | K=2048 N=1024 | **VARIANT** | 2 | 25.5% | 1 | 15072 | gemv, splitk[2], splitk[4] |
| addmm | bfloat16 | K=2048 N=4096 | **VARIANT** | 2 | 25.7% | 1 | 42 | gemv, steel |
| addmm | bfloat16 | K=4096 N=1024 | **VARIANT** | 2 | 26.9% | 1 | 4 | gemv, splitk[2], splitk[4], splitk[8] |
| addmm | bfloat16 | K=4096 N=4096 | **VARIANT** | 2 | 25.7% | 1 | 26 | gemv, splitk[2], steel |
| addmm | bfloat16 | K=512 N=1024 | **VARIANT** | 2 | 25.1% | 1 | 15089 | gemv, steel |
| addmm | bfloat16 | K=512 N=4096 | **VARIANT** | 2 | 26.3% | 1 | 125 | gemv, steel |
| addmm | float16 | K=1024 N=1024 | **VARIANT** | 2 | 24.3% | 1 | 12 | gemv, splitk[2] |
| addmm | float16 | K=1024 N=4096 | **VARIANT** | 2 | 26.3% | 1 | 130 | gemv, steel |
| addmm | float16 | K=11008 N=1024 | **VARIANT** | 2 | 27.7% | 1 | 6 | gemv, splitk[16], splitk[32], splitk[8] |
| addmm | float16 | K=11008 N=4096 | **VARIANT** | 2 | 25.6% | 1 | 25 | gemv, splitk[2], splitk[8], steel |
| addmm | float16 | K=2048 N=1024 | **VARIANT** | 2 | 27.5% | 1 | 8 | gemv, splitk[2], splitk[4] |
| addmm | float16 | K=2048 N=4096 | **VARIANT** | 2 | 25.6% | 1 | 48 | gemv, steel |
| addmm | float16 | K=4096 N=1024 | **VARIANT** | 2 | 27.1% | 1 | 11 | gemv, splitk[2], splitk[4], splitk[8] |
| addmm | float16 | K=4096 N=4096 | **VARIANT** | 2 | 25.6% | 1 | 56 | gemv, splitk[2], steel |
| addmm | float16 | K=512 N=1024 | **VARIANT** | 2 | 23.2% | 1 | 12 | gemv, steel |
| addmm | float16 | K=512 N=4096 | **VARIANT** | 2 | 26.4% | 1 | 31 | gemv, steel |
| addmm | float32 | K=1024 N=1024 | **VARIANT** | 2 | 93.2% | 5 | 4000 | gemv, splitk[2] |
| addmm | float32 | K=1024 N=4096 | **VARIANT** | 2 | 95.0% | 7 | 41104 | gemv, steel |
| addmm | float32 | K=11008 N=1024 | **VARIANT** | 2 | 96.3% | 11 | 14592 | gemv, splitk[16], splitk[32], splitk[8] |
| addmm | float32 | K=11008 N=4096 | **VARIANT** | 2 | 98.6% | 22 | 164864 | gemv, splitk[2], splitk[8], steel |
| addmm | float32 | K=2048 N=1024 | **VARIANT** | 2 | 94.9% | 7 | 11520 | gemv, splitk[2], splitk[4] |
| addmm | float32 | K=2048 N=4096 | **VARIANT** | 2 | 97.0% | 9 | 21376 | gemv, steel |
| addmm | float32 | K=4096 N=1024 | **VARIANT** | 2 | 97.4% | 10 | 544768 | gemv, splitk[2], splitk[4], splitk[8] |
| addmm | float32 | K=4096 N=4096 | **VARIANT** | 2 | 97.3% | 13 | 143360 | gemv, splitk[2], steel |
| addmm | float32 | K=512 N=1024 | **VARIANT** | 2 | 91.8% | 5 | 1448 | gemv, steel |
| addmm | float32 | K=512 N=4096 | **VARIANT** | 2 | 92.4% | 5 | 4160 | gemv, steel |
| layer_norm | bfloat16 | D=1024 | invariant | - | 0.0% | 0 | 0 | block |
| layer_norm | bfloat16 | D=4096 | invariant | - | 0.0% | 0 | 0 | block |
| layer_norm | bfloat16 | D=8192 | invariant | - | 0.0% | 0 | 0 | looped |
| layer_norm | float16 | D=1024 | invariant | - | 0.0% | 0 | 0 | block |
| layer_norm | float16 | D=4096 | invariant | - | 0.0% | 0 | 0 | block |
| layer_norm | float16 | D=8192 | invariant | - | 0.0% | 0 | 0 | looped |
| layer_norm | float32 | D=1024 | invariant | - | 0.0% | 0 | 0 | block |
| layer_norm | float32 | D=4096 | invariant | - | 0.0% | 0 | 0 | block |
| layer_norm | float32 | D=8192 | invariant | - | 0.0% | 0 | 0 | looped |
| matmul | bfloat16 | K=1024 N=1024 | invariant | - | 0.0% | 0 | 0 | gemv, splitk[2] |
| matmul | bfloat16 | K=1024 N=4096 | invariant | - | 0.0% | 0 | 0 | gemv, steel |
| matmul | bfloat16 | K=11008 N=1024 | invariant | - | 0.0% | 0 | 0 | gemv, splitk[16], splitk[32], splitk[8] |
| matmul | bfloat16 | K=11008 N=4096 | **VARIANT** | 2 | 0.1% | 1 | 1 | gemv, splitk[2], splitk[8], steel |
| matmul | bfloat16 | K=2048 N=1024 | **VARIANT** | 2 | 0.2% | 1 | 1 | gemv, splitk[2], splitk[4] |
| matmul | bfloat16 | K=2048 N=4096 | invariant | - | 0.0% | 0 | 0 | gemv, steel |
| matmul | bfloat16 | K=4096 N=1024 | **VARIANT** | 2 | 0.2% | 1 | 1 | gemv, splitk[2], splitk[4], splitk[8] |
| matmul | bfloat16 | K=4096 N=4096 | **VARIANT** | 2 | 0.1% | 1 | 1 | gemv, splitk[2], steel |
| matmul | bfloat16 | K=512 N=1024 | invariant | - | 0.0% | 0 | 0 | gemv, steel |
| matmul | bfloat16 | K=512 N=4096 | invariant | - | 0.0% | 0 | 0 | gemv, steel |
| matmul | float16 | K=1024 N=1024 | **VARIANT** | 2 | 0.2% | 1 | 1 | gemv, splitk[2] |
| matmul | float16 | K=1024 N=4096 | **VARIANT** | 2 | 0.2% | 1 | 1 | gemv, steel |
| matmul | float16 | K=11008 N=1024 | **VARIANT** | 2 | 0.7% | 1 | 1 | gemv, splitk[16], splitk[32], splitk[8] |
| matmul | float16 | K=11008 N=4096 | **VARIANT** | 2 | 1.1% | 1 | 3 | gemv, splitk[2], splitk[8], steel |
| matmul | float16 | K=2048 N=1024 | **VARIANT** | 2 | 1.0% | 1 | 7 | gemv, splitk[2], splitk[4] |
| matmul | float16 | K=2048 N=4096 | **VARIANT** | 2 | 0.5% | 1 | 9 | gemv, steel |
| matmul | float16 | K=4096 N=1024 | **VARIANT** | 2 | 0.6% | 1 | 1 | gemv, splitk[2], splitk[4], splitk[8] |
| matmul | float16 | K=4096 N=4096 | **VARIANT** | 2 | 0.6% | 1 | 33 | gemv, splitk[2], steel |
| matmul | float16 | K=512 N=1024 | invariant | - | 0.0% | 0 | 0 | gemv, steel |
| matmul | float16 | K=512 N=4096 | **VARIANT** | 2 | 0.2% | 1 | 1 | gemv, steel |
| matmul | float32 | K=1024 N=1024 | **VARIANT** | 2 | 93.5% | 5 | 4096 | gemv, splitk[2] |
| matmul | float32 | K=1024 N=4096 | **VARIANT** | 2 | 95.3% | 7 | 35707 | gemv, steel |
| matmul | float32 | K=11008 N=1024 | **VARIANT** | 2 | 96.3% | 11 | 4144 | gemv, splitk[16], splitk[32], splitk[8] |
| matmul | float32 | K=11008 N=4096 | **VARIANT** | 2 | 98.6% | 22 | 39913 | gemv, splitk[2], splitk[8], steel |
| matmul | float32 | K=2048 N=1024 | **VARIANT** | 2 | 94.9% | 7 | 51200 | gemv, splitk[2], splitk[4] |
| matmul | float32 | K=2048 N=4096 | **VARIANT** | 2 | 97.0% | 9 | 3594004 | gemv, steel |
| matmul | float32 | K=4096 N=1024 | **VARIANT** | 2 | 97.4% | 10 | 47104 | gemv, splitk[2], splitk[4], splitk[8] |
| matmul | float32 | K=4096 N=4096 | **VARIANT** | 2 | 97.3% | 13 | 54905 | gemv, splitk[2], steel |
| matmul | float32 | K=512 N=1024 | **VARIANT** | 2 | 92.7% | 5 | 2306 | gemv, steel |
| matmul | float32 | K=512 N=4096 | **VARIANT** | 2 | 92.7% | 5 | 1379 | gemv, steel |
| mean | bfloat16 | R=1024 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| mean | bfloat16 | R=128 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| mean | bfloat16 | R=4096 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| mean | bfloat16 | R=64 | invariant | - | 0.0% | 0 | 0 | row_reduce_small |
| mean | float16 | R=1024 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| mean | float16 | R=128 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| mean | float16 | R=4096 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| mean | float16 | R=64 | invariant | - | 0.0% | 0 | 0 | row_reduce_small |
| mean | float32 | R=1024 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| mean | float32 | R=128 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| mean | float32 | R=4096 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| mean | float32 | R=64 | invariant | - | 0.0% | 0 | 0 | row_reduce_small |
| rms_norm | bfloat16 | D=1024 | invariant | - | 0.0% | 0 | 0 | block |
| rms_norm | bfloat16 | D=4096 | invariant | - | 0.0% | 0 | 0 | block |
| rms_norm | bfloat16 | D=8192 | invariant | - | 0.0% | 0 | 0 | looped |
| rms_norm | float16 | D=1024 | invariant | - | 0.0% | 0 | 0 | block |
| rms_norm | float16 | D=4096 | invariant | - | 0.0% | 0 | 0 | block |
| rms_norm | float16 | D=8192 | invariant | - | 0.0% | 0 | 0 | looped |
| rms_norm | float32 | D=1024 | invariant | - | 0.0% | 0 | 0 | block |
| rms_norm | float32 | D=4096 | invariant | - | 0.0% | 0 | 0 | block |
| rms_norm | float32 | D=8192 | invariant | - | 0.0% | 0 | 0 | looped |
| sdpa | bfloat16 | H=8 Hkv=2 L=128 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | bfloat16 | H=8 Hkv=2 L=2048 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | bfloat16 | H=8 Hkv=2 L=4096 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector_2pass |
| sdpa | bfloat16 | H=8 Hkv=2 L=512 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | bfloat16 | H=8 Hkv=2 L=8192 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector_2pass |
| sdpa | bfloat16 | H=8 Hkv=8 L=128 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | bfloat16 | H=8 Hkv=8 L=2048 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | bfloat16 | H=8 Hkv=8 L=4096 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | bfloat16 | H=8 Hkv=8 L=512 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | bfloat16 | H=8 Hkv=8 L=8192 D=128 | invariant (1 B skipped) | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float16 | H=8 Hkv=2 L=128 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float16 | H=8 Hkv=2 L=2048 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float16 | H=8 Hkv=2 L=4096 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector_2pass |
| sdpa | float16 | H=8 Hkv=2 L=512 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float16 | H=8 Hkv=2 L=8192 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector_2pass |
| sdpa | float16 | H=8 Hkv=8 L=128 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float16 | H=8 Hkv=8 L=2048 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float16 | H=8 Hkv=8 L=4096 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float16 | H=8 Hkv=8 L=512 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float16 | H=8 Hkv=8 L=8192 D=128 | invariant (1 B skipped) | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float32 | H=8 Hkv=2 L=128 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float32 | H=8 Hkv=2 L=2048 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float32 | H=8 Hkv=2 L=4096 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector_2pass |
| sdpa | float32 | H=8 Hkv=2 L=512 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float32 | H=8 Hkv=2 L=8192 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector_2pass |
| sdpa | float32 | H=8 Hkv=8 L=128 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float32 | H=8 Hkv=8 L=2048 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float32 | H=8 Hkv=8 L=4096 D=128 | invariant (1 B skipped) | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float32 | H=8 Hkv=8 L=512 D=128 | invariant | - | 0.0% | 0 | 0 | sdpa_vector |
| sdpa | float32 | H=8 Hkv=8 L=8192 D=128 | invariant (2 B skipped) | - | 0.0% | 0 | 0 | sdpa_vector |
| softmax | bfloat16 | D=1024 | invariant | - | 0.0% | 0 | 0 | block |
| softmax | bfloat16 | D=4096 | invariant | - | 0.0% | 0 | 0 | block |
| softmax | bfloat16 | D=8192 | invariant | - | 0.0% | 0 | 0 | looped |
| softmax | float16 | D=1024 | invariant | - | 0.0% | 0 | 0 | block |
| softmax | float16 | D=4096 | invariant | - | 0.0% | 0 | 0 | block |
| softmax | float16 | D=8192 | invariant | - | 0.0% | 0 | 0 | looped |
| softmax | float32 | D=1024 | invariant | - | 0.0% | 0 | 0 | block |
| softmax | float32 | D=4096 | invariant | - | 0.0% | 0 | 0 | block |
| softmax | float32 | D=8192 | invariant | - | 0.0% | 0 | 0 | looped |
| sum | bfloat16 | R=1024 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| sum | bfloat16 | R=128 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| sum | bfloat16 | R=4096 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| sum | bfloat16 | R=64 | invariant | - | 0.0% | 0 | 0 | row_reduce_small |
| sum | float16 | R=1024 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| sum | float16 | R=128 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| sum | float16 | R=4096 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| sum | float16 | R=64 | invariant | - | 0.0% | 0 | 0 | row_reduce_small |
| sum | float32 | R=1024 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| sum | float32 | R=128 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| sum | float32 | R=4096 | invariant | - | 0.0% | 0 | 0 | row_reduce_looped, row_reduce_simple |
| sum | float32 | R=64 | invariant | - | 0.0% | 0 | 0 | row_reduce_small |


## Divergence detail — matmul and addmm

Read `B 2-32 -> splitk[8] 97.6% bits, med 13 / max 28672 ULP` as: for batch sizes 2 through 32 the
dispatch model predicts 8-way split-K, and 97.6% of that row's output elements differ from the
batch-1 result, by a median of 13 ULP.

- **addmm bfloat16 K=1024 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> splitk[2] 26.5% bits, med 1 / max 8 ULP
- **addmm bfloat16 K=1024 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 26.3% bits, med 1 / max 15 ULP
- **addmm bfloat16 K=11008 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[32] 27.9% bits, med 1 / max 3 ULP; B 33-64 -> splitk[16] 28.0% bits, med 1 / max 3 ULP; B=128 -> splitk[8] 28.0% bits, med 1 / max 3 ULP
- **addmm bfloat16 K=11008 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 25.5% bits, med 1 / max 6 ULP; B 33-64 -> splitk[2] 25.4% bits, med 1 / max 7 ULP; B=128 -> steel 25.5% bits, med 1 / max 7 ULP
- **addmm bfloat16 K=2048 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[4] 25.5% bits, med 1 / max 15072 ULP; B 33-128 -> splitk[2] 25.5% bits, med 1 / max 15071 ULP
- **addmm bfloat16 K=2048 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 25.7% bits, med 1 / max 42 ULP
- **addmm bfloat16 K=4096 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 26.9% bits, med 1 / max 4 ULP; B 33-64 -> splitk[4] 26.9% bits, med 1 / max 4 ULP; B=128 -> splitk[2] 26.8% bits, med 1 / max 4 ULP
- **addmm bfloat16 K=4096 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-64 -> splitk[2] 25.7% bits, med 1 / max 26 ULP; B=128 -> steel 25.7% bits, med 1 / max 26 ULP
- **addmm bfloat16 K=512 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 25.1% bits, med 1 / max 15089 ULP
- **addmm bfloat16 K=512 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 26.3% bits, med 1 / max 125 ULP
- **addmm float16 K=1024 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> splitk[2] 24.3% bits, med 1 / max 12 ULP
- **addmm float16 K=1024 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 26.3% bits, med 1 / max 130 ULP
- **addmm float16 K=11008 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[32] 27.6% bits, med 1 / max 3 ULP; B 33-64 -> splitk[16] 27.7% bits, med 1 / max 4 ULP; B=128 -> splitk[8] 27.5% bits, med 1 / max 6 ULP
- **addmm float16 K=11008 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 25.4% bits, med 1 / max 16 ULP; B 33-64 -> splitk[2] 25.6% bits, med 1 / max 17 ULP; B=128 -> steel 25.4% bits, med 1 / max 25 ULP
- **addmm float16 K=2048 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[4] 27.5% bits, med 1 / max 8 ULP; B 33-128 -> splitk[2] 27.5% bits, med 1 / max 7 ULP
- **addmm float16 K=2048 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 25.6% bits, med 1 / max 48 ULP
- **addmm float16 K=4096 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 27.1% bits, med 1 / max 10 ULP; B 33-64 -> splitk[4] 27.0% bits, med 1 / max 11 ULP; B=128 -> splitk[2] 27.1% bits, med 1 / max 10 ULP
- **addmm float16 K=4096 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-64 -> splitk[2] 25.6% bits, med 1 / max 56 ULP; B=128 -> steel 25.5% bits, med 1 / max 52 ULP
- **addmm float16 K=512 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 23.2% bits, med 1 / max 12 ULP
- **addmm float16 K=512 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 26.4% bits, med 1 / max 31 ULP
- **addmm float32 K=1024 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> splitk[2] 93.2% bits, med 5 / max 4000 ULP
- **addmm float32 K=1024 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 95.0% bits, med 7 / max 41104 ULP
- **addmm float32 K=11008 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[32] 95.4% bits, med 9 / max 9856 ULP; B 33-64 -> splitk[16] 96.3% bits, med 9 / max 14592 ULP; B=128 -> splitk[8] 96.1% bits, med 11 / max 7168 ULP
- **addmm float32 K=11008 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 97.6% bits, med 13 / max 164864 ULP; B 33-64 -> splitk[2] 98.2% bits, med 19 / max 144384 ULP; B=128 -> steel 98.6% bits, med 22 / max 126208 ULP
- **addmm float32 K=2048 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[4] 93.2% bits, med 6 / max 11520 ULP; B 33-128 -> splitk[2] 94.9% bits, med 7 / max 6400 ULP
- **addmm float32 K=2048 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 97.0% bits, med 9 / max 21376 ULP
- **addmm float32 K=4096 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 93.5% bits, med 7 / max 135168 ULP; B 33-64 -> splitk[4] 94.5% bits, med 8 / max 3200 ULP; B=128 -> splitk[2] 97.4% bits, med 10 / max 544768 ULP
- **addmm float32 K=4096 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-64 -> splitk[2] 96.7% bits, med 10 / max 143360 ULP; B=128 -> steel 97.3% bits, med 13 / max 111008 ULP
- **addmm float32 K=512 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 91.8% bits, med 5 / max 1448 ULP
- **addmm float32 K=512 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 92.4% bits, med 5 / max 4160 ULP
- **matmul bfloat16 K=11008 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 0.0% bits, med 1 / max 1 ULP; B 33-64 -> splitk[2] 0.1% bits, med 1 / max 1 ULP; B=128 -> steel 0.1% bits, med 1 / max 1 ULP
- **matmul bfloat16 K=2048 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[4] 0.2% bits, med 1 / max 1 ULP; B 33-128 -> splitk[2] 0.1% bits, med 1 / max 1 ULP
- **matmul bfloat16 K=4096 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 0.2% bits, med 1 / max 1 ULP; B 33-64 -> splitk[4] 0.1% bits, med 1 / max 1 ULP; B=128 -> splitk[2] 0.1% bits, med 1 / max 1 ULP
- **matmul bfloat16 K=4096 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-64 -> splitk[2] 0.1% bits, med 1 / max 1 ULP; B=128 -> steel 0.0% bits, med 1 / max 1 ULP
- **matmul float16 K=1024 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> splitk[2] 0.2% bits, med 1 / max 1 ULP
- **matmul float16 K=1024 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 0.2% bits, med 1 / max 1 ULP
- **matmul float16 K=11008 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[32] 0.2% bits, med 1 / max 1 ULP; B 33-64 -> splitk[16] 0.5% bits, med 1 / max 1 ULP; B=128 -> splitk[8] 0.7% bits, med 1 / max 1 ULP
- **matmul float16 K=11008 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 0.7% bits, med 1 / max 2 ULP; B 33-64 -> splitk[2] 0.9% bits, med 1 / max 2 ULP; B=128 -> steel 1.1% bits, med 1 / max 3 ULP
- **matmul float16 K=2048 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[4] 1.0% bits, med 1 / max 7 ULP; B 33-128 -> splitk[2] 0.8% bits, med 1 / max 5 ULP
- **matmul float16 K=2048 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 0.5% bits, med 1 / max 9 ULP
- **matmul float16 K=4096 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 0.4% bits, med 1 / max 1 ULP; B 33-64 -> splitk[4] 0.5% bits, med 1 / max 1 ULP; B=128 -> splitk[2] 0.6% bits, med 1 / max 1 ULP
- **matmul float16 K=4096 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-64 -> splitk[2] 0.5% bits, med 1 / max 33 ULP; B=128 -> steel 0.6% bits, med 1 / max 18 ULP
- **matmul float16 K=512 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 0.2% bits, med 1 / max 1 ULP
- **matmul float32 K=1024 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> splitk[2] 93.5% bits, med 5 / max 4096 ULP
- **matmul float32 K=1024 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 95.3% bits, med 7 / max 35707 ULP
- **matmul float32 K=11008 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[32] 95.4% bits, med 9 / max 4144 ULP; B 33-64 -> splitk[16] 96.3% bits, med 9 / max 3648 ULP; B=128 -> splitk[8] 96.1% bits, med 11 / max 2736 ULP
- **matmul float32 K=11008 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 97.6% bits, med 13 / max 28672 ULP; B 33-64 -> splitk[2] 98.3% bits, med 18 / max 16896 ULP; B=128 -> steel 98.6% bits, med 22 / max 39913 ULP
- **matmul float32 K=2048 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[4] 93.2% bits, med 6 / max 51200 ULP; B 33-128 -> splitk[2] 94.9% bits, med 7 / max 47104 ULP
- **matmul float32 K=2048 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 97.0% bits, med 9 / max 3594004 ULP
- **matmul float32 K=4096 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-32 -> splitk[8] 93.5% bits, med 7 / max 6144 ULP; B 33-64 -> splitk[4] 94.6% bits, med 8 / max 9216 ULP; B=128 -> splitk[2] 97.4% bits, med 10 / max 47104 ULP
- **matmul float32 K=4096 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-64 -> splitk[2] 96.7% bits, med 10 / max 24576 ULP; B=128 -> steel 97.3% bits, med 13 / max 54905 ULP
- **matmul float32 K=512 N=1024**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 92.7% bits, med 5 / max 2306 ULP
- **matmul float32 K=512 N=4096**: B=1 -> gemv 0.0% bits, med 0 / max 0 ULP; B 2-128 -> steel 92.7% bits, med 5 / max 1379 ULP

Notes on magnitude. The large max-ULP values (up to 3.6M in float32) are catastrophic cancellation:
output elements that land near zero have enormous relative spacing, so a small absolute difference
is a huge ULP distance. The median is the honest summary of typical divergence; the max shows that
some elements are destroyed entirely. Both are reported and neither is the whole picture.

The low-precision numbers deserve care in the other direction. `matmul` in bfloat16 reads
"invariant" at 5 of 10 shapes. That is not because the arithmetic agrees — the fp32 accumulators
differ exactly as they do in float32 — but because rounding to 8 mantissa bits usually lands on the
same value. It is masking, not invariance, and it will not hold for every input distribution. The
shapes that do diverge (0.1–1% of elements at 1 ULP) prove the underlying variance is still there.

## Finding 2 — the fused `addmm` epilogue double-rounds

`addmm` diverges far more than `matmul` in low precision (25–28% of bits vs 0.1–1%). This is not
reduction order. At batch 1 the `gemv_axbpy` path produces bit-identical output to
`(x @ W) + c`; at batch 8 the fused `steel` axpby epilogue does not, because it adds `C` to the
float32 accumulator before rounding to the output dtype, while the batch-1 path rounds first and
then adds. In float32 the two agree and only reduction order remains.

| dtype | comparison | bits differing (of 1024) |
|---|---|---|
| bfloat16 | addmm: B=1 vs B=8 | 261 |
| bfloat16 | matmul: B=1 vs B=8 | 2 |
| bfloat16 | matmul then add: B=1 vs B=8 | 0 |
| bfloat16 | addmm vs matmul-then-add at B=1 | 0 |
| bfloat16 | addmm vs matmul-then-add at B=8 | 261 |
| float16 | addmm: B=1 vs B=8 | 282 |
| float16 | matmul: B=1 vs B=8 | 10 |
| float16 | matmul then add: B=1 vs B=8 | 5 |
| float16 | addmm vs matmul-then-add at B=1 | 0 |
| float16 | addmm vs matmul-then-add at B=8 | 283 |
| float32 | addmm: B=1 vs B=8 | 954 |
| float32 | matmul: B=1 vs B=8 | 954 |
| float32 | matmul then add: B=1 vs B=8 | 954 |
| float32 | addmm vs matmul-then-add at B=1 | 0 |
| float32 | addmm vs matmul-then-add at B=8 | 0 |

This is a second, independent class of batch variance, and in bfloat16 it is the dominant one. A
batch-invariant `addmm` must pin the epilogue precision, not just the reduction order.

## Finding 3 — split-KV attention is context-variant, not batch-variant

The batch sweep holds context length fixed, so it cannot see split-KV. Forcing `MLX_SDPA_BLOCKS`
(the override at `scaled_dot_product_attention.cpp:477`) to a range of values at fixed input
isolates it:

| Hkv | L | predicted | distinct results across forced block counts | default matches |
|---|---|---|---|---|
| 8 | 2048 | sdpa_vector | 1 | 32, 64, 128, 256, 512, 1024 |
| 8 | 4096 | sdpa_vector | 1 | 32, 64, 128, 256, 512, 1024 |
| 8 | 8192 | sdpa_vector | 1 | 32, 64, 128, 256, 512, 1024 |
| 2 | 2048 | sdpa_vector | 1 | 32, 64, 128, 256, 512, 1024 |
| 2 | 4096 | sdpa_vector_2pass | 6 | 64 |
| 2 | 8192 | sdpa_vector_2pass | 6 | 64 |

For MHA the environment variable is inert — one result across all settings — confirming the
single-pass `sdpa_vector` kernel. For GQA at `L >= 4096` there are 6 distinct results across 7
settings: **the KV split count changes the answer bitwise**, and the default heuristic happens to
select 64 for these shapes.

The gate in source is:

```cpp
if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||
    (k.shape(1) < q.shape(1) && k.shape(2) >= 4096))
```

On `devc == 'g'` the first clause is dead, so split-KV fires only for GQA past 4096 tokens of
context. The split count is then a step function of `L` inside `sdpa_vector_2pass`. Consequence:
attention on this machine is bitwise stable with respect to batch size and with respect to which
other sequences share the batch, but *not* with respect to how much context has accumulated — and
on a Max or Ultra it would fire from 1024 tokens for every model, GQA or not. Portability of any
determinism claim across Apple GPU tiers is therefore not free.

## Finding 4 — `mx.compile` changes results, uniformly across batch

`mx.mean(x, axis=-1)` in float32 at `R=4096` returns a different bit pattern under `mx.compile`
than eager — 2 ULP — and does so identically at batch 1, 8, 32 and 128. This is a compilation
boundary effect, not batch variance, and it was the only such case in the sweep. It still matters:
a batch-invariant library has to pin whether an op runs compiled, or it will hand back different
answers for reasons unrelated to batching. Everything else agreed across all three evaluation
modes at every batch size, so `mx.eval()` placement itself is not a variance source.

## Negative results

These were expected to be variant and are not. Each is a genuine finding, and each removes work.

- **`mx.fast.rms_norm`, `mx.fast.layer_norm`, `mx.softmax`** — invariant at every batch size, every
  dtype, `D ∈ {1024, 4096, 8192}`. Dispatch keys only on `axis_size`, so the reduction structure is
  fixed per row and the batch only multiplies the grid. The `block` → `looped` kernel switch at
  `D > 4096` is a shape change, not a batch change, and does not break anything the library needs
  to hold. **RMSNorm drops out of Phase 1 entirely.**
- **`mx.sum` / `mx.mean` along an axis** — invariant, including across the `n_rows >= 32` boundary
  where `row_reduce_looped` is replaced by `row_reduce_simple`. Two different Metal kernels, same
  bits. This one was the most likely candidate for a clean, easily-fixed bug, and it is not a bug.
- **`mx.fast.scaled_dot_product_attention`** — invariant across batch at every context length in
  both MHA and GQA, in all three dtypes, including at `L = 4096` and `8192` where the GQA path is
  running the 2-pass split-KV kernel. The split count depends on `L`, not on `B`, so batch
  invariance holds even though the kernel is doing flash-decoding-style parallel-over-KV work.

  > **Amended by Phase 2.** This result is correct and it is also incomplete. The sweep varied the
  > batch dimension, `q.shape[0]`, because that is what "batch invariance" conventionally means.
  > MLX's attention dispatch also keys on `q.shape[2]`, the number of query tokens submitted
  > together, and there it *is* variant: `qL <= 8` takes `sdpa_vector` and `qL >= 9` takes
  > `steel_attention` (3883 of 4096 float32 bits differ for an unchanged query row), and the
  > split-KV block count is `64` when `(H/Hkv) * qL >= 4` and `32` otherwise, so at a GQA factor
  > of 2 decode and chunked prefill diverge at `qL = 2`. The measurements are in the Phase 2
  > section of BENCHMARK.md and the assertions are in `tests/test_invariance.py`. Phase 2 does
  > have work in it after all; this document under-scoped it by sweeping one axis.

## Kill criterion

Not met — matmul is genuinely batch-variant from batch 2, in all dtypes, at realistic inference
shapes. But the honest reading is that the project as originally scoped is roughly one third the
size it was expected to be. There is no RMSNorm work. There is no batch-invariant attention work,
in the sense the phase plan meant it — though see the amendment above: Phase 2 found attention
variance along the query-length axis that this sweep did not cover. What remains is:
**make `matmul` and `addmm` batch-invariant**,
plus two adjacent hazards worth fixing in the same library (the `addmm` epilogue, and pinning
`mx.compile` behaviour), plus one hazard that is out of reach of a batch sweep and belongs to
whoever wants long-context determinism (split-KV).

## Implications for Phase 1

On `devc == 'g'` the `GEMM_TPARAM_MACRO` tile parameters (`bm, bn, bk, wm, wn`) depend only on
dtype and transpose — never on M. Within `steel_matmul_regular` the K loop is therefore already
chunked identically regardless of batch, and each threadgroup owns its full K reduction. The
measured variance comes entirely from *which of three kernels is selected*, not from the selected
kernel behaving differently. If that holds up under direct test, batch-invariant matmul on this
device may not require writing a Metal kernel at all — it may only require forcing every batch size
down the regular `steel` path and disabling the `gemv` and split-K specialisations. That would be a
far smaller and far more maintainable artifact than a hand-written GEMM, and it should be tested
before any `mx.fast.metal_kernel` source is written.

The open question that decides it: MLX offers no public switch for this, so forcing the path means
either padding M (unacceptable — changes the math and wastes work) or reimplementing the dispatch.
Phase 1 should start by measuring whether `steel_matmul_regular` is in fact M-invariant, by
comparing batch sizes that stay inside the regular path at both ends (e.g. `K < N` shapes, where
split-K never fires) — the current data already contains this case and it shows divergence only at
the batch-1 gemv boundary, which is consistent with the hypothesis but not yet proof.
