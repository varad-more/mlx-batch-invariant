# mlx-batch-invariant

Batch-invariant, bitwise-deterministic kernels for [MLX](https://github.com/ml-explore/mlx)
on Apple Silicon.

Run the same prompt through a model twice. If the second run happens to be batched
with someone else's request, MLX gives you different logits — not "slightly
different", *82% of the bits*. This library gives you the same bits every time.

```
| path            | batch | logits differing from batch 1 |
|-----------------|------:|------------------------------:|
| stock MLX       |     8 |                26491 / 32000  |
| batch invariant |     8 |                    0 / 32000  |
```

## Install

```bash
pip install mlx-batch-invariant
```

Requires macOS on Apple Silicon and a native arm64 Python. It will not do anything
useful under Rosetta.

## Use

```python
import mlx_batch_invariant as bi

with bi.batch_invariant_mode():
    logits = model(tokens)
```

Inside the block, `mx.matmul`, `mx.addmm`, the `@` operator, `nn.Linear` and
`mx.fast.scaled_dot_product_attention` route through invariant kernels. Anything the
library cannot make invariant raises `NotImplementedError` rather than silently
falling back — pass `batch_invariant_mode(strict=False)` to get the fallback
instead, which is useful for finding out what a model needs but is not something to
ship.

Or call the kernels directly:

```python
y = bi.linear(x, w, bias)                          # w is (N, K), the nn.Linear layout
y = bi.matmul(a, b)                                # b is (K, N)
y = bi.addmm(c, a, b, alpha=1.0, beta=1.0)
y = bi.scaled_dot_product_attention(q, k, v, mask="causal")
```

## Verify it on your own machine

```bash
mlx-bi verify
```

Sweeps three dtypes, ten-ish shapes, eleven batch sizes and nine query lengths,
comparing raw `uint32`/`uint16` bit patterns. It also checks that stock MLX is
*still* batch-variant — if that control ever passes, MLX has been fixed and this
library is obsolete, which is a result worth printing.

## What "batch-invariant" means here

The output for a given row is bitwise identical regardless of:

* how many other rows were in the batch (**batch size**),
* which rows they were (**batch neighbours**),
* how many query tokens were submitted together (**query length** — decode vs
  chunked prefill),
* how long the KV cache is when the route changes under it (**context length**),
* whether the call was eager, inside a graph, or inside `mx.compile`.

Comparisons are on `uint32`/`uint16` bit patterns. `np.allclose` and `==` on floats
appear nowhere in the invariance assertions.

## Why MLX is not invariant

Measured on an M4, MLX 0.32.0 — the full sweep is in [PHASE0.md](PHASE0.md).

**Matmul.** Batch 1 dispatches `gemv`; batch 2 and up dispatch `steel_gemm` or
`steel_gemm_splitk`, and the split-K partition count is an explicit function of M.
Each of those kernels is bitwise stable on its own — 100% of the variance is which
one gets picked. Divergence starts at **batch 2**, in every dtype, at every shape:
92–99% of float32 bits differ, at a median of 5–22 ULP.

**Attention.** Three switches, none of them in the batch dimension:

* `q.shape[2] <= 8` uses `sdpa_vector`, `>= 9` uses `steel_attention`. A token in a
  prefill chunk of 8 and the same token in a chunk of 9 get different answers.
* The split-KV block count is 64 if `(H/Hkv) * q.shape[2] >= 4`, else 32. For a
  model with a GQA factor of 2, decode and chunked prefill land on different
  reduction trees.
* Crossing 4096 KV entries with GQA switches from one pass to two, so growing the
  context changes the answer for keys that were already there — 3575 of 4096
  float32 bits, even when the newly added key is fully masked out.

**Everything else is already fine.** `rms_norm`, `layer_norm`, `softmax`, `sum`,
`mean` are bitwise batch-invariant across the whole sweep, including across genuine
kernel switches. This library does not wrap them; it has a regression test asserting
they stay that way. Phase 0's negative results were the most useful part of it: the
project turned out to be a third the size it was assumed to be.

## How the kernels work

**GEMM.** One `mx.fast.metal_kernel`, a compile-time 32×32×16 tile with a 4×4
register tile, 64 threads. One threadgroup owns the entire reduction over K. The
loop trip count is a function of K alone and a given row always lands at the same
position in the same tile, so nothing about the summation order can move when M
changes. No split-K, no atomics, float32 accumulators. The bias is added into the
float32 accumulator and rounded once — MLX rounds before adding on its gemv path
and after adding on its fused path, which is a second, separate source of variance.

**Attention.** A single-pass vector kernel: one threadgroup per query row, 32
simdgroups walking the key sequence in fixed stride-32 order, online softmax in
float32, then a fixed cross-simdgroup reduction. No split-KV, no second pass, no
route switching. Structure follows MLX's own `sdpa_vector`, which means that
wherever MLX takes its single-pass path this kernel is **bit-for-bit identical to
stock MLX** — including all three mask forms. Adopting it costs no accuracy at all
in the common case.

## Accuracy

Against a float64 NumPy reference:

| | ours | MLX | ratio |
|---|---|---|---|
| GEMM float32 | 2.1e-6 – 4.5e-6 | 5.0e-7 – 4.0e-6 | 1.0 – 5.6× |
| GEMM float16/bfloat16 | — | — | **1.00×** (identical bits) |
| Attention, MLX single-pass shapes | — | — | **1.00×** (identical bits) |

In float32 GEMM we are up to 5.6× less accurate than MLX at small M. That is the
price, and it is expected: MLX's split-K is effectively a partial pairwise
summation, while a single sequential pass over K is what makes the order fixed. The
absolute error stays at or below 4.5e-6 relative.

## Cost

Full numbers and methodology in [BENCHMARK.md](BENCHMARK.md).

| | throughput cost |
|---|---:|
| decode attention, 512 → 32768 ctx, batch 1 → 32 | **~0%** (0.97–1.06×) |
| end-to-end decode, 0.94B fp16 | 59.0% |
| end-to-end prefill 256, 0.94B fp16 | 65.2% |

For reference, Thinking Machines' CUDA `batch_invariant_ops` reports roughly 61.5%
and SGLang's tuned deterministic mode roughly 34.35%. Different hardware, model and
framework — directional only.

Decode attention is free. The entire end-to-end cost is the GEMM, and it is not a
cost of determinism: the kernel uses scalar FMAs where MLX uses `simdgroup_matrix`.
A simdgroup-matrix kernel with the same fixed tile would be equally invariant and
much faster. See [ROADMAP.md](ROADMAP.md).

## Quantized models

4-bit checkpoints are covered. `mx.quantized_matmul` — which is what
`nn.QuantizedLinear` calls — is routed through dequantize-then-invariant-GEMM, so a
real MLX checkpoint is invariant end to end:

```
$ .venv/bin/python bench/real_model.py     # mlx-community/Qwen1.5-0.5B-Chat-4bit
stock MLX      : B=2 113360/151936 bits, B=4 121209/151936 bits, B=8 117380/151936 bits
batch-invariant: B=2 0/151936 bits, B=4 0/151936 bits, B=8 0/151936 bits
```

It costs 2.7–6.9× on the quantized layers, because the whole weight is dequantized
on every call. That ratio is a ceiling set by an unwritten fused kernel, not by
invariance. See [ROADMAP.md](ROADMAP.md).

## Limitations

Attention sinks are unsupported. Head dimensions must be multiples of 32. Forward
pass only, no gradients. M5 Neural Accelerator / Metal 4 tensor paths are
permanently out of scope and no number here assumes them.

## Prior art

* Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), and `batch_invariant_ops`
* vLLM's batch-invariant backend, SGLang's deterministic mode
* Megatron-Core `batch_invariant_kernels`
* DeepSeek-V4's dual-kernel reproducible attention — discussed in [ROADMAP.md](ROADMAP.md)

All of the above are CUDA. This is the Metal one.

## Development

```bash
uv venv --python 3.12 && uv pip install "mlx==0.32.0" numpy
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=. .venv/bin/python probe/probe.py selftest     # Phase 0 harness self-check
PYTHONPATH=. .venv/bin/python bench/bench.py all
uv pip install mlx-lm && .venv/bin/python bench/real_model.py   # real 4-bit checkpoint
```

CI runs the suite on GitHub's macOS runners, whose GPU is virtualised
(`Apple Paravirtual device`). That checks portability, not behaviour on real
Apple Silicon — stock MLX's quantized kernels are already invariant there, so the
quantized negative control skips. `mlx-bi verify` on the hardware you actually
run on is the proof.

## License

MIT.
