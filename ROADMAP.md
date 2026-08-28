# Roadmap

v0.1 made matmul and attention bitwise batch-invariant on Apple Silicon and proved
it. v0.2 made them fast enough to be worth using: the GEMM moved to
`simdgroup_matrix`, dequantisation fused into it, and attention sinks landed. This
is what is still not done, and what was tried and rejected.

## Done in v0.2

**The GEMM runs on `simdgroup_matrix`** (was win #1). An output element still
accumulates over `k0` ascending then `kk` ascending — a fixed sequence of
fixed-width MMAs, for every M — so the invariance argument is unchanged. Prefill
overhead went from ~2.6× to ~1.95×, decode from ~2.0× to ~1.7×, and small-batch
GEMM (M = 8 to 32) is now at parity with stock MLX. End to end: 65.2% → 53.9% of
prefill throughput, 59.0% → 49.7% of decode.

**Dequantisation is fused into the tile load** (was win #3). v0.1 materialised an
fp16 copy of the whole weight per call; v0.2 unpacks the affine layout inside the
K loop. A group's scale and bias are per-weight, not per-batch, so this only ever
changed memory traffic. int4 decode went from 3.3–6.9× to 2.1–4.1×, prefill from
2.7–2.8× to 2.1–2.5×. 2-bit weights now work too.

**Attention sinks are supported.** A sink is an extra logit with no value vector:
it enters the running max and the denominator, never the numerator. Seeded in
simdgroup 0, exactly where MLX seeds it, so the single-pass case stays bitwise
identical to stock across all three dtypes.

## Known limitations in v0.2

**Prefill attention is 3.0–4.8× slower than `steel_attention`.** The invariant
kernel is a vector kernel — one threadgroup per query row, walking the whole key
sequence — so it re-reads K and V once per query row. This is measured, not
inferred: on 1×32/8×512×4096 fp16 it issues 34.4 GB of K/V loads against
`steel_attention`'s 0.017 GB, and sustains 642 GB/s doing it. The arithmetic is not
the problem; the redundant traffic is. See below for why the obvious fix was
rejected.

**Decode pays for a 32-row tile it does not use.** At M=1 the GEMM computes 32 rows
to keep 1, and the quantized kernel dequantises a 32-row slab of weight to use one
row of it. Decode is memory-bound on streaming weights so the observed penalty is
1.7× rather than 32×, but it is why int4 decode is the worst cell in BENCHMARK.md.
Fixing it means choosing a narrower tile when M is small, which is the exact
shape-dependent dispatch Phase 0 identified as the root cause of the bug. Not
doing it.

**Stock MLX's quantized variance is device-dependent.** On this M4
(`applegpu_g16g`) stock `quantized_matmul` moves most of the output bits between
batch 1 and batch 2. On the virtualised GPU GitHub's macOS runners expose
(`Apple Paravirtual device`, `air64_v27`) MLX selects different quantized kernels
and the same shapes come out already invariant, so the negative control skips
there instead of failing. The float `matmul` control is variant on both. Two
consequences: CI is a portability smoke test, not the proof — run `mlx-bi verify`
on the real hardware you care about — and any Phase 0-style measurement must be
redone per device rather than assumed from this one.

**Head dimensions must be multiples of 32.** 64, 96, 128 and 256 all work. MLA-style
asymmetric head dims (`V != D`) are handled. Anything not divisible by the SIMD
width raises.

**3, 5 and 6 bit quantized weights fall back to stock MLX.** The fused kernel
unpacks `32 / bits` values per uint32 word, which is only the actual layout for 2,
4 and 8 bits. The others are packed differently and are not implemented, so under
`strict=True` they raise rather than silently returning a variant result.

## Tried in v0.2 and rejected

**Blocking query rows together in the attention kernel.** This is the direct fix
for the redundant K/V traffic above, and it is bitwise free: a query-tiled variant
that shares each k/v load across BQ rows, while keeping each row's fold over the
keys in the order it would have had alone, produces *byte-identical* output to the
shipped kernel across every dtype, mask mode and query length tested.

It is also slower where it matters. Interleaved A/B, median of 15 pairs:

| BQ | prefill qL=512 | decode qL=1 |
|---|---:|---:|
| 2 | 0.94x | 1.05–1.16x |
| 4 | **0.85x** | **1.09–1.13x** |
| 8 | 2.1x | 1.7x |

The kernel already runs 1024 threads per threadgroup for one query row. Holding BQ
rows of query, output, running max and running sum multiplies the per-thread
register footprint, and past BQ=4 it spills outright. Trading a 13% decode
regression for a 15% prefill win is the wrong direction for an inference library:
decode is where determinism is actually being bought, and it is one forward pass
per token.

Getting real prefill speed needs a genuine flash-attention kernel on
`simdgroup_matrix`, which is a *different* kernel rather than a tiling of this one.
That kernel would have to replace the vector kernel for every shape, because using
it only for large `qL` is precisely the `qL`-dependent dispatch this library exists
to prevent — and it would give up the bitwise identity with stock MLX's single-pass
path that decode currently enjoys. That is a real project with a real trade-off,
not an optimisation, which is why it is not queued below.

**A bigger GEMM tile.** 64×64×16 measures worse than 32×32×16 on *both* prefill
(2.50× vs 2.01×) and decode (3.02× vs 1.66×), so there is nothing to win and no
dispatch question to answer. Table in BENCHMARK.md.

**A separate tile for the quantized kernel.** Swept eight configurations; the
float kernel's 32×32×16 was the best on both decode and prefill there too. No
second constant to keep in step.

## Fastest wins, in order

1. **Double-buffer the GEMM tile loads.** The tile loop stalls on its global reads
   before every MMA block instead of prefetching the next tile behind the current
   one. This is the whole remaining prefill gap now that tile size has been ruled
   out — 1.7 TFLOP/s against stock's 3.5 on 512×4096×4096 fp16. Nothing about
   double buffering touches the summation order.
2. **Vectorise the tile loads.** Each thread reads one element at a time into
   threadgroup memory; MLX's loader reads `vec<T, 4>` or wider. Same argument:
   changes when bytes arrive, not what is summed.
3. **A flash-attention kernel on `simdgroup_matrix`, replacing the vector kernel
   everywhere.** See the rejection note above — this is a design decision about
   whether to give up free decode attention for faster prefill, not a
   micro-optimisation, and it should be made with a dual-kernel design (below) on
   the table.

## Out of scope, permanently

**The M5 Neural Accelerator and the Metal 4 tensor path.** No M5 hardware was
available, so no line of this library targets it and no number in BENCHMARK.md
assumes it. On this device `is_nax_available()` is false: the architecture string
is `applegpu_g16g`, giving `arch_gen = 16`, and the NAX path requires generation 17
or later. Anyone extending this to M5 should redo Phase 0 from scratch there — a
tensor-core path is a new dispatch tree and therefore a new set of invariance
hazards, not a faster version of this one.

**Training, fine-tuning and gradients.** Every kernel here is forward-only. Nothing
has a registered vjp. Backward invariance is a strictly harder problem — the
gradient of a reduction is a broadcast, and the gradient of a broadcast is a
reduction whose order is chosen by the autodiff engine rather than by the kernel.

**Upstreaming into `vllm-metal`.** Not until the standalone library has been run
against real checkpoints by someone other than its author.

## Techniques deliberately not used

**DeepSeek-V4's dual-kernel attention.** DeepSeek-V4 ships two attention kernels
per shape — a fast one and a bitwise-reproducible one — and switches between them
by request flag, so a caller can pay for determinism only on the requests that need
it (evaluation, debugging, RL rollouts) while everything else takes the fast path.

v0.1 deferred this on the grounds that a switch is only honest once both kernels
exist and both are measured, and that reaching that point meant doing the
`simdgroup_matrix` GEMM first. That has now happened, and the picture has changed
in a way that makes the switch *less* attractive rather than more:

* For the GEMM, there is no longer much to switch between. Small-batch invariant
  GEMM is at parity with stock and decode is within 1.7×; a flag would be buying
  under a factor of two on the path it is easiest to justify paying for.
* For attention, the flag is real — prefill is 3–4.8× — but the fast side of it
  does not exist yet, and writing it means the flash kernel in win #3. The switch
  is downstream of that decision, not an alternative to it.

Revisit when someone actually needs faster prefill enough to write the flash
kernel. The flag is then twenty lines; the kernel is the work.
