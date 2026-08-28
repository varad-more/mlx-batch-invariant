# Roadmap

v0.1 makes matmul and attention bitwise batch-invariant on Apple Silicon and proves
it. What follows is what it deliberately does not do.

## Known limitations in v0.1

**Quantized matmul is invariant but slow.** Stock `mx.quantized_matmul` is
badly batch-variant — 113360 of 151936 logit bits move between batch 1 and batch 2
on a real 4-bit Qwen1.5-0.5B. `batch_invariant_mode()` fixes it by dequantizing the
weight and running the invariant GEMM, which is correct (dequantization is
elementwise, so it cannot depend on the batch) and costs 2.7–6.9×. The whole weight
is materialised in fp16 for every call, which is why decode is the worst case: a
single row of activations pays for the entire matrix. A fused invariant kernel that
dequantizes per tile inside the K loop would remove nearly all of it.

**Stock MLX's quantized variance is device-dependent.** On this M4
(`applegpu_g16g`) stock `quantized_matmul` moves most of the output bits between
batch 1 and batch 2. On the virtualised GPU GitHub's macOS runners expose
(`Apple Paravirtual device`, `air64_v27`) MLX selects different quantized kernels
and the same shapes come out already invariant, so the negative control skips
there instead of failing. The float `matmul` control is variant on both. Two
consequences: CI is a portability smoke test, not the proof — run `mlx-bi verify`
on the real hardware you care about — and any Phase 0-style measurement must be
redone per device rather than assumed from this one.

**Attention sinks are unsupported.** `mx.fast.scaled_dot_product_attention(...,
sinks=...)` falls through to stock MLX (or raises, under `strict=True`). The kernel
change is small — seed the running max and sum from the sink logit in simdgroup 0,
as MLX does — but it is untested here so it is not claimed.

**Head dimensions must be multiples of 32.** 64, 96, 128 and 256 all work. MLA-style
asymmetric head dims (`V != D`) are handled. Anything not divisible by the SIMD
width raises.

**Prefill uses a vector attention kernel.** Correct, invariant, and 3–5× slower than
`steel_attention` at long query lengths. See BENCHMARK.md.

**No simdgroup-matrix GEMM.** The GEMM is scalar-FMA. This is the single largest
source of overhead in the library and the first thing to fix.

## Fastest wins, in order

1. **Rewrite the GEMM on `simdgroup_matrix`.** A fixed tile with one threadgroup
   owning the entire K reduction is just as invariant with `simdgroup_multiply_
   accumulate` as with scalar FMAs. Expected to remove most of the 2.6× prefill gap
   and much of the 2.0× decode gap. Nothing about invariance requires the slow
   instruction; the current kernel uses it because it was the shortest correct
   thing to write first.
2. **A tiled, invariant prefill attention kernel.** Fixed query and key block
   sizes, fixed iteration order, no `align_Q`/`align_K` specialisation. The
   invariance requirement is only that the block sizes are compile-time constants
   rather than functions of `qL` or `N`.
3. **A fused invariant quantized kernel.** Dequantize per tile inside the K loop
   instead of materialising the whole weight up front. The invariance argument is
   unchanged — group scales are per-weight, not per-batch — so this is purely a
   memory-traffic fix, worth 2.7–6.9× on every quantized layer. Measured in
   BENCHMARK.md under `bench.py quant`.

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

That is the right long-term design and it is explicitly *not* what v0.1 does. v0.1
has one kernel, always invariant. The reasons are:

* A dual-kernel design is only honest once both kernels exist and both are
  measured. Shipping a switch whose "fast" side is stock MLX and whose "reproducible"
  side is a scalar-FMA GEMM would mean shipping a 2.6× cliff and calling it a
  feature.
* The interesting half of the DeepSeek result is that the reproducible kernel is
  close enough in speed to make the switch cheap. Reaching that point here means
  doing roadmap item 1 first.
* One kernel is testable. `mlx-bi verify` can assert a single property. A switch
  needs the property asserted on one side and deliberately *not* asserted on the
  other, plus a test that the flag actually routes — three times the surface for a
  v0.1.

Revisit after the simdgroup-matrix GEMM lands, when the fast path and the invariant
path are near enough that a flag is worth having.
