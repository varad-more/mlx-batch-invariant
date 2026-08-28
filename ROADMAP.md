# Roadmap

v0.1 makes matmul and attention bitwise batch-invariant on Apple Silicon and proves
it. What follows is what it deliberately does not do.

## Known limitations in v0.1

**Quantized models are not covered.** `mx.quantized_matmul` and
`nn.QuantizedLinear` are untouched, so `batch_invariant_mode()` does nothing useful
for a 4-bit checkpoint — which is most of what people actually run under MLX on a
laptop. The invariance question there is different and probably easier: the
dequantise-and-accumulate path is already a single kernel. It needs measuring
before it needs fixing, exactly as Phase 0 did for the float path.

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
3. **Quantized matmul.** Measure first.

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
