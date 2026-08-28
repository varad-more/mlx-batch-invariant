"""Batch-invariant GEMM.

Phase 0 established that on this hardware MLX's matmul is batch-variant purely
because of dispatch selection: batch 1 takes a `gemv` kernel, larger batches take
`steel_gemm` or `steel_gemm_splitk`, and the split-K partition count is an explicit
function of M.  Each of those kernels is itself bitwise stable as M varies.

So the fix is one kernel that is always used.  Every threadgroup owns the whole
reduction over K, the tile shape is a compile-time constant, and nothing about the
launch depends on M except how many tiles are launched.

The reduction runs on `simdgroup_multiply_accumulate` over 8x8 fragments.  Nothing
about invariance requires the slow instruction -- an output element accumulates over
`k0` ascending and then `kk` ascending, a fixed sequence of fixed-width MMAs, for
every M -- so v0.1's scalar-FMA inner loop was simply the shortest correct thing to
write first.  The fragment loops carry `unroll(full)` because without it the
compiler spills the accumulators to thread memory and the MMA loses to the FMAs.
"""

import mlx.core as mx

# Compile-time tile. Fixed for every shape and every dtype -- deliberately not tuned
# per input, because that is the bug this library exists to prevent. SG_M x SG_N
# simdgroups, so each one owns a (BM / SG_M) x (BN / SG_N) corner of the tile.
BM, BN, BK, SG_M, SG_N = 32, 32, 16, 2, 2
NSG = SG_M * SG_N  # 4 simdgroups = 128 threads per threadgroup
TILE = dict(BM=BM, BN=BN, BK=BK, SG_M=SG_M, SG_N=SG_N)

_SOURCE = """
  const int M = a_shape[0];
  const int K = a_shape[1];
  const int N = TRANSPOSE_B ? b_shape[0] : b_shape[1];

  constexpr int NT = SG_M * SG_N * 32;
  constexpr int WM = BM / SG_M;   // rows of the tile owned by one simdgroup
  constexpr int WN = BN / SG_N;   // columns likewise
  constexpr int FM = WM / 8;      // 8x8 accumulator fragments, vertically
  constexpr int FN = WN / 8;      // ... and horizontally

  const uint ti = thread_index_in_threadgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int sgm = sg / SG_N;
  const int sgn = sg % SG_N;
  const int row0 = threadgroup_position_in_grid.y * BM;
  const int col0 = threadgroup_position_in_grid.x * BN;

  threadgroup float As[BM * BK];
  threadgroup float Bs[BK * BN];
  threadgroup float Cs[BM * BN];

  simdgroup_float8x8 acc[FM][FN];
  #pragma clang loop unroll(full)
  for (int i = 0; i < FM; i++) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < FN; j++) {
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  // One threadgroup, the whole reduction. The trip count depends on K alone, so the
  // summation order for a given output element is identical for every M.
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = ti; idx < BM * BK; idx += NT) {
      const int r = idx / BK;
      const int c = idx % BK;
      const int gr = row0 + r;
      const int gk = k0 + c;
      As[idx] = (gr < M && gk < K) ? static_cast<float>(a[gr * K + gk]) : 0.0f;
    }
    for (int idx = ti; idx < BK * BN; idx += NT) {
      const int r = idx / BN;
      const int c = idx % BN;
      const int gk = k0 + r;
      const int gn = col0 + c;
      float v = 0.0f;
      if (gk < K && gn < N) {
        v = TRANSPOSE_B ? static_cast<float>(b[gn * K + gk])
                        : static_cast<float>(b[gk * N + gn]);
      }
      Bs[idx] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 af[FM];
      simdgroup_float8x8 bf[FN];
      #pragma clang loop unroll(full)
      for (int i = 0; i < FM; i++) {
        simdgroup_load(af[i], As + (sgm * WM + i * 8) * BK + kk, BK);
      }
      #pragma clang loop unroll(full)
      for (int j = 0; j < FN; j++) {
        simdgroup_load(bf[j], Bs + kk * BN + sgn * WN + j * 8, BN);
      }
      #pragma clang loop unroll(full)
      for (int i = 0; i < FM; i++) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < FN; j++) {
          simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma clang loop unroll(full)
  for (int i = 0; i < FM; i++) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < FN; j++) {
      simdgroup_store(acc[i][j], Cs + (sgm * WM + i * 8) * BN + sgn * WN + j * 8, BN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Epilogue: the bias is added to the float32 accumulator and rounded once. MLX is
  // inconsistent here -- its gemv path rounds before adding, its fused path after --
  // which is a second source of batch variance in float16 and bfloat16.
  for (int idx = ti; idx < BM * BN; idx += NT) {
    const int gr = row0 + idx / BN;
    const int gn = col0 + idx % BN;
    if (gr >= M || gn >= N) {
      continue;
    }
    float v = Cs[idx];
    if (HAS_BIAS) {
      v = coef[0] * v + coef[1] * static_cast<float>(bias[gn]);
    }
    out[gr * N + gn] = static_cast<T>(v);
  }
"""

_KERNEL = mx.fast.metal_kernel(
    name="bi_gemm",
    input_names=["a", "b", "bias", "coef"],
    output_names=["out"],
    source=_SOURCE,
    header="#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
    ensure_row_contiguous=True,
)

# Anything <= 8 elements would be passed in the constant address space, which changes
# the pointer type of the placeholder between launches.
_PAD = 16
_NO_BIAS = None


def _gemm(a, b, bias, transpose_b, alpha=1.0, beta=1.0):
    global _NO_BIAS
    if a.dtype != b.dtype:
        raise ValueError("a and b must share a dtype, got %s and %s" % (a.dtype, b.dtype))
    if a.ndim < 2 or b.ndim != 2:
        raise ValueError("expected a.ndim >= 2 and b.ndim == 2")

    lead, K = a.shape[:-1], a.shape[-1]
    M = 1
    for d in lead:
        M *= d
    kb, N = (b.shape[1], b.shape[0]) if transpose_b else (b.shape[0], b.shape[1])
    if kb != K:
        raise ValueError("inner dimensions do not match: %d vs %d" % (K, kb))

    a2 = a.reshape(M, K)
    if bias is None:
        if _NO_BIAS is None or _NO_BIAS.dtype != a.dtype:
            _NO_BIAS = mx.zeros((_PAD,), dtype=a.dtype)
        bias_arr, has_bias = _NO_BIAS, 0
    else:
        if bias.shape != (N,):
            raise ValueError("bias must have shape (%d,), got %s" % (N, bias.shape))
        bias_arr, has_bias = bias, 1

    grid = ((N + BN - 1) // BN * 32, (M + BM - 1) // BM * NSG, 1)
    coef = mx.array([alpha, beta] + [0.0] * (_PAD - 2), dtype=mx.float32)
    out = _KERNEL(
        inputs=[a2, b, bias_arr, coef],
        template=[("T", a.dtype), ("BM", BM), ("BN", BN), ("BK", BK),
                  ("SG_M", SG_M), ("SG_N", SG_N),
                  ("TRANSPOSE_B", int(transpose_b)), ("HAS_BIAS", has_bias)],
        grid=grid,
        threadgroup=(32, NSG, 1),
        output_shapes=[(M, N)],
        output_dtypes=[a.dtype],
    )[0]
    return out.reshape(*lead, N)


def matmul(a, b):
    """a @ b with b laid out (K, N). Bitwise independent of a's leading dimensions."""
    return _gemm(a, b, None, transpose_b=False)


def addmm(c, a, b, alpha=1.0, beta=1.0):
    """alpha * (a @ b) + beta * c, with b laid out (K, N) and c of shape (N,)."""
    return _gemm(a, b, c, transpose_b=False, alpha=alpha, beta=beta)


def linear(x, w, bias=None):
    """x @ w.T (+ bias) with w laid out (N, K), the nn.Linear convention.

    Takes the weight in its stored layout so no transposed copy is made.
    """
    return _gemm(x, w, bias, transpose_b=True)
