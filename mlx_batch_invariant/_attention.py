"""Batch-invariant scaled dot product attention.

Phase 2 measured three ways MLX's attention changes its answer for a fixed query
token on this device, none of which involve the batch dimension MLX is usually
audited on:

  * ``q.shape[2] <= 8`` routes to ``sdpa_vector``, ``>= 9`` routes to
    ``steel_attention``.  A token in a prefill chunk of 8 and the same token in a
    chunk of 9 get different bits.
  * The split-KV block count is ``64`` if ``(H / Hkv) * q.shape[2] >= 4`` else
    ``32``.  For a model with a GQA factor of 2, decode and chunked prefill land on
    different reduction trees.
  * ``k.shape[2] >= 4096`` with GQA switches from one pass to two, so growing the
    context past 4096 changes the answer for keys that were already there.

This kernel has none of those switches.  One threadgroup owns one query row and the
entire reduction over N, in 32 fixed strides, for every shape.  The only thing the
summation order depends on is N, which is an input rather than a scheduling
decision.

Structure follows MLX's own ``sdpa_vector`` (MIT) so that the short-context,
single-pass case stays bitwise identical to stock MLX.
"""

import math

import mlx.core as mx

BN = 32  # simdgroups per threadgroup -- also the stride over the key sequence
BD = 32  # threads per simdgroup

_SOURCE = """
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  const int H = q_shape[1];
  const int qL = q_shape[2];
  const int Hkv = k_shape[1];
  const int N = k_shape[2];
  const int gqa = H / Hkv;

  const int bh = threadgroup_position_in_grid.z;
  const int s = threadgroup_position_in_grid.y;
  const int b = bh / H;
  const int h = bh - b * H;
  const int kvh = b * Hkv + h / gqa;

  const uint sg = simdgroup_index_in_threadgroup;
  const uint sl = thread_index_in_simdgroup;

  auto qp = q + (size_t)(bh * qL + s) * D + sl * qk_per_thread;
  auto kp = k + ((size_t)kvh * N + sg) * D + sl * qk_per_thread;
  auto vp = v + ((size_t)kvh * N + sg) * V + sl * v_per_thread;
  auto mp = mask;
  if (MASK_MODE == 2) {
    mp += (size_t)b * mstride[0] + (size_t)h * mstride[1] + (size_t)s * mstride[2] + sg;
  }

  thread U qv[qk_per_thread];
  thread U kv[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const U sc = scale[0];
  for (int i = 0; i < qk_per_thread; i++) {
    qv[i] = sc * static_cast<U>(qp[i]);
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  // Lowest finite float, not -inf: a fully masked row must not produce a NaN when
  // the running max is subtracted from itself.
  U max_score = -3.40282347e+38f;
  U sum_exp_score = 0;

  // Fixed 32-stride walk over the whole key sequence. No split, no second pass.
  for (int i = sg; i < N; i += BN) {
    bool use_key = true;
    U bias = 0;
    if (MASK_MODE == 1) {
      use_key = i <= (N - qL + s);
    } else if (MASK_MODE == 2) {
      bias = static_cast<U>(mp[0]);
      use_key = bias > -3.0e38f;
    }
    if (use_key) {
      for (int j = 0; j < qk_per_thread; j++) {
        kv[j] = static_cast<U>(kp[j]);
      }
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += qv[j] * kv[j];
      }
      score = simd_sum(score) + bias;

      U new_max = max(max_score, score);
      U factor = metal::fast::exp(max_score - new_max);
      U exp_score = metal::fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(vp[j]);
      }
    }
    kp += (size_t)BN * D;
    vp += (size_t)BN * V;
    if (MASK_MODE == 2) {
      mp += BN;
    }
  }

  // Combine the 32 partial softmaxes. Fixed shape, so fixed order.
  if (sl == 0) {
    max_scores[sg] = max_score;
    sum_exp_scores[sg] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[sl];
  U new_max = simd_max(max_score);
  U factor = metal::fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[sl] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[sl * BD + sg] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[sg * BD + sl] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (sl == 0) {
    auto op = out + (size_t)(bh * qL + s) * V + sg * v_per_thread;
    for (int i = 0; i < v_per_thread; i++) {
      op[i] = static_cast<T>(o[i]);
    }
  }
"""

_KERNEL = mx.fast.metal_kernel(
    name="bi_sdpa",
    input_names=["q", "k", "v", "mask", "mstride", "scale"],
    output_names=["out"],
    source=_SOURCE,
    header="#include <metal_simdgroup>\n",
    ensure_row_contiguous=True,
)

# Anything <= 8 elements is passed in the constant address space, which would change
# the pointer type of the unused-mask placeholder between launches.
_PAD = 16
_dummies = {}


def _dummy(dtype):
    d = _dummies.get(dtype)
    if d is None:
        d = _dummies[dtype] = mx.zeros((_PAD,), dtype=dtype)
    return d


def _mask_args(mask, B, H, qL, N, dtype):
    """Returns (array, [b_stride, h_stride, q_stride], mode)."""
    if mask is None:
        return _dummy(dtype), _dummy(mx.int32), 0
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError("string mask must be 'causal', got %r" % (mask,))
        return _dummy(dtype), _dummy(mx.int32), 1

    m = mask
    if m.dtype == mx.bool_:
        m = mx.where(m, mx.array(0, dtype), mx.array(-mx.inf, dtype))
    m = m.astype(dtype)
    while m.ndim < 4:
        m = mx.expand_dims(m, 0)
    if m.ndim != 4:
        raise ValueError("mask must have at most 4 dimensions, got %d" % mask.ndim)
    Bm, Hm, Lm, Nm = m.shape
    if Nm != N or Lm not in (1, qL) or Hm not in (1, H) or Bm not in (1, B):
        raise ValueError(
            "mask shape %s is not broadcastable to (%d, %d, %d, %d)"
            % (tuple(m.shape), B, H, qL, N)
        )
    m = mx.contiguous(m)
    strides = mx.array(
        [0 if Bm == 1 else Hm * Lm * N, 0 if Hm == 1 else Lm * N, 0 if Lm == 1 else N]
        + [0] * (_PAD - 3),
        dtype=mx.int32,
    )
    return m, strides, 2


def scaled_dot_product_attention(q, k, v, *, scale=None, mask=None):
    """Bitwise-invariant SDPA. ``q`` is (B, H, L, D), ``k``/``v`` are (B, Hkv, N, ...).

    ``mask`` may be ``None``, ``"causal"``, or an additive/boolean array
    broadcastable over the batch, head and query dimensions.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be 4-dimensional (B, H, L, D)")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k and v must share a dtype")

    B, H, qL, D = q.shape
    Bk, Hkv, N, Dk = k.shape
    Bv, Hv, Nv, V = v.shape
    if (Bk, Hv, Nv, Dk) != (B, Hkv, N, D) or Bv != B:
        raise ValueError("inconsistent q/k/v shapes: %s %s %s" % (q.shape, k.shape, v.shape))
    if H % Hkv:
        raise ValueError("head count %d is not a multiple of kv head count %d" % (H, Hkv))
    if D % BD or V % BD:
        raise ValueError(
            "head dimensions must be multiples of %d, got D=%d V=%d" % (BD, D, V)
        )

    scale = 1.0 / math.sqrt(D) if scale is None else float(scale)
    m, mstride, mode = _mask_args(mask, B, H, qL, N, q.dtype)

    return _KERNEL(
        inputs=[q, k, v, m, mstride, mx.full((_PAD,), scale, dtype=mx.float32)],
        template=[("T", q.dtype), ("D", D), ("V", V), ("BN", BN), ("BD", BD),
                  ("MASK_MODE", mode)],
        grid=(BD, BN * qL, B * H),
        threadgroup=(BD, BN, 1),
        output_shapes=[(B, H, qL, V)],
        output_dtypes=[q.dtype],
    )[0]
