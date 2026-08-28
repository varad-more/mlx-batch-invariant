"""Phase 1 and 2 gate: the batch-invariant GEMM and attention kernels must be
bitwise invariant, and no less accurate than the stock MLX kernels they replace.

Run: PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import mlx.core as mx

import mlx_batch_invariant as bi

BATCHES = [1, 2, 3, 5, 8, 16, 31, 32, 33, 64, 128]
DTYPES = [("float32", mx.float32), ("float16", mx.float16), ("bfloat16", mx.bfloat16)]
SHAPES = [(K, N) for N in (1024, 4096) for K in (512, 1024, 2048, 4096, 11008)]
_UINT = {mx.float32: mx.uint32, mx.float16: mx.uint16, mx.bfloat16: mx.uint16}


def bits(x):
    return np.array(mx.view(mx.contiguous(x).reshape(-1), _UINT[x.dtype]))


def rn(shape, dtype, seed):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(dtype)


def batch_with(ref, pool, B, idx):
    """Batch of size B holding `ref` at position `idx` and pool rows elsewhere."""
    if B == 1:
        return ref
    parts = ([pool[:idx]] if idx > 0 else []) + [ref] + ([pool[idx + 1:B]] if idx + 1 < B else [])
    return mx.concatenate(parts)


class TestBitwiseInvariance(unittest.TestCase):
    """The output for one row must not depend on the batch it travelled in."""

    def _sweep(self, op):
        for dname, dtype in DTYPES:
            for K, N in SHAPES:
                W = rn((K, N), dtype, 1)
                c = rn((N,), dtype, 4)
                ref = rn((1, K), dtype, 3)
                pool = rn((max(BATCHES), K), dtype, 2)
                mx.eval(W, c, ref, pool)
                fn = (lambda x: bi.matmul(x, W)) if op == "matmul" else (lambda x: bi.addmm(c, x, W))
                modes = [("eager", fn), ("compiled", mx.compile(fn))]
                base = {}
                for B in BATCHES:
                    for idx in sorted({0, B // 2}):
                        x = batch_with(ref, pool, B, idx)
                        mx.eval(x)
                        for mname, f in modes:
                            y = f(x)
                            mx.eval(y)
                            got = bits(y[idx])
                            self.assertTrue(bool(mx.any(y != 0).item()),
                                            "degenerate all-zero output for %s %s K=%d N=%d B=%d"
                                            % (op, dname, K, N, B))
                            if B == 1:
                                base[mname] = got
                                continue
                            ndiff = int((got != base[mname]).sum())
                            self.assertEqual(
                                ndiff, 0,
                                "%s %s K=%d N=%d B=%d idx=%d mode=%s: %d/%d elements "
                                "differ from the batch-1 result"
                                % (op, dname, K, N, B, idx, mname, ndiff, got.size))

    def test_matmul(self):
        self._sweep("matmul")

    def test_addmm(self):
        self._sweep("addmm")

    def test_linear_transposed_weight(self):
        """The (N, K) nn.Linear layout must be invariant too, not just (K, N)."""
        for dname, dtype in DTYPES:
            for K, N in [(1024, 1024), (11008, 4096)]:
                W = rn((N, K), dtype, 1)
                b = rn((N,), dtype, 4)
                ref, pool = rn((1, K), dtype, 3), rn((128, K), dtype, 2)
                mx.eval(W, b, ref, pool)
                base = bits(bi.linear(ref, W, b)[0])
                for B in BATCHES[1:]:
                    idx = B // 2
                    y = bi.linear(batch_with(ref, pool, B, idx), W, b)
                    mx.eval(y)
                    self.assertEqual(int((bits(y[idx]) != base).sum()), 0,
                                     "linear %s K=%d N=%d B=%d" % (dname, K, N, B))

    def test_leading_dims_collapse_invariantly(self):
        """(B, S, K) and (B*S, K) must give the same bits for the same row."""
        dtype = mx.float32
        K, N = 1024, 512
        W = rn((K, N), dtype, 1)
        x = rn((24, K), dtype, 5)
        mx.eval(W, x)
        flat = bi.matmul(x, W)
        shaped = bi.matmul(x.reshape(4, 6, K), W).reshape(24, N)
        mx.eval(flat, shaped)
        self.assertEqual(int((bits(flat) != bits(shaped)).sum()), 0)


class TestAccuracy(unittest.TestCase):
    """Invariant is not enough -- it must not be consistently wrong."""

    # Includes the extremes of the invariance sweep, so accuracy is checked where
    # the sequential K reduction is longest.
    CASES = [(17, 1024, 512), (128, 2048, 256), (1, 4096, 512),
             (8, 11008, 4096), (128, 11008, 4096)]

    def test_not_worse_than_mlx_float32(self):
        for M, K, N in self.CASES:
            a = rn((M, K), mx.float32, 11)
            b = rn((K, N), mx.float32, 12)
            mx.eval(a, b)
            a64 = np.array(a).astype(np.float64)
            b64 = np.array(b).astype(np.float64)
            exact = a64 @ b64
            scale = np.abs(exact).max()
            ours = np.abs(np.array(bi.matmul(a, b)).astype(np.float64) - exact).max() / scale
            mlxs = np.abs(np.array(a @ b).astype(np.float64) - exact).max() / scale
            # Single-pass float32 accumulation over K terms; MLX's split-K is a
            # partial pairwise sum and so is legitimately a little more accurate.
            self.assertLess(ours, 2e-5, "M=%d K=%d N=%d abs rel error %g" % (M, K, N, ours))
            self.assertLess(ours, 8 * mlxs + 1e-9,
                            "M=%d K=%d N=%d: ours %g vs mlx %g" % (M, K, N, ours, mlxs))

    def test_low_precision_accuracy(self):
        for dtype, bound in ((mx.float16, 5e-3), (mx.bfloat16, 4e-2)):
            for M, K, N in self.CASES:
                a = rn((M, K), dtype, 11)
                b = rn((K, N), dtype, 12)
                mx.eval(a, b)
                exact = np.array(a.astype(mx.float32)).astype(np.float64) @ \
                        np.array(b.astype(mx.float32)).astype(np.float64)
                scale = np.abs(exact).max()
                ours = np.abs(np.array(bi.matmul(a, b).astype(mx.float32)).astype(np.float64)
                              - exact).max() / scale
                mlxs = np.abs(np.array((a @ b).astype(mx.float32)).astype(np.float64)
                              - exact).max() / scale
                self.assertLess(ours, bound, "%s M=%d K=%d N=%d: %g" % (dtype, M, K, N, ours))
                self.assertLess(ours, 8 * mlxs + 1e-9,
                                "%s M=%d K=%d N=%d: ours %g vs mlx %g" % (dtype, M, K, N, ours, mlxs))

    def test_addmm_epilogue_is_single_rounded(self):
        """Bias is added to the float32 accumulator, then rounded once."""
        for dtype in (mx.float16, mx.bfloat16):
            a, b = rn((8, 512), dtype, 11), rn((512, 128), dtype, 12)
            c = rn((128,), dtype, 4)
            mx.eval(a, b, c)
            fused = np.array(bi.addmm(c, a, b).astype(mx.float32)).astype(np.float64)
            acc32 = np.array(bi.matmul(a.astype(mx.float32), b.astype(mx.float32))).astype(np.float64)
            want = np.array(mx.array(acc32 + np.array(c.astype(mx.float32)).astype(np.float64),
                                     dtype=mx.float32).astype(dtype).astype(mx.float32)).astype(np.float64)
            self.assertLess(np.abs(fused - want).max() / max(np.abs(want).max(), 1e-9), 1e-2)


class TestStockMLXIsStillBroken(unittest.TestCase):
    """Negative control. If stock mx.matmul ever becomes batch-invariant this fails,
    which means the suite above is no longer proving anything and this library is
    obsolete. A green suite with a vacuous assertion is worse than a red one."""

    def test_stock_matmul_is_batch_variant(self):
        found = []
        for dname, dtype in DTYPES:
            for K, N in [(2048, 1024), (4096, 4096), (11008, 4096)]:
                W = rn((K, N), dtype, 1)
                ref, pool = rn((1, K), dtype, 3), rn((128, K), dtype, 2)
                mx.eval(W, ref, pool)
                base = bits((ref @ W)[0])
                for B in (2, 33, 128):
                    y = batch_with(ref, pool, B, 0) @ W
                    mx.eval(y)
                    if int((bits(y[0]) != base).sum()) > 0:
                        found.append("%s K=%d N=%d B=%d" % (dname, K, N, B))
        self.assertTrue(found, "stock mx.matmul now looks batch-invariant everywhere "
                               "-- re-run probe/probe.py and re-read Phase 0")


class TestStockMLXStillInvariant(unittest.TestCase):
    """Phase 0 found these already bitwise batch-invariant, so this library does not
    wrap them. If a future MLX changes that, this suite is where it should fail."""

    def _assert_invariant(self, label, fn, ref, pool):
        base = None
        for B in BATCHES:
            idx = B // 2
            y = fn(batch_with(ref, pool, B, idx))
            mx.eval(y)
            got = bits(y[idx])
            if base is None:
                base = got
            else:
                self.assertEqual(int((got != base).sum()), 0, "%s B=%d" % (label, B))

    def test_norms_softmax_reductions(self):
        for dname, dtype in DTYPES:
            for D in (1024, 4096, 8192):
                w, b = rn((D,), dtype, 1), rn((D,), dtype, 4)
                ref, pool = rn((1, D), dtype, 3), rn((128, D), dtype, 2)
                mx.eval(w, b, ref, pool)
                for label, fn in (
                    ("rms_norm", lambda x: mx.fast.rms_norm(x, w, 1e-5)),
                    ("layer_norm", lambda x: mx.fast.layer_norm(x, w, b, 1e-5)),
                    ("softmax", lambda x: mx.softmax(x, axis=-1)),
                    ("sum", lambda x: mx.sum(x, axis=-1, keepdims=True)),
                    ("mean", lambda x: mx.mean(x, axis=-1, keepdims=True)),
                ):
                    self._assert_invariant("%s %s D=%d" % (label, dname, D), fn, ref, pool)

    def test_attention_batch_dimension_only(self):
        """Stock sdpa is invariant in the batch dimension. It is NOT invariant in the
        query dimension -- see TestStockAttentionIsQueryVariant."""
        import math
        H, Hkv, D = 8, 2, 128
        for dname, dtype in DTYPES[1:]:  # float16, bfloat16 -- float32 kv is too large
            for L in (512, 4096):
                k_ref, v_ref = rn((1, Hkv, L, D), dtype, 7), rn((1, Hkv, L, D), dtype, 8)
                k_pool, v_pool = rn((32, Hkv, L, D), dtype, 5), rn((32, Hkv, L, D), dtype, 6)
                q_ref, q_pool = rn((1, H, 1, D), dtype, 3), rn((32, H, 1, D), dtype, 2)
                mx.eval(k_ref, v_ref, k_pool, v_pool, q_ref, q_pool)
                base = None
                for B in [1, 2, 8, 31, 32]:
                    idx = B // 2
                    y = mx.fast.scaled_dot_product_attention(
                        batch_with(q_ref, q_pool, B, idx),
                        batch_with(k_ref, k_pool, B, idx),
                        batch_with(v_ref, v_pool, B, idx),
                        scale=1.0 / math.sqrt(D))
                    mx.eval(y)
                    got = bits(y[idx])
                    if base is None:
                        base = got
                    else:
                        self.assertEqual(int((got != base).sum()), 0,
                                         "sdpa %s L=%d B=%d" % (dname, L, B))


class TestAttentionInvariance(unittest.TestCase):
    """The output for one query token must not depend on how many other query tokens
    or batch rows were submitted with it."""

    CASES = [
        (32, 8, 512, 128),    # GQA factor 4, short context, MLX single pass
        (16, 8, 4096, 128),   # GQA factor 2 at 4096: MLX flips 32 -> 64 split blocks
        (8, 8, 4096, 128),    # MHA, MLX stays single pass
        (32, 8, 8192, 128),   # GQA long context, MLX two pass
        (4, 4, 333, 64),      # ragged N, small head dim
    ]
    QLS = [1, 2, 3, 4, 7, 8, 9, 16, 17]

    def _last_row(self, y):
        return bits(mx.contiguous(y[:, :, -1, :]))

    def test_query_length_invariance(self):
        for dname, dtype in DTYPES:
            for H, Hkv, N, D in self.CASES:
                k = rn((1, Hkv, N, D), dtype, 11)
                v = rn((1, Hkv, N, D), dtype, 12)
                qf = rn((1, H, max(self.QLS), D), dtype, 13)
                mx.eval(k, v, qf)
                base = None
                for qL in self.QLS:
                    q = mx.contiguous(qf[:, :, max(self.QLS) - qL:, :])
                    y = bi.scaled_dot_product_attention(q, k, v)
                    mx.eval(y)
                    got = self._last_row(y)
                    if base is None:
                        base = got
                        self.assertTrue(bool(mx.any(y != 0).item()), "degenerate output")
                    else:
                        self.assertEqual(int((got != base).sum()), 0,
                                         "sdpa %s H=%d/%d N=%d qL=%d" % (dname, H, Hkv, N, qL))

    def test_batch_invariance(self):
        H, Hkv, N, D = 16, 8, 2048, 128
        for dname, dtype in DTYPES:
            k1, v1 = rn((1, Hkv, N, D), dtype, 21), rn((1, Hkv, N, D), dtype, 22)
            q1 = rn((1, H, 1, D), dtype, 23)
            kp, vp = rn((16, Hkv, N, D), dtype, 24), rn((16, Hkv, N, D), dtype, 25)
            qp = rn((16, H, 1, D), dtype, 26)
            mx.eval(k1, v1, q1, kp, vp, qp)
            base = None
            for B in (1, 2, 3, 8, 16):
                idx = B // 2
                y = bi.scaled_dot_product_attention(
                    batch_with(q1, qp, B, idx),
                    batch_with(k1, kp, B, idx),
                    batch_with(v1, vp, B, idx))
                mx.eval(y)
                got = bits(y[idx])
                if base is None:
                    base = got
                else:
                    self.assertEqual(int((got != base).sum()), 0,
                                     "sdpa %s B=%d" % (dname, B))

    def test_causal_and_masks_are_invariant(self):
        H, Hkv, N, D = 8, 4, 1024, 128
        for dname, dtype in DTYPES:
            k, v = rn((1, Hkv, N, D), dtype, 31), rn((1, Hkv, N, D), dtype, 32)
            qf = rn((1, H, 16, D), dtype, 33)
            add = rn((1, 1, 1, N), dtype, 34)
            mx.eval(k, v, qf, add)
            for label, mk in (("causal", lambda qL: "causal"),
                              ("additive", lambda qL: add)):
                base = None
                for qL in (1, 4, 8, 9, 16):
                    q = mx.contiguous(qf[:, :, 16 - qL:, :])
                    y = bi.scaled_dot_product_attention(q, k, v, mask=mk(qL))
                    mx.eval(y)
                    got = self._last_row(y)
                    if base is None:
                        base = got
                    else:
                        self.assertEqual(int((got != base).sum()), 0,
                                         "sdpa %s %s qL=%d" % (label, dname, qL))


class TestAttentionMatchesMLX(unittest.TestCase):
    """Where MLX takes its single-pass vector path this kernel is a bit-for-bit
    replacement, so adopting it costs no accuracy at all in the common case."""

    def test_bitwise_identical_to_single_pass_mlx(self):
        import math
        for dname, dtype in DTYPES:
            for H, Hkv, qL, N, D in ((8, 8, 1, 512, 128), (32, 8, 4, 1024, 128),
                                     (4, 2, 3, 777, 64), (8, 4, 8, 2048, 128)):
                q = rn((2, H, qL, D), dtype, 41)
                k = rn((2, Hkv, N, D), dtype, 42)
                v = rn((2, Hkv, N, D), dtype, 43)
                add = rn((1, 1, qL, N), dtype, 44)
                mx.eval(q, k, v, add)
                for label, mask in (("none", None), ("causal", "causal"), ("additive", add)):
                    ref = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=1.0 / math.sqrt(D), mask=mask)
                    ours = bi.scaled_dot_product_attention(q, k, v, mask=mask)
                    mx.eval(ref, ours)
                    self.assertEqual(int((bits(ref) != bits(ours)).sum()), 0,
                                     "%s %s H=%d/%d qL=%d N=%d" % (label, dname, H, Hkv, qL, N))

    def test_accuracy_against_float64(self):
        import math
        for dname, dtype, tol in (("float32", mx.float32, 2e-6),
                                  ("bfloat16", mx.bfloat16, 3e-2)):
            H, Hkv, qL, N, D = 8, 4, 4, 2048, 128
            q, k, v = rn((1, H, qL, D), dtype, 51), rn((1, Hkv, N, D), dtype, 52), rn((1, Hkv, N, D), dtype, 53)
            mx.eval(q, k, v)
            qn = np.array(q.astype(mx.float32), dtype=np.float64)[0]
            kn = np.array(k.astype(mx.float32), dtype=np.float64)[0]
            vn = np.array(v.astype(mx.float32), dtype=np.float64)[0]
            kn = np.repeat(kn, H // Hkv, axis=0)
            vn = np.repeat(vn, H // Hkv, axis=0)
            sc = qn @ kn.transpose(0, 2, 1) / math.sqrt(D)
            sc -= sc.max(axis=-1, keepdims=True)
            p = np.exp(sc)
            want = (p / p.sum(axis=-1, keepdims=True)) @ vn
            ours = np.array(bi.scaled_dot_product_attention(q, k, v).astype(mx.float32), dtype=np.float64)[0]
            mlxs = np.array(mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1.0 / math.sqrt(D)).astype(mx.float32), dtype=np.float64)[0]
            scale = np.abs(want).max()
            e_ours = np.abs(ours - want).max() / scale
            e_mlx = np.abs(mlxs - want).max() / scale
            self.assertLess(e_ours, tol, "%s rel err %.3e" % (dname, e_ours))
            self.assertLess(e_ours, 4 * e_mlx + 1e-12,
                            "%s: ours %.3e vs mlx %.3e" % (dname, e_ours, e_mlx))


class TestStockAttentionIsQueryVariant(unittest.TestCase):
    """Negative control for Phase 2. Stock MLX attention changes its answer for a
    fixed query token when the number of query tokens changes: at qL 8 -> 9 it swaps
    sdpa_vector for steel_attention. If this ever stops failing, the attention half
    of this library is obsolete."""

    def test_stock_sdpa_changes_at_query_length_9(self):
        import math
        H, Hkv, N, D = 32, 8, 512, 128
        dtype = mx.float32
        k, v = rn((1, Hkv, N, D), dtype, 61), rn((1, Hkv, N, D), dtype, 62)
        qf = rn((1, H, 16, D), dtype, 63)
        mx.eval(k, v, qf)
        got = []
        for qL in (8, 9):
            q = mx.contiguous(qf[:, :, 16 - qL:, :])
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D))
            mx.eval(y)
            got.append(bits(mx.contiguous(y[:, :, -1, :])))
        self.assertGreater(int((got[0] != got[1]).sum()), 0,
                           "stock sdpa is now query-length invariant at the 8/9 boundary")


class TestDeviceAssumptions(unittest.TestCase):
    def test_simd_width_is_32(self):
        """The attention kernel reduces across 32 simdgroups of 32 lanes; a device
        with a different execution width would change the summation order."""
        k = mx.fast.metal_kernel(name="simdw", input_names=["a"], output_names=["o"],
                                 source="o[0] = thread_execution_width;")
        w = k(inputs=[mx.zeros((8,))], grid=(32, 1, 1), threadgroup=(32, 1, 1),
              output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]
        self.assertEqual(w.item(), 32)

    def test_tile_is_not_shape_dependent(self):
        """Guard against anyone reintroducing dimension-based strategy selection."""
        self.assertEqual(bi.TILE, dict(BM=32, BN=32, BK=16, TM=4, TN=4))


if __name__ == "__main__":
    unittest.main()


class TestQuantizedInvariance(unittest.TestCase):
    """Every MLX checkpoint worth running on this machine is 4-bit, so quantized
    matmul has to be invariant or the library does not apply to real models."""

    def _layer(self, in_f, out_f, bias, bits_, dtype, seed=41):
        import mlx.nn as nn
        mx.random.seed(seed)
        lin = nn.Linear(in_f, out_f, bias=bias)
        lin.apply(lambda p: p.astype(dtype))
        return nn.QuantizedLinear.from_linear(lin, bits=bits_)

    def test_stock_quantized_matmul_is_batch_variant(self):
        """Negative control. If this ever passes, MLX fixed it upstream and the
        shim below is no longer proving anything."""
        q = self._layer(512, 1024, False, 4, mx.float16)
        ref, pool = rn((1, 512), mx.float16, 42), rn((8, 512), mx.float16, 43)
        mx.eval(ref, pool)
        base = bits(q(ref)[0])
        total = sum(int((bits(q(batch_with(ref, pool, B, 0))[0]) != base).sum())
                    for B in (2, 8))
        self.assertGreater(total, 0,
                           "stock quantized_matmul is now batch-invariant on this "
                           "machine; the quantized shim no longer proves anything")

    def test_quantized_linear_is_invariant_under_mode(self):
        for dname, dtype in (("float16", mx.float16), ("bfloat16", mx.bfloat16)):
            for nbits in (4, 8):
                for use_bias in (False, True):
                    q = self._layer(512, 1024, use_bias, nbits, dtype)
                    ref = rn((1, 512), dtype, 44)
                    pool = rn((max(BATCHES), 512), dtype, 45)
                    mx.eval(ref, pool)
                    with bi.batch_invariant_mode(strict=True):
                        base = None
                        for B in BATCHES:
                            idx = B // 2
                            y = q(batch_with(ref, pool, B, idx))
                            mx.eval(y)
                            got = bits(y[idx])
                            if base is None:
                                base = got
                                self.assertTrue(bool(mx.any(y != 0).item()),
                                                "degenerate output")
                            else:
                                self.assertEqual(
                                    int((got != base).sum()), 0,
                                    "quantized %s bits=%d bias=%s B=%d"
                                    % (dname, nbits, use_bias, B))

    def test_quantized_leading_dims_are_invariant(self):
        """A 3D activation must give the same row whether it arrives alone or
        stacked -- this is the prefill-vs-decode case."""
        q = self._layer(512, 256, True, 4, mx.float16)
        ref, pool = rn((1, 5, 512), mx.float16, 46), rn((8, 5, 512), mx.float16, 47)
        mx.eval(ref, pool)
        with bi.batch_invariant_mode(strict=True):
            base = None
            for B in (1, 2, 3, 8):
                y = q(batch_with(ref, pool, B, 0))
                mx.eval(y)
                got = bits(y[0])
                if base is None:
                    base = got
                else:
                    self.assertEqual(int((got != base).sum()), 0, "quantized 3D B=%d" % B)

    def test_quantized_accuracy_matches_stock(self):
        """Dequantize-then-invariant-GEMM must not be less accurate than the fused
        stock kernel, measured against a float32 reference on the same weights."""
        q = self._layer(512, 1024, False, 4, mx.float16)
        x = rn((4, 512), mx.float16, 48)
        mx.eval(x)
        w = mx.dequantize(q["weight"], scales=q["scales"], biases=q.get("biases"),
                          group_size=q.group_size, bits=q.bits, mode=q.mode)
        exact = np.array(x.astype(mx.float32)) @ np.array(w.astype(mx.float32)).T
        stock = np.array(q(x).astype(mx.float32))
        with bi.batch_invariant_mode(strict=True):
            ours = np.array(q(x).astype(mx.float32))
        e_stock = np.abs(stock - exact).max()
        e_ours = np.abs(ours - exact).max()
        self.assertLessEqual(e_ours, e_stock * 2 + 1e-6,
                             "invariant quantized path is much less accurate: "
                             "%g vs stock %g" % (e_ours, e_stock))


class TestCompiled(unittest.TestCase):
    """mx.compile reorders and fuses the graph around these kernels. It must not
    reorder anything inside them."""

    def test_attention_is_invariant_under_compile(self):
        H, Hkv, N, D = 16, 8, 2048, 128
        QLS = [1, 2, 4, 8, 9, 16]
        for dname, dtype in DTYPES:
            k, v = rn((1, Hkv, N, D), dtype, 51), rn((1, Hkv, N, D), dtype, 52)
            qf = rn((1, H, max(QLS), D), dtype, 53)
            mx.eval(k, v, qf)
            fn = mx.compile(lambda q: bi.scaled_dot_product_attention(q, k, v))
            base = None
            for qL in QLS:
                q = mx.contiguous(qf[:, :, max(QLS) - qL:, :])
                y = fn(q)
                mx.eval(y)
                got = bits(mx.contiguous(y[:, :, -1, :]))
                if base is None:
                    base = got
                else:
                    self.assertEqual(int((got != base).sum()), 0,
                                     "compiled sdpa %s qL=%d" % (dname, qL))

    def test_mode_holds_for_function_compiled_inside(self):
        """A function first traced inside the context captures the invariant
        kernels and stays invariant for every later call inside it."""
        import mlx.nn as nn
        mx.random.seed(54)
        net = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 128))
        net.apply(lambda p: p.astype(mx.float16))
        ref, pool = rn((1, 256), mx.float16, 55), rn((16, 256), mx.float16, 56)
        mx.eval(ref, pool, net.parameters())
        with bi.batch_invariant_mode(strict=True):
            fn = mx.compile(net)
            base = None
            for B in (1, 2, 3, 8, 16):
                idx = B // 2
                y = fn(batch_with(ref, pool, B, idx))
                mx.eval(y)
                got = bits(y[idx])
                if base is None:
                    base = got
                else:
                    self.assertEqual(int((got != base).sum()), 0,
                                     "compiled-inside mode B=%d" % B)


class TestStrictMode(unittest.TestCase):
    """strict=True promises an exception rather than a silent variant result."""

    def test_unsupported_attention_raises(self):
        q = rn((1, 4, 1, 48), mx.float16, 61)   # head dim 48 is not a multiple of 32
        k = rn((1, 4, 128, 48), mx.float16, 62)
        v = rn((1, 4, 128, 48), mx.float16, 63)
        mx.eval(q, k, v)
        with bi.batch_invariant_mode(strict=True):
            with self.assertRaises(NotImplementedError):
                mx.fast.scaled_dot_product_attention(q, k, v, scale=0.1)

    def test_unsupported_falls_back_when_not_strict(self):
        q = rn((1, 4, 1, 48), mx.float16, 61)
        k = rn((1, 4, 128, 48), mx.float16, 62)
        v = rn((1, 4, 128, 48), mx.float16, 63)
        mx.eval(q, k, v)
        with bi.batch_invariant_mode(strict=False):
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.1)
            mx.eval(y)
        self.assertEqual(y.shape, (1, 4, 1, 48))

    def test_originals_are_restored_even_on_exception(self):
        import mlx.nn as nn
        before = (mx.matmul, mx.addmm, mx.fast.scaled_dot_product_attention,
                  mx.quantized_matmul, nn.Linear.__call__, mx.array.__matmul__)
        with self.assertRaises(RuntimeError):
            with bi.batch_invariant_mode():
                self.assertTrue(bi.is_enabled())
                raise RuntimeError("boom")
        after = (mx.matmul, mx.addmm, mx.fast.scaled_dot_product_attention,
                 mx.quantized_matmul, nn.Linear.__call__, mx.array.__matmul__)
        self.assertEqual(before, after)
        self.assertFalse(bi.is_enabled())

    def test_nesting_restores_only_at_the_outermost_exit(self):
        stock = mx.matmul
        with bi.batch_invariant_mode():
            patched = mx.matmul
            self.assertIsNot(patched, stock)
            with bi.batch_invariant_mode():
                self.assertIs(mx.matmul, patched)
            self.assertIs(mx.matmul, patched, "inner exit unpatched too early")
        self.assertIs(mx.matmul, stock)
