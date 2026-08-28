"""Phase 3: what invariance costs on this machine.

Baseline is always stock MLX on the same shapes, dtypes and thermal state.  A and B
are interleaved within every repeat so that thermal drift lands on both sides
equally, and the reported number is the median of the per-repeat ratios rather than
a ratio of medians.

  .venv/bin/python bench/bench.py micro
  .venv/bin/python bench/bench.py e2e
"""

import argparse
import contextlib
import json
import math
import statistics
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, ".")
import mlx_batch_invariant as bi

# Qwen3-1.7B-shaped block, 16 layers and a 32k vocab so the whole thing plus both
# code paths fits comfortably in 16 GB of unified memory.
CFG = dict(hidden=2048, inter=6144, heads=16, kv_heads=8, head_dim=128,
           layers=16, vocab=32000)


def ab(a_fn, b_fn, repeats=9, inner=3):
    """Interleaved A/B. Returns (a_seconds, b_seconds, median ratio b/a)."""
    mx.eval(a_fn())
    mx.eval(b_fn())
    a_s, b_s, ratios = [], [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            mx.eval(a_fn())
        ta = (time.perf_counter() - t0) / inner
        t0 = time.perf_counter()
        for _ in range(inner):
            mx.eval(b_fn())
        tb = (time.perf_counter() - t0) / inner
        a_s.append(ta)
        b_s.append(tb)
        ratios.append(tb / ta)
    return statistics.median(a_s), statistics.median(b_s), statistics.median(ratios)


def micro(dtype, out):
    h, i, hd = CFG["hidden"], CFG["inter"], CFG["head_dim"]
    print("\n### GEMM  (stock mx.matmul vs mlx_batch_invariant.linear)\n")
    print("| shape (M x K x N) | role | stock ms | invariant ms | ratio |")
    print("|---|---|---:|---:|---:|")
    shapes = [
        (1, h, h, "o_proj, decode"),
        (1, h, 2 * i, "gate+up, decode"),
        (1, i, h, "down_proj, decode"),
        (8, h, 2 * i, "gate+up, batch 8"),
        (32, h, 2 * i, "gate+up, batch 32"),
        (128, h, 2 * i, "gate+up, batch 128"),
        (512, h, 2 * i, "gate+up, prefill 512"),
        (2048, h, h, "o_proj, prefill 2048"),
    ]
    for M, K, N, role in shapes:
        x = mx.random.normal((M, K)).astype(dtype)
        w = mx.random.normal((N, K)).astype(dtype)
        mx.eval(x, w)
        wt = mx.contiguous(w.T)
        mx.eval(wt)
        a, b, r = ab(lambda: mx.matmul(x, wt), lambda: bi.linear(x, w))
        print("| %d x %d x %d | %s | %.3f | %.3f | %.2fx |"
              % (M, K, N, role, a * 1e3, b * 1e3, r))
        out.append(dict(kind="gemm", dtype=str(dtype), M=M, K=K, N=N, role=role,
                        stock_ms=a * 1e3, bi_ms=b * 1e3, ratio=r))

    print("\n### Attention  (stock mx.fast.scaled_dot_product_attention vs ours)\n")
    print("| B x H/Hkv x qL x N | role | stock ms | invariant ms | ratio |")
    print("|---|---|---:|---:|---:|")
    H, Hkv = CFG["heads"], CFG["kv_heads"]
    cases = [
        (1, 1, 512, "decode, 512 ctx"),
        (1, 1, 2048, "decode, 2k ctx"),
        (1, 1, 8192, "decode, 8k ctx"),
        (1, 1, 32768, "decode, 32k ctx"),
        (8, 1, 2048, "decode, batch 8"),
        (32, 1, 2048, "decode, batch 32"),
        (1, 8, 2048, "chunked prefill, 8 queries"),
        (1, 128, 128, "prefill 128"),
        (1, 512, 512, "prefill 512"),
    ]
    scale = 1.0 / math.sqrt(hd)
    for B, qL, N, role in cases:
        q = mx.random.normal((B, H, qL, hd)).astype(dtype)
        k = mx.random.normal((B, Hkv, N, hd)).astype(dtype)
        v = mx.random.normal((B, Hkv, N, hd)).astype(dtype)
        mx.eval(q, k, v)
        a, b, r = ab(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale),
                     lambda: bi.scaled_dot_product_attention(q, k, v, scale=scale))
        print("| %d x %d/%d x %d x %d | %s | %.3f | %.3f | %.2fx |"
              % (B, H, Hkv, qL, N, role, a * 1e3, b * 1e3, r))
        out.append(dict(kind="sdpa", dtype=str(dtype), B=B, qL=qL, N=N, role=role,
                        stock_ms=a * 1e3, bi_ms=b * 1e3, ratio=r))


def build_model(dtype):
    import mlx.nn as nn

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            h, i, H, Hkv, hd = (CFG["hidden"], CFG["inter"], CFG["heads"],
                                CFG["kv_heads"], CFG["head_dim"])
            s.H, s.Hkv, s.hd = H, Hkv, hd
            s.q = nn.Linear(h, H * hd, bias=False)
            s.k = nn.Linear(h, Hkv * hd, bias=False)
            s.v = nn.Linear(h, Hkv * hd, bias=False)
            s.o = nn.Linear(H * hd, h, bias=False)
            s.gate = nn.Linear(h, i, bias=False)
            s.up = nn.Linear(h, i, bias=False)
            s.down = nn.Linear(i, h, bias=False)
            s.n1 = nn.RMSNorm(h)
            s.n2 = nn.RMSNorm(h)

        def __call__(s, x, cache):
            B, L, _ = x.shape
            y = s.n1(x)
            q = s.q(y).reshape(B, L, s.H, s.hd).transpose(0, 2, 1, 3)
            k = s.k(y).reshape(B, L, s.Hkv, s.hd).transpose(0, 2, 1, 3)
            v = s.v(y).reshape(B, L, s.Hkv, s.hd).transpose(0, 2, 1, 3)
            if cache is not None:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
            a = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1.0 / math.sqrt(s.hd),
                mask="causal" if L > 1 else None)
            x = x + s.o(a.transpose(0, 2, 1, 3).reshape(B, L, s.H * s.hd))
            y = s.n2(x)
            return x + s.down(nn.silu(s.gate(y)) * s.up(y)), (k, v)

    class Model(nn.Module):
        def __init__(s):
            super().__init__()
            s.embed = nn.Embedding(CFG["vocab"], CFG["hidden"])
            s.blocks = [Block() for _ in range(CFG["layers"])]
            s.norm = nn.RMSNorm(CFG["hidden"])
            s.head = nn.Linear(CFG["hidden"], CFG["vocab"], bias=False)

        def __call__(s, ids, caches):
            x = s.embed(ids)
            new = []
            for blk, c in zip(s.blocks, caches):
                x, c = blk(x, c)
                new.append(c)
            return s.head(s.norm(x)), new

    m = Model()
    m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m


def e2e(dtype, out, prompt=256, gen=32, batch=1, repeats=3):
    """Prefill and decode are timed separately -- they stress different kernels and
    the invariant GEMM behaves very differently at M=1 and M=256."""
    m = build_model(dtype)
    nlayer = CFG["layers"]
    nparam = sum(v.size for _, v in tree_flatten(m.parameters()))

    def measure(invariant):
        ctx = bi.batch_invariant_mode() if invariant else contextlib.nullcontext()
        with ctx:
            ids = mx.random.randint(0, CFG["vocab"], (batch, prompt))
            logits, caches = m(ids, [None] * nlayer)   # warm the jit and the caches
            tok = mx.argmax(logits[:, -1:], axis=-1)
            mx.eval(tok, [c for kv in caches for c in kv])

            t0 = time.perf_counter()
            logits, caches = m(ids, [None] * nlayer)
            tok = mx.argmax(logits[:, -1:], axis=-1)
            mx.eval(tok, [c for kv in caches for c in kv])
            tp = time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in range(gen):
                logits, caches = m(tok, caches)
                tok = mx.argmax(logits[:, -1:], axis=-1)
                mx.eval(tok, [c for kv in caches for c in kv])
            td = time.perf_counter() - t0
        mx.clear_cache()
        return tp, td

    pre, dec = {}, {}
    for _ in range(repeats):
        for invariant in (False, True):   # interleaved: both sides see the same drift
            tp, td = measure(invariant)
            pre.setdefault(invariant, []).append(tp)
            dec.setdefault(invariant, []).append(td)

    print("\n### End to end  (%d layers, %s, %.2fB params, batch %d)\n"
          % (nlayer, str(dtype).rsplit(".", 1)[-1], nparam / 1e9, batch))
    print("| phase | stock MLX | batch invariant | ratio | throughput cost |")
    print("|---|---:|---:|---:|---:|")
    for name, d, ntok in (("prefill %d tok" % prompt, pre, batch * prompt),
                          ("decode %d tok" % gen, dec, batch * gen)):
        a, b = statistics.median(d[False]), statistics.median(d[True])
        r = b / a
        print("| %s | %.1f tok/s | %.1f tok/s | %.2fx | %.1f%% |"
              % (name, ntok / a, ntok / b, r, 100 * (1 - 1 / r)))
        out.append(dict(kind="e2e", dtype=str(dtype), phase=name, batch=batch,
                        layers=nlayer, params=nparam, stock_s=a, bi_s=b, ratio=r,
                        cost_pct=100 * (1 - 1 / r)))


def determinism(dtype, out):
    """The payoff: same prompt, different batch, same bits."""
    import numpy as np

    m = build_model(dtype)
    nlayer, prompt = CFG["layers"], 64
    ids = mx.random.randint(0, CFG["vocab"], (1, prompt))
    pool = mx.random.randint(0, CFG["vocab"], (7, prompt))
    mx.eval(ids, pool)

    def logits_for(B, invariant):
        x = ids if B == 1 else mx.concatenate([ids, pool[:B - 1]])
        ctx = bi.batch_invariant_mode() if invariant else contextlib.nullcontext()
        with ctx:
            y, _ = m(x, [None] * nlayer)
            mx.eval(y)
        return np.array(mx.view(mx.contiguous(y[0, -1]), mx.uint16))

    print("\n### Determinism  (%d-layer %s, 64-token prompt at batch position 0)\n"
          % (nlayer, str(dtype).rsplit(".", 1)[-1]))
    print("| path | batch | logits differing from batch 1 |")
    print("|---|---:|---:|")
    for label, invariant in (("stock MLX", False), ("batch invariant", True)):
        base = logits_for(1, invariant)
        for B in (2, 4, 8):
            d = int((logits_for(B, invariant) != base).sum())
            print("| %s | %d | %d / %d |" % (label, B, d, base.size))
            out.append(dict(kind="determinism", path=label, batch=B, differing=d,
                            total=int(base.size)))
        mx.clear_cache()


def quant(dtype, out):
    """Stock fused quantized_matmul vs the fused invariant quantized kernel.

    Both unpack the weight inside the K loop; what differs is the tile. Decode is
    the worst cell because a fixed 32-row tile dequantises a slab to use one row.
    """
    import mlx.nn as nn

    h, i = CFG["hidden"], CFG["inter"]
    print("\n### Quantized linear  (stock int4 quantized_matmul vs invariant)\n")
    print("| shape (M x K x N) | role | stock ms | invariant ms | ratio |")
    print("|---|---|---:|---:|---:|")
    for M, K, N, role in [
        (1, h, h, "o_proj, decode"),
        (1, h, 2 * i, "gate+up, decode"),
        (1, i, h, "down_proj, decode"),
        (256, h, h, "o_proj, prefill"),
        (256, h, 2 * i, "gate+up, prefill"),
    ]:
        lin = nn.Linear(K, N, bias=False)
        lin.apply(lambda a: a.astype(dtype))
        q = nn.QuantizedLinear.from_linear(lin, bits=4)
        x = mx.random.normal((M, K)).astype(dtype)
        mx.eval(x, q.parameters())

        def stock_fn():
            return q(x)

        def inv_fn():
            with bi.batch_invariant_mode():
                return q(x)

        ta, tb, r = ab(stock_fn, inv_fn)
        print("| %d x %d x %d | %s | %.3f | %.3f | %.2fx |"
              % (M, K, N, role, ta * 1e3, tb * 1e3, r))
        out.append(dict(kind="quant", M=M, K=K, N=N, role=role,
                        stock_ms=ta * 1e3, invariant_ms=tb * 1e3, ratio=r))
        mx.clear_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["micro", "quant", "e2e", "determinism", "all"])
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--json", default=None)
    a = p.parse_args()
    dtype = dict(float32=mx.float32, float16=mx.float16, bfloat16=mx.bfloat16)[a.dtype]

    print("mlx %s | %s | dtype %s"
          % (mx.__version__, mx.device_info().get("architecture", "?"), a.dtype))
    out = []
    if a.what in ("micro", "all"):
        micro(dtype, out)
    if a.what in ("quant", "all"):
        quant(dtype, out)
    if a.what in ("e2e", "all"):
        e2e(dtype, out)
    if a.what in ("determinism", "all"):
        determinism(dtype, out)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
