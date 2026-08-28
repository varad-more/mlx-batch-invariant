#!/usr/bin/env python3
"""Phase 0 kill test for mlx-batch-invariant.

Question: on this machine, does MLX return bitwise-identical output for a fixed
reference row when that row is embedded in batches of different sizes?

No timing here. Correctness gate only. Comparison is on integer bit patterns via
mx.view -- allclose/== on floats are deliberately absent.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import mlx.core as mx

BATCHES = [1, 2, 3, 5, 8, 16, 31, 32, 33, 64, 128]
DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
MODES = ["eager", "graph", "compiled"]
BUDGET = 3 << 30  # bytes of live batched input per case

_UINT = {mx.float32: (mx.uint32, 32), mx.float16: (mx.uint16, 16), mx.bfloat16: (mx.uint16, 16)}


# --------------------------------------------------------------------------- bits

def bit_pattern(a):
    """Contiguous flat array -> (numpy uint bit patterns, bit width)."""
    u, n = _UINT[a.dtype]
    return np.array(mx.view(mx.contiguous(a).reshape(-1), u)), n


def ulp_distance(x, y, n):
    """IEEE754 ULP distance between two uint bit-pattern arrays of width n."""
    top = 1 << (n - 1)
    xi, yi = x.astype(np.int64), y.astype(np.int64)
    xo = np.where(xi >= top, top - xi, xi)
    yo = np.where(yi >= top, top - yi, yi)
    return np.abs(xo - yo)


# ------------------------------------------------------------------- kernel model
# Mirrors the dispatch conditions read out of mlx 0.32.0 for applegpu_g16g
# (devc == 'g', is_nax_available() == False). These are predictions, not probes.

def _npo2(n):
    return 0 if n <= 0 else (1 << (n - 1).bit_length() if n & (n - 1) else n)


def predict_matmul(M, N, K):
    if min(M, N) == 1:
        return "gemv"
    tm, tn, tk = -(-M // 16), -(-N // 16), K // 16
    if tm * tn <= 1024 and tk >= 8 and K >= max(M, N):  # threshold 1024 for devc=='g'
        parts = min(max(2, _npo2(tk // (-(-M // 32) * -(-N // 32)))), 32)
        return "splitk[%d]" % parts
    return "steel"


def predict_rowreduce(n_rows, row_size):
    if row_size <= 64:
        return "row_reduce_small"
    if n_rows >= 32:
        return "row_reduce_simple"
    return "row_reduce_looped"


def predict_sdpa(H, Hkv, L):
    # devc=='g' kills the (devc=='d'||'s') && L>=1024 clause; only GQA @ L>=4096 left
    return "sdpa_vector_2pass" if (Hkv < H and L >= 4096) else "sdpa_vector"


def predict_norm(D, limit=4096):
    return "looped" if D > limit else "block"


# ------------------------------------------------------------------------- cases

def _rn(shape, dtype, seed):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(dtype)


def case_matmul(dtype, bcap, K, N, addmm):
    pool = (_rn((bcap, K), dtype, 2),)
    ref = (_rn((1, K), dtype, 3),)
    W = _rn((K, N), dtype, 1)
    if addmm:
        c = _rn((N,), dtype, 4)
        return dict(pool=pool, ref=ref, extras=(W, c),
                    fn=lambda x, W, c: mx.addmm(c, x, W),
                    predict=lambda B: predict_matmul(B, N, K))
    return dict(pool=pool, ref=ref, extras=(W,),
                fn=lambda x, W: x @ W,
                predict=lambda B: predict_matmul(B, N, K))


def case_norm(dtype, bcap, D, kind):
    pool = (_rn((bcap, D), dtype, 2),)
    ref = (_rn((1, D), dtype, 3),)
    if kind == "rms_norm":
        w = _rn((D,), dtype, 1)
        fn, extras = (lambda x, w: mx.fast.rms_norm(x, w, 1e-5)), (w,)
    elif kind == "layer_norm":
        w, b = _rn((D,), dtype, 1), _rn((D,), dtype, 4)
        fn, extras = (lambda x, w, b: mx.fast.layer_norm(x, w, b, 1e-5)), (w, b)
    else:
        fn, extras = (lambda x: mx.softmax(x, axis=-1)), ()
    return dict(pool=pool, ref=ref, extras=extras, fn=fn,
                predict=lambda B: predict_norm(D))


def case_reduce(dtype, bcap, R, kind):
    pool = (_rn((bcap, R), dtype, 2),)
    ref = (_rn((1, R), dtype, 3),)
    fn = (lambda x: mx.sum(x, axis=-1)) if kind == "sum" else (lambda x: mx.mean(x, axis=-1))
    return dict(pool=pool, ref=ref, extras=(), fn=fn,
                predict=lambda B: predict_rowreduce(B, R))


def case_sdpa(dtype, bcap, H, Hkv, L, D=128):
    pool = (_rn((bcap, H, 1, D), dtype, 2),
            _rn((bcap, Hkv, L, D), dtype, 5),
            _rn((bcap, Hkv, L, D), dtype, 6))
    ref = (_rn((1, H, 1, D), dtype, 3),
           _rn((1, Hkv, L, D), dtype, 7),
           _rn((1, Hkv, L, D), dtype, 8))
    scale = 1.0 / math.sqrt(D)
    fn = lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    return dict(pool=pool, ref=ref, extras=(), fn=fn,
                predict=lambda B: predict_sdpa(H, Hkv, L))


def all_specs():
    """(op, shape-dict, rowbytes(itemsize), build(dtype, bcap))"""
    out = []
    for op, addmm in (("matmul", False), ("addmm", True)):
        for N in (1024, 4096):
            for K in (512, 1024, 2048, 4096, 11008):
                out.append((op, dict(K=K, N=N), (lambda i, K=K: K * i),
                            lambda d, b, K=K, N=N, a=addmm: case_matmul(d, b, K, N, a)))
    for op in ("rms_norm", "layer_norm", "softmax"):
        for D in (1024, 4096, 8192):
            out.append((op, dict(D=D), (lambda i, D=D: D * i),
                        lambda d, b, D=D, k=op: case_norm(d, b, D, k)))
    for op in ("sum", "mean"):
        for R in (64, 128, 1024, 4096):
            out.append((op, dict(R=R), (lambda i, R=R: R * i),
                        lambda d, b, R=R, k=op: case_reduce(d, b, R, k)))
    for H, Hkv in ((8, 8), (8, 2)):
        for L in (128, 512, 2048, 4096, 8192):
            out.append(("sdpa", dict(H=H, Hkv=Hkv, L=L, D=128),
                        (lambda i, H=H, Hkv=Hkv, L=L: (H * 128 + 2 * Hkv * L * 128) * i),
                        lambda d, b, H=H, Hkv=Hkv, L=L: case_sdpa(d, b, H, Hkv, L)))
    return out


# ------------------------------------------------------------------------- runner

def assemble(pool, ref, B, idx):
    if B == 1:
        return tuple(ref)
    out = []
    for p, r in zip(pool, ref):
        parts = ([p[:idx]] if idx > 0 else []) + [r] + ([p[idx + 1:B]] if idx + 1 < B else [])
        out.append(mx.concatenate(parts, axis=0))
    return tuple(out)


def execute(fn, compiled, args, extras, mode):
    if mode == "compiled":
        return compiled(*args, *extras)
    y = fn(*args, *extras)
    if mode == "graph":
        # keep the op inside a larger graph at eval time so fusion sees context
        mx.eval(y, mx.sum(y.astype(mx.float32) ** 2))
    else:
        mx.eval(y)
    return y


def run_case(op, shape, rowbytes, build, dtype_name, sink):
    dtype = DTYPES[dtype_name]
    isz = 4 if dtype == mx.float32 else 2
    rb = rowbytes(isz)
    allowed = [b for b in BATCHES if b * rb <= BUDGET] or [1]
    bcap = max(allowed)

    c = build(dtype, bcap)
    mx.eval(*c["pool"], *c["ref"], *c["extras"])
    compiled = mx.compile(c["fn"])

    base = {}
    for B in BATCHES:
        if B > bcap:
            sink(dict(op=op, dtype=dtype_name, shape=shape, B=B, idx=0, mode="-",
                      pred="-", status="skipped_oom_guard"))
            continue
        for idx in sorted({0, B // 2}):
            x = assemble(c["pool"], c["ref"], B, idx)
            mx.eval(*x)
            for mode in MODES:
                y = execute(c["fn"], compiled, x, c["extras"], mode)
                mx.eval(y)
                pat, nbits = bit_pattern(y[idx])
                rec = dict(op=op, dtype=dtype_name, shape=shape, B=B, idx=idx,
                           mode=mode, pred=c["predict"](B), status="ok")
                if B == 1 and idx == 0:
                    base[mode] = pat
                    rec["ndiff"] = 0
                    rec["max_ulp"] = 0
                    # cross-mode agreement at the baseline itself
                    rec["mode_matches_eager"] = bool(np.array_equal(pat, base["eager"]))
                else:
                    d = pat != base[mode]
                    rec["ndiff"] = int(d.sum())
                    rec["n"] = int(pat.size)
                    if rec["ndiff"]:
                        u = ulp_distance(pat[d], base[mode][d], nbits)
                        rec["max_ulp"] = int(u.max())
                        rec["med_ulp"] = int(np.median(u))
                    else:
                        rec["max_ulp"] = 0
                sink(rec)
    del c, compiled
    mx.clear_cache()


def cmd_run(args):
    specs = all_specs()
    if args.op:
        specs = [s for s in specs if s[0] in args.op]
    dtypes = args.dtype or list(DTYPES)
    with open(args.out, "w") as f:
        def sink(rec):
            f.write(json.dumps(rec) + "\n")
        for dtype_name in dtypes:
            for op, shape, rowbytes, build in specs:
                print("%-10s %-9s %s" % (op, dtype_name, shape), flush=True)
                run_case(op, shape, rowbytes, build, dtype_name, sink)
                f.flush()
    print("wrote", args.out)



# ------------------------------------------------------------------ split-KV probe
# Batch sweeps hold context length fixed, so they cannot see the split-KV hazard.
# This forces MLX_SDPA_BLOCKS (mlx 0.32.0, scaled_dot_product_attention.cpp:477) to
# a range of values and asks whether the KV split count changes the answer bitwise.

_BLOCKS = ["default", "32", "64", "128", "256", "512", "1024"]


def _kv_digest(L, Hkv, H=8, B=4, D=128, dtype=mx.float16):
    q = _rn((B, H, 1, D), dtype, 2)
    k = _rn((B, Hkv, L, D), dtype, 5)
    v = _rn((B, Hkv, L, D), dtype, 6)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D))
    mx.eval(o)
    pat, _ = bit_pattern(o)
    return hashlib.sha256(pat.tobytes()).hexdigest()[:16]


def cmd_splitkv(args):
    if args.child:
        L, Hkv = map(int, args.child)
        print(_kv_digest(L, Hkv))
        return
    print("| Hkv | L | predicted | distinct results across forced block counts | default matches |")
    print("|---|---|---|---|---|")
    for Hkv in (8, 2):
        for L in (2048, 4096, 8192):
            digests = {}
            for b in _BLOCKS:
                env = dict(os.environ)
                env.pop("MLX_SDPA_BLOCKS", None)
                if b != "default":
                    env["MLX_SDPA_BLOCKS"] = b
                out = subprocess.run(
                    [sys.executable, __file__, "splitkv", "--child", str(L), str(Hkv)],
                    env=env, capture_output=True, text=True, check=True).stdout.strip()
                digests[b] = out
            uniq = len(set(digests.values()))
            same = [b for b in _BLOCKS[1:] if digests[b] == digests["default"]]
            print("| %d | %d | %s | %d | %s |" % (
                Hkv, L, predict_sdpa(8, Hkv, L), uniq,
                ", ".join(same) if same else "none"))



# ------------------------------------------------------- addmm epilogue diagnostic
# Low-precision addmm diverges far more than plain matmul. This isolates why: the
# fused axpby epilogue adds C to the fp32 accumulator, while the gemv path taken at
# batch 1 rounds to the output dtype first. Double rounding, not reduction order.

def cmd_epilogue(args):
    K, N = 2048, 1024
    print("| dtype | comparison | bits differing (of %d) |" % N)
    print("|---|---|---|")
    for name, dt in (("bfloat16", mx.bfloat16), ("float16", mx.float16), ("float32", mx.float32)):
        W, c = _rn((K, N), dt, 1), _rn((N,), dt, 4)
        r = _rn((1, K), dt, 3)
        x8 = mx.concatenate([r, _rn((8, K), dt, 2)[1:8]])
        mx.eval(W, c, r, x8)
        for lbl, fn in (("addmm", lambda x: mx.addmm(c, x, W)),
                        ("matmul", lambda x: x @ W),
                        ("matmul then add", lambda x: (x @ W) + c)):
            a, _ = bit_pattern(fn(r)[0])
            b, _ = bit_pattern(fn(x8)[0])
            print("| %s | %s: B=1 vs B=8 | %d |" % (name, lbl, int((a != b).sum())))
        for lbl, x in (("B=1", r), ("B=8", x8)):
            a, _ = bit_pattern(mx.addmm(c, x, W)[0])
            b, _ = bit_pattern(((x @ W) + c)[0])
            print("| %s | addmm vs matmul-then-add at %s | %d |" % (name, lbl, int((a != b).sum())))


# ---------------------------------------------------------------------- summarize

def _shape_str(s):
    return " ".join("%s=%s" % kv for kv in s.items())


def _sig(rs):
    """(pct bits differing, median ulp, max ulp) for a bag of records."""
    n = max((r.get("n", 0) for r in rs), default=0)
    d = max(r.get("ndiff", 0) for r in rs)
    return (round(100.0 * d / n, 1) if n else 0.0,
            max((r.get("med_ulp", 0) for r in rs), default=0),
            max(r.get("max_ulp", 0) for r in rs))


def cmd_summarize(args):
    rows = [json.loads(l) for l in open(args.out)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["op"], r["dtype"], _shape_str(r["shape"]))].append(r)

    print("| op | dtype | shape | verdict | first div B | bits differ | med ULP | max ULP | predicted kernels |")
    print("|---|---|---|---|---|---|---|---|---|")
    detail = []
    for key in sorted(groups):
        rs = groups[key]
        ok = [r for r in rs if r["status"] == "ok"]
        bad = [r for r in ok if r.get("ndiff", 0) > 0]
        nskip = len({r["B"] for r in rs if r["status"] != "ok"})
        preds = sorted({r["pred"] for r in ok})
        verdict = "**VARIANT**" if bad else "invariant"
        if nskip:
            verdict += " (%d B skipped)" % nskip
        pct, med, mx_ = _sig(bad) if bad else (0.0, 0, 0)
        print("| %s | %s | %s | %s | %s | %s%% | %s | %s | %s |" %
              (key[0], key[1], key[2], verdict, min((r["B"] for r in bad), default="-"),
               pct, med, mx_, ", ".join(preds)))
        if not bad:
            continue
        # compress consecutive batch sizes with identical divergence signature
        per_b, spans = {}, []
        for B in sorted({r["B"] for r in ok}):
            rb = [r for r in ok if r["B"] == B]
            per_b[B] = (_sig(rb), sorted({r["pred"] for r in rb})[0],
                        len({tuple(sorted(r.items(), key=str)) and r["ndiff"] for r in rb}) > 1)
        for B in sorted(per_b):
            s = per_b[B]
            if spans and spans[-1][2] == s:
                spans[-1][1] = B
            else:
                spans.append([B, B, s])
        detail.append("- **%s %s %s**: %s" % (
            key[0], key[1], key[2],
            "; ".join("B%s%s -> %s %s%% bits, med %s / max %s ULP%s" % (
                "=" if a == b else " %d-" % a, b, s[1], s[0][0], s[0][1], s[0][2],
                ", POSITION/MODE-DEPENDENT" if s[2] else "")
                for a, b, s in spans)))

    print("\n### eval-placement effect (compiled/graph vs eager at B=1)")
    dis = [r for r in rows if r.get("mode_matches_eager") is False]
    print("none - mx.eval() placement and mx.compile() do not change any baseline result"
          if not dis else "\n".join(
              "- %s %s %s mode=%s" % (r["op"], r["dtype"], _shape_str(r["shape"]), r["mode"]) for r in dis))

    print("\n### divergence detail")
    print("\n".join(detail) if detail else "none")


def cmd_selftest(args):
    """Positive control: the comparison pipeline must see a 1-ULP difference."""
    for dt, step in ((mx.float32, 1), (mx.float16, 1), (mx.bfloat16, 1)):
        a = mx.array([1.0], dtype=dt)
        pa, n = bit_pattern(a)
        pb = pa + step
        u = ulp_distance(pa, pb, n)
        assert int((pa != pb).sum()) == 1, dt
        assert int(u[0]) == 1, (dt, u)
    # sign handling: -0.0 and +0.0 are 0 ULP apart, +/-1 ULP around zero is 1
    pz, n = bit_pattern(mx.array([0.0, -0.0], dtype=mx.float32))
    assert int(ulp_distance(pz[:1], pz[1:], n)[0]) == 0
    # and the harness must call a genuinely batch-variant op variant
    W = _rn((2048, 1024), mx.float32, 1)
    r = _rn((1, 2048), mx.float32, 3)
    p1, n = bit_pattern((r @ W)[0])
    p2, _ = bit_pattern((mx.concatenate([r, _rn((7, 2048), mx.float32, 2)]) @ W)[0])
    assert int((p1 != p2).sum()) > 0, "positive control failed: matmul looks invariant"
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "summarize", "selftest", "splitkv", "epilogue"])
    p.add_argument("--child", nargs=2)
    p.add_argument("--op", nargs="*")
    p.add_argument("--dtype", nargs="*", choices=list(DTYPES))
    p.add_argument("--out", default="probe/results.jsonl")
    a = p.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize, "selftest": cmd_selftest,
     "splitkv": cmd_splitkv, "epilogue": cmd_epilogue}[a.cmd](a)
