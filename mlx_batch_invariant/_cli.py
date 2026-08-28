"""`mlx-bi verify` -- prove on this machine that the kernels are batch-invariant."""

import argparse
import platform
import sys

import mlx.core as mx
import numpy as np

from . import __version__, batch_invariant_mode, scaled_dot_product_attention
from ._gemm import linear, matmul

_UINT = {mx.float32: mx.uint32, mx.float16: mx.uint16, mx.bfloat16: mx.uint16}
_DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _bits(a):
    return np.array(mx.view(mx.contiguous(a).reshape(-1), _UINT[a.dtype]))


def _gemm_case(dtype, K, N, batches, fn):
    """Rows of a fixed input must not change when other rows are added."""
    x = mx.random.normal((max(batches), K)).astype(dtype)
    w = mx.random.normal((N, K)).astype(dtype)
    mx.eval(x, w)
    ref, bad = None, []
    for B in batches:
        y = fn(mx.contiguous(x[:B]), w)
        mx.eval(y)
        b = _bits(y[0])
        if ref is None:
            ref = b
        elif (b != ref).any():
            bad.append(B)
    return bad


def _attn_case(dtype, H, Hkv, N, D, qLs, fn):
    """One query token must not change when other query tokens ride along."""
    k = mx.random.normal((1, Hkv, N, D)).astype(dtype)
    v = mx.random.normal((1, Hkv, N, D)).astype(dtype)
    q = mx.random.normal((1, H, max(qLs), D)).astype(dtype)
    mx.eval(k, v, q)
    ref, bad = None, []
    for qL in qLs:
        y = fn(mx.contiguous(q[:, :, max(qLs) - qL:, :]), k, v)
        mx.eval(y)
        b = _bits(mx.contiguous(y[:, :, -1, :]))
        if ref is None:
            ref = b
        elif (b != ref).any():
            bad.append(qL)
    return bad


def _quant_case(nbits, K, N, batches):
    """Quantized weights must be invariant too -- every shipped MLX model is 4-bit."""
    import mlx.nn as nn

    lin = nn.Linear(K, N, bias=False)
    lin.apply(lambda p: p.astype(mx.float16))
    q = nn.QuantizedLinear.from_linear(lin, bits=nbits)
    x = mx.random.normal((max(batches), K)).astype(mx.float16)
    mx.eval(x, q.parameters())
    ref, bad = None, []
    with batch_invariant_mode(strict=True):
        for B in batches:
            y = q(mx.contiguous(x[:B]))
            mx.eval(y)
            b = _bits(y[0])
            if ref is None:
                ref = b
            elif (b != ref).any():
                bad.append(B)
    return bad


def verify(argv=None):
    p = argparse.ArgumentParser(prog="mlx-bi verify")
    p.add_argument("--dtype", action="append", choices=sorted(_DTYPES),
                   help="repeatable; default is all three")
    p.add_argument("--quick", action="store_true", help="one shape per op")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    mx.random.seed(args.seed)
    dtypes = [_DTYPES[d] for d in (args.dtype or sorted(_DTYPES))]
    batches = [1, 2, 8, 32] if args.quick else [1, 2, 3, 5, 8, 16, 31, 32, 33, 64, 128]
    qLs = [1, 2, 8, 9] if args.quick else [1, 2, 3, 4, 7, 8, 9, 16, 17]
    gemms = [(4096, 4096)] if args.quick else [(512, 1024), (4096, 4096), (11008, 4096)]
    attns = [(8, 8, 1024, 128)] if args.quick else [
        (32, 8, 512, 128), (16, 8, 4096, 128), (8, 8, 4096, 128), (32, 8, 8192, 128)]
    quants = [(4, 2048, 2048)] if args.quick else [
        (4, 2048, 2048), (4, 4096, 11008), (8, 2048, 2048)]

    print("mlx-batch-invariant %s | mlx %s | %s %s"
          % (__version__, mx.__version__, platform.machine(), platform.mac_ver()[0]))
    print("device: %s\n" % (mx.device_info().get("architecture", "?"),))

    fails = 0
    for dtype in dtypes:
        name = str(dtype).rsplit(".", 1)[-1]
        for K, N in gemms:
            bad = _gemm_case(dtype, K, N, batches, lambda x, w: linear(x, w))
            fails += bool(bad)
            print("  %-9s linear   K=%-6d N=%-6d %s"
                  % (name, K, N, "ok" if not bad else "VARIANT at batch %s" % bad))
        for H, Hkv, N, D in attns:
            bad = _attn_case(dtype, H, Hkv, N, D, qLs,
                             lambda q, k, v: scaled_dot_product_attention(q, k, v))
            fails += bool(bad)
            print("  %-9s sdpa     H=%d/%-2d N=%-6d %s"
                  % (name, H, Hkv, N, "ok" if not bad else "VARIANT at qL %s" % bad))

    for nbits, K, N in quants:
        bad = _quant_case(nbits, K, N, batches)
        fails += bool(bad)
        print("  int%-6d quantized K=%-6d N=%-6d %s"
              % (nbits, K, N, "ok" if not bad else "VARIANT at batch %s" % bad))

    # A suite that cannot fail proves nothing: confirm stock MLX is still variant.
    stock = _gemm_case(mx.float32, 4096, 4096, batches, lambda x, w: x @ w.T)
    print("\ncontrol: stock mx.matmul float32 K=4096 N=4096 -> %s"
          % ("VARIANT at batch %s (expected)" % stock if stock
             else "invariant -- stock MLX may have been fixed; this library "
                  "is no longer proving anything"))
    if not stock:
        fails += 1

    import mlx.nn as nn
    lin = nn.Linear(2048, 2048, bias=False)
    lin.apply(lambda a: a.astype(mx.float16))
    qlin = nn.QuantizedLinear.from_linear(lin, bits=4)
    xq = mx.random.normal((max(batches), 2048)).astype(mx.float16)
    mx.eval(xq, qlin.parameters())
    qref, qbad = None, []
    for B in batches:
        b = _bits(qlin(mx.contiguous(xq[:B]))[0])
        if qref is None:
            qref = b
        elif (b != qref).any():
            qbad.append(B)
    print("control: stock quantized_matmul int4  K=2048 N=2048 -> %s"
          % ("VARIANT at batch %s (expected)" % qbad if qbad
             else "invariant -- stock MLX may have been fixed"))
    if not qbad:
        fails += 1

    print("\n%s" % ("FAILED (%d cases)" % fails if fails else "all invariant"))
    return 1 if fails else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "verify":
        return verify(argv[1:])
    print("usage: mlx-bi verify [--quick] [--dtype float32|float16|bfloat16] [--seed N]",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
