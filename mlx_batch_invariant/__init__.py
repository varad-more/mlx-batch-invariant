"""Batch-invariant kernels for MLX on Apple Silicon.

Every function here returns bitwise identical results for a given row of input
regardless of what else was in the batch, how many query tokens were submitted
together, or how the runtime felt like scheduling the work.

    import mlx_batch_invariant as bi

    with bi.batch_invariant_mode():
        logits = model(tokens)

See PHASE0.md for the measurements that motivate each kernel and BENCHMARK.md for
what it costs.
"""

import contextlib

import mlx.core as mx

from ._attention import scaled_dot_product_attention
from ._gemm import TILE, addmm, linear, matmul

__all__ = [
    "matmul",
    "addmm",
    "linear",
    "scaled_dot_product_attention",
    "batch_invariant_mode",
    "is_enabled",
    "TILE",
    "__version__",
]

__version__ = "0.1.0"

_depth = 0


def is_enabled():
    """True inside a :func:`batch_invariant_mode` block."""
    return _depth > 0


def _unsupported(what, strict, fallback, *args, **kwargs):
    if strict:
        raise NotImplementedError(
            "%s is not batch-invariant in this build. Pass strict=False to "
            "batch_invariant_mode() to fall back to stock MLX for this call, "
            "accepting that its result is not invariant." % what
        )
    return fallback(*args, **kwargs)


@contextlib.contextmanager
def batch_invariant_mode(strict=True):
    """Route MLX's matmul, addmm, ``@``, nn.Linear and SDPA through invariant kernels.

    Ops this library cannot make invariant raise ``NotImplementedError``.  With
    ``strict=False`` they fall back to stock MLX instead, which means the result is
    no longer guaranteed bitwise reproducible -- use it to find out what a model
    needs, not to ship.
    """
    global _depth
    import mlx.nn as nn

    if _depth:  # already patched; nesting must not stack another set of shims
        _depth += 1
        try:
            yield
        finally:
            _depth -= 1
        return

    saved = {
        "matmul": mx.matmul,
        "addmm": mx.addmm,
        "matmul_op": mx.array.__matmul__,
        "sdpa": mx.fast.scaled_dot_product_attention,
        "linear": nn.Linear.__call__,
    }

    def bi_matmul(a, b, *, stream=None):
        if a.ndim < 2 or b.ndim != 2:
            return _unsupported(
                "matmul with shapes %s @ %s" % (a.shape, b.shape),
                strict, saved["matmul"], a, b, stream=stream)
        return matmul(a, b)

    def bi_addmm(c, a, b, alpha=1.0, beta=1.0, *, stream=None):
        if a.ndim < 2 or b.ndim != 2 or c.ndim != 1 or c.shape[0] != b.shape[-1]:
            return _unsupported(
                "addmm with shapes %s, %s @ %s" % (c.shape, a.shape, b.shape),
                strict, saved["addmm"], c, a, b, alpha, beta, stream=stream)
        return addmm(c, a, b, alpha, beta)

    def bi_sdpa(q, k, v, *, scale, mask=None, sinks=None, stream=None):
        if sinks is not None or q.ndim != 4 or q.shape[-1] % 32 or v.shape[-1] % 32:
            return _unsupported(
                "attention with sinks or head dim %s" % (q.shape[-1],),
                strict, saved["sdpa"], q, k, v,
                scale=scale, mask=mask, sinks=sinks, stream=stream)
        return scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    def bi_linear_call(self, x):
        w = self["weight"]
        if x.ndim < 2 or w.ndim != 2:
            return saved["linear"](self, x)
        return linear(x, w, self["bias"] if "bias" in self else None)

    mx.matmul = bi_matmul
    mx.addmm = bi_addmm
    mx.array.__matmul__ = lambda a, b: bi_matmul(a, b)
    mx.fast.scaled_dot_product_attention = bi_sdpa
    nn.Linear.__call__ = bi_linear_call
    _depth = 1
    try:
        yield
    finally:
        _depth = 0
        mx.matmul = saved["matmul"]
        mx.addmm = saved["addmm"]
        mx.array.__matmul__ = saved["matmul_op"]
        mx.fast.scaled_dot_product_attention = saved["sdpa"]
        nn.Linear.__call__ = saved["linear"]
