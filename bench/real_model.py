"""Phase 4 gate: run the library against a real 4-bit checkpoint, not random weights.

Everything else in this repo is measured on synthetic tensors. That proves the
kernels are invariant; it does not prove they compute the thing a trained model
needs. This loads a real MLX checkpoint and checks both.

    .venv/bin/python bench/real_model.py [--model mlx-community/Qwen1.5-0.5B-Chat-4bit]

Requires mlx-lm, which is a validation dependency only -- the package itself does
not import it.
"""

import argparse
import sys

import mlx.core as mx

sys.path.insert(0, ".")
import mlx_batch_invariant as bi

DEFAULT = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
PROMPT = "The three laws of thermodynamics are"
_UINT = {mx.float32: mx.uint32, mx.float16: mx.uint16, mx.bfloat16: mx.uint16}


def bits(x):
    return mx.view(mx.contiguous(x).reshape(-1), _UINT[x.dtype])


def last_logits(model, ids_batch):
    """Logits for the final position of every row of a (B, L) token batch."""
    out = model(ids_batch)
    logits = out[0] if isinstance(out, tuple) else out
    mx.eval(logits)
    return logits[:, -1, :]


def batch_sweep(model, ids, filler, batches):
    """Bits of row 0's final-position logits as batch size grows around it."""
    base, diffs = None, []
    for B in batches:
        rows = [ids] + [filler[(i - 1) % len(filler)] for i in range(1, B)]
        got = bits(last_logits(model, mx.stack(rows))[0])
        mx.eval(got)
        if base is None:
            base = got
        else:
            diffs.append((B, int(mx.sum(got != base).item()), got.size))
        mx.clear_cache()
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--tokens", type=int, default=40)
    args = ap.parse_args()

    from mlx_lm import generate, load

    print("loading %s ..." % args.model)
    model, tokenizer = load(args.model)

    ids = mx.array(tokenizer.encode(PROMPT))
    L = ids.size
    # Neighbours must share the prompt's length so one forward pass covers the batch;
    # only their content differs, which is exactly the variable under test.
    filler = [mx.array(tokenizer.encode(t)[:L] + [tokenizer.eos_token_id] * L)[:L]
              for t in ("Write a haiku about the sea and the",
                        "In 1969 the first crewed mission that",
                        "A recipe for bread needs flour and")]

    print("\n=== 1. batch invariance of the final logits (%d tokens of prompt) ===" % L)
    stock = batch_sweep(model, ids, filler, args.batches)
    print("  stock MLX      :", ", ".join("B=%d %d/%d bits" % d for d in stock))
    with bi.batch_invariant_mode(strict=True):
        ours = batch_sweep(model, ids, filler, args.batches)
    print("  batch-invariant:", ", ".join("B=%d %d/%d bits" % d for d in ours))

    stock_bad = sum(d[1] for d in stock)
    ours_bad = sum(d[1] for d in ours)
    if ours_bad:
        print("  FAIL: invariant path still moved %d bits" % ours_bad)
        return 1
    if not stock_bad:
        print("  INCONCLUSIVE: stock MLX did not drift on this model either, so this "
              "run proves nothing. Try a longer prompt or a larger batch.")
        return 1
    print("  ok: stock moved %d bits across the sweep, invariant path moved 0"
          % stock_bad)

    print("\n=== 2. the kernels still produce sane text on trained weights ===")
    mx.clear_cache()
    with bi.batch_invariant_mode(strict=True):
        text = generate(model, tokenizer, prompt=PROMPT, max_tokens=args.tokens,
                        verbose=False)
    print("  prompt: %r" % PROMPT)
    print("  output: %r" % text.strip()[:400])
    if not text.strip():
        print("  FAIL: empty generation")
        return 1
    print("\nall real-checkpoint checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
