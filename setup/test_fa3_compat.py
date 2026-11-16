#!/usr/bin/env python3
"""
Correctness check for the SDPA shim, against a naive reference attention.

Runs on CPU in float32, so it works anywhere -- no GPU, no kernels hub. If the
shim's arithmetic matches the reference here, swapping FA3 for it changes
training numerics only by the usual kernel-level tolerance.

    python setup/test_fa3_compat.py
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fa3_compat import flash_attn_func, get_kernel  # noqa: E402


def reference(q, k, v, causal=True, window_size=(-1, -1)):
    """Textbook attention over (B, S, H, D) tensors. Slow and obviously right."""
    B, S, H, D = q.shape
    qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
    scores = (qt @ kt.transpose(-2, -1)) / (D ** 0.5)

    idx = torch.arange(S)
    delta = idx[:, None] - idx[None, :]
    allowed = torch.ones(S, S, dtype=torch.bool)
    if causal:
        allowed &= delta >= 0
    left, right = window_size
    if left >= 0:
        allowed &= delta <= left
    if right >= 0:
        allowed &= delta >= -right
    allowed |= torch.eye(S, dtype=torch.bool)

    scores = scores.masked_fill(~allowed, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ vt
    return out.transpose(1, 2)


CASES = [
    ("full causal", dict(causal=True, window_size=(-1, -1))),
    ("window 8 left", dict(causal=True, window_size=(8, 0))),
    ("window 3 left", dict(causal=True, window_size=(3, 0))),
    ("window 1 left", dict(causal=True, window_size=(1, 0))),
    ("half context", dict(causal=True, window_size=(31, 0))),
]


def main():
    torch.manual_seed(0)
    B, S, H, D = 2, 64, 4, 32
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)

    failures = []
    for name, kw in CASES:
        got = flash_attn_func(q, k, v, **kw)
        want = reference(q, k, v, **kw)
        if got.shape != want.shape:
            failures.append("%s: shape %s != %s" % (name, got.shape, want.shape))
            print("  %-16s FAIL (shape)" % name)
            continue
        err = (got.float() - want).abs().max().item()
        ok = err < 1e-4
        print("  %-16s %-5s max abs err %.2e" % (name, "PASS" if ok else "FAIL", err))
        if not ok:
            failures.append("%s: max abs err %.2e" % (name, err))

    # The shim must also satisfy the call shape train.py uses.
    fa3 = get_kernel("whatever").flash_attn_interface
    out = fa3.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
    shape_ok = out.shape == q.shape
    print("  %-16s %-5s get_kernel(...).flash_attn_interface.flash_attn_func"
          % ("call shape", "PASS" if shape_ok else "FAIL"))
    if not shape_ok:
        failures.append("call shape: %s != %s" % (out.shape, q.shape))

    print("\n%d/%d checks pass." % (len(CASES) + 1 - len(failures), len(CASES) + 1))
    if failures:
        for f in failures:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
