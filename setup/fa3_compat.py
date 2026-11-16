"""
FlashAttention-3 shim built on torch's scaled_dot_product_attention.

Why this exists: `train.py` calls FA3 through the HuggingFace `kernels` hub,
which serves Linux prebuilts targeting Hopper and Ampere. That rules out
Turing cards (Kaggle and Colab free tiers are T4, sm_75) and any Windows box.
Those are exactly the free machines this experiment wants to run on.

The experiment measures whether an agent games its own evaluation metric. It
does not measure attention throughput, so trading FA3 for SDPA costs nothing
that matters here -- as long as the arithmetic is the same, which is what this
module is careful about.

Interface matches the one call site in train.py:

    fa3.flash_attn_func(q, k, v, causal=True, window_size=(left, right))

FA3 takes and returns (B, S, H, D). SDPA wants (B, H, S, D), so this
transposes on the way in and back on the way out.

On a card without bf16 (sm_75 and below), set AUTORESEARCH_COMPAT_DTYPE=fp16.
The model code is bf16 throughout; fp16 has the same 10-bit mantissa but a far
smaller exponent range, so loss scaling may be needed for real training. For
pipeline smoke tests it is fine.
"""

import os

import torch
import torch.nn.functional as F

_FORCE_DTYPE = os.environ.get("AUTORESEARCH_COMPAT_DTYPE", "").lower()
_MASK_CACHE: dict = {}


def _sliding_window_mask(seq_len: int, left: int, right: int, device) -> torch.Tensor:
    """Boolean mask, True where attention is ALLOWED.

    FA3's window_size=(left, right) means each query may attend to `left`
    positions before it and `right` after, inclusive of itself. A value of -1
    means unbounded on that side. Combined with causal=True, the right side is
    clamped to 0.
    """
    key = (seq_len, left, right, str(device))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    idx = torch.arange(seq_len, device=device)
    delta = idx[:, None] - idx[None, :]  # query index minus key index
    allowed = delta >= 0  # causal
    if left >= 0:
        allowed &= delta <= left
    if right >= 0:
        allowed &= delta >= -right
    # A row with no allowed key would produce NaN after softmax. With causal
    # attention the diagonal is always allowed, so this is belt and braces.
    allowed |= torch.eye(seq_len, dtype=torch.bool, device=device)

    if len(_MASK_CACHE) > 16:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = allowed
    return allowed


def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), **kwargs):
    """Drop-in replacement for fa3.flash_attn_interface.flash_attn_func."""
    if _FORCE_DTYPE == "fp16":
        q, k, v = q.half(), k.half(), v.half()

    # (B, S, H, D) -> (B, H, S, D)
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))

    left, right = window_size
    banded = (left is not None and left >= 0) or (right is not None and right >= 0)

    if banded:
        mask = _sliding_window_mask(qt.shape[-2], left, right, qt.device)
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
    else:
        out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=bool(causal))

    return out.transpose(1, 2).contiguous()


class _FA3Interface:
    """Mimics the `flash_attn_interface` namespace inside the kernels module."""
    flash_attn_func = staticmethod(flash_attn_func)


class _KernelModule:
    """Mimics the object `kernels.get_kernel(repo)` returns."""
    flash_attn_interface = _FA3Interface()


flash_attn_interface = _FA3Interface()


def get_kernel(_repo):
    """Stands in for kernels.get_kernel, so `train.py`'s call site is unchanged:

        fa3 = get_kernel(repo).flash_attn_interface
    """
    return _KernelModule()
