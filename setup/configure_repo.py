#!/usr/bin/env python3
"""
Configure a fresh autoresearch clone for a small/free GPU.

Two independent transforms:

  --shrink   Cut the model and the time budget so experiments take ~2 minutes
             instead of 5, giving roughly 25 per hour. Values follow the
             upstream README's own advice for small hardware.

  --compat   Replace the FlashAttention-3 call with the SDPA shim, so the repo
             runs on cards the kernels hub does not serve: Turing (Kaggle and
             Colab free tiers are T4, sm_75) and anything on Windows.

  --tiny     Applied on top of --shrink, for 4GB cards. Cuts the micro-batch
             so the unchunked cross-entropy logits tensor fits. Intended for
             running the AUDIT locally; the agent should not run on a machine
             holding your credentials.

Both are applied BEFORE the run baseline commit, so neither shows up in any
diff the monitors score.

Usage:
    python setup/configure_repo.py --repo /work/agent/autoresearch --shrink --compat
    python setup/configure_repo.py --repo /work/agent/autoresearch --shrink --check
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

# (file, exact old line, new line, why)
SHRINK = [
    ("prepare.py",
     "MAX_SEQ_LEN = 2048       # context length",
     "MAX_SEQ_LEN = 256        # context length",
     "context length; the single biggest cost lever"),
    ("prepare.py",
     "TIME_BUDGET = 300        # training time budget in seconds (5 minutes)",
     "TIME_BUDGET = 120        # training time budget in seconds (2 minutes)",
     "2-minute experiments, ~25/hour"),
    ("prepare.py",
     "EVAL_TOKENS = 40 * 524288  # number of tokens for val eval",
     "EVAL_TOKENS = 8 * 65536  # number of tokens for val eval",
     "smaller validation pass"),
    ("train.py",
     "DEPTH = 8               # number of transformer layers",
     "DEPTH = 4               # number of transformer layers",
     "primary model-complexity knob"),
    ("train.py",
     "DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)",
     "DEVICE_BATCH_SIZE = 32  # per-device batch size (reduce if OOM)",
     "fit a small card"),
    ("train.py",
     "TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step",
     "TOTAL_BATCH_SIZE = 2**15 # ~32K tokens per optimizer step",
     "more optimizer steps inside a short budget"),
    ("train.py",
     'WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context',
     'WINDOW_PATTERN = "L" # sliding window pattern: L=full, S=half context',
     "banded attention is inefficient at short context"),
]

# Applied ON TOP of --shrink, for 4GB consumer cards (RTX 2050/3050, GTX 1650).
#
# The binding constraint on a small card is not the model -- at DEPTH=4 the
# weights, gradients and optimizer state together are only ~400MB. It is the
# loss head. train.py computes an unchunked
#   F.cross_entropy(logits.view(-1, 32768), ...)
# so the full (DEVICE_BATCH_SIZE * MAX_SEQ_LEN, vocab_size) logits tensor is
# materialized, then upcast to fp32 inside log_softmax, and its gradient is
# held for the backward pass. At batch 32, seq 256: 537MB bf16 + 1074MB fp32
# + 537MB grad = ~2.1GB transient, which OOMs a 4GB card that is also driving
# a display. At batch 8 the same three terms are ~550MB.
#
# grad_accum_steps absorbs the change (TOTAL_BATCH_SIZE is unchanged, so the
# optimizer sees identical batches), which means this costs throughput but
# does not alter the training math.
TINY = [
    ("train.py",
     "DEVICE_BATCH_SIZE = 32  # per-device batch size (reduce if OOM)",
     "DEVICE_BATCH_SIZE = 8  # per-device batch size (reduce if OOM)",
     "4GB: logits tensor 2.1GB -> 0.5GB; grad_accum absorbs it"),
]

COMPAT_OLD = """from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface"""

COMPAT_NEW = '''# --- attention backend (patched by judge-audit/setup/configure_repo.py) ---
# FA3 from the kernels hub needs a Linux prebuilt for an Ampere or Hopper card.
# Where that is unavailable (Turing, Windows, no hub access), fall back to a
# shim over torch's scaled_dot_product_attention. The shim is verified against
# a reference implementation in setup/test_fa3_compat.py.
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
try:
    if cap[0] < 8:
        raise ImportError("FlashAttention needs sm_80+; this device is sm_%d%d" % cap)
    from kernels import get_kernel
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    fa3 = get_kernel(repo).flash_attn_interface
    print("attention backend: FlashAttention-3 (%s)" % repo)
except Exception as _e:
    import fa3_compat
    fa3 = fa3_compat.flash_attn_interface
    print("attention backend: SDPA shim (%s)" % _e)
# --- end patch ---'''


def patch_file(path: pathlib.Path, old: str, new: str, dry: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if new in text and old not in text:
        return "already"
    if old not in text:
        return "MISSING"
    if not dry:
        if not path.with_suffix(path.suffix + ".orig").exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".orig"))
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "ok"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=pathlib.Path)
    ap.add_argument("--shrink", action="store_true")
    ap.add_argument("--compat", action="store_true")
    ap.add_argument("--tiny", action="store_true",
                    help="4GB cards; requires --shrink")
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    repo = args.repo.expanduser().resolve()
    if not (repo / "train.py").exists():
        sys.exit("%s does not look like an autoresearch repo" % repo)
    if not (args.shrink or args.compat or args.tiny):
        sys.exit("pass --shrink, --compat and/or --tiny")
    if args.tiny and not args.shrink:
        text = (repo / "train.py").read_text(encoding="utf-8")
        if "DEVICE_BATCH_SIZE = 128" in text:
            sys.exit("--tiny patches the shrunk DEVICE_BATCH_SIZE (32), but this "
                     "repo still has the stock 128. Pass --shrink --tiny together.")

    problems = 0

    if args.shrink:
        print("shrink:")
        for fname, old, new, why in SHRINK:
            status = patch_file(repo / fname, old, new, args.check)
            if status == "MISSING":
                problems += 1
                print("  MISSING  %-11s %s" % (fname, old.split("=")[0].strip()))
                print("           upstream text has changed; update SHRINK in this file")
            else:
                print("  %-8s %-11s %-22s (%s)"
                      % (status, fname, new.split("#")[0].strip(), why))

    if args.tiny:
        print("\ntiny (4GB):")
        for fname, old, new, why in TINY:
            status = patch_file(repo / fname, old, new, args.check)
            # TINY patches lines that SHRINK produces. Under --check nothing was
            # written, so the anchor is legitimately absent; look for the stock
            # line SHRINK would have turned into it and call that pending, not a
            # failure. Without this, --shrink --tiny --check always reports a
            # false MISSING.
            if status == "MISSING" and args.check and args.shrink:
                pred = next((s_old for s_file, s_old, s_new, _ in SHRINK
                             if s_file == fname and s_new == old), None)
                if pred and pred in (repo / fname).read_text(encoding="utf-8"):
                    print("  pending  %-11s %-22s (applied after --shrink)"
                          % (fname, new.split("#")[0].strip()))
                    continue
            if status == "MISSING":
                problems += 1
                print("  MISSING  %-11s %s" % (fname, old.split("=")[0].strip()))
                print("           expected the shrunk value; run --shrink first")
            else:
                print("  %-8s %-11s %-22s (%s)"
                      % (status, fname, new.split("#")[0].strip(), why))

    if args.compat:
        print("\ncompat:")
        status = patch_file(repo / "train.py", COMPAT_OLD, COMPAT_NEW, args.check)
        if status == "MISSING":
            problems += 1
            print("  MISSING  the FA3 import block in train.py has changed")
        else:
            print("  %-8s train.py  FA3 -> SDPA fallback" % status)
            shim = pathlib.Path(__file__).parent / "fa3_compat.py"
            if not args.check:
                shutil.copy2(shim, repo / "fa3_compat.py")
            print("  %-8s fa3_compat.py copied beside train.py"
                  % ("would" if args.check else "ok"))

    if problems:
        print("\n%d anchor(s) not found. Nothing was written for those."
              % problems)
        return 1

    if args.check:
        print("\n--check: nothing written.")
    else:
        print("\nOriginals saved as *.orig. Commit this state as the run "
              "baseline BEFORE the agent starts:")
        print("  git -C %s add -A && git -C %s commit -m 'configure for small GPU'"
              % (repo, repo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
