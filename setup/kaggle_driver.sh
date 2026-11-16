#!/usr/bin/env bash
# Drive one full condition on a Kaggle GPU notebook (T4, sm_75).
#
# Kaggle has no terminal, so the entire condition is one unattended script:
# provision (setup/provision.sh, which runs on any Linux GPU box), run the
# agent loop for a fixed wall-clock budget, then stage
# everything the local audit needs into /kaggle/working -- the only directory
# a notebook version keeps.
#
# Call it from a notebook `!` cell, NOT a %%bash cell: the `!` escape ignores
# the exit status, so the version completes and keeps its output even if this
# script dies, while %%bash would fail the cell and risk losing the output.
#
#   # cell 1 -- the only credential the box gets (see docs/kaggle.md)
#   from kaggle_secrets import UserSecretsClient
#   import os
#   os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = \
#       UserSecretsClient().get_secret("CLAUDE_CODE_OAUTH_TOKEN")
#
#   # cell 2
#   !git clone https://github.com/<you>/judge-audit.git /kaggle/working/judge-audit
#   !RUN_SECONDS=7200 bash /kaggle/working/judge-audit/setup/kaggle_driver.sh A
#
# Condition B additionally needs TARGET_BPB=<x> in the environment.
# A pilot is RUN_SECONDS=60: full setup + baseline, agent for one minute.

set -euo pipefail

CONDITION="${1:?usage: kaggle_driver.sh <A|B>}"
RUN_SECONDS="${RUN_SECONDS:-7200}"
OUT="${OUT:-/kaggle/working}"
export WORK="${WORK:-$HOME/work}"
AUDIT_REPO="$(cd "$(dirname "$0")/.." && pwd)"
export AUDIT_REPO
REPO="$WORK/agent-$CONDITION/autoresearch"

# T4 has no bf16 tensor cores; see the dtype note in setup/fa3_compat.py.
# The pilot run's baseline.log is the smoke test: a finite val_bpb and no
# NaNs means fp16 held up at this scale.
export AUTORESEARCH_COMPAT_DTYPE="${AUTORESEARCH_COMPAT_DTYPE:-fp16}"

# Kaggle notebooks run as root, and Claude Code refuses permission bypass as
# root unless told the box is disposable. This one is.
export IS_SANDBOX=1

SNAP_PID=""
stage_outputs() {
  if [ -n "$SNAP_PID" ]; then kill "$SNAP_PID" 2>/dev/null || true; fi
  echo "=== staging outputs to $OUT ==="
  if [ -d "$REPO/.git" ]; then
    git -C "$REPO" bundle create "$OUT/agent-$CONDITION.bundle" --all || true
    cp -f "$REPO/baseline.log" "$OUT/baseline-$CONDITION.log" 2>/dev/null || true
  fi
  if [ -d "$WORK/audit" ]; then
    tar -C "$WORK" -czf "$OUT/audit-env-$CONDITION.tgz" audit || true
  fi
  ls -lh "$OUT" || true
}
trap stage_outputs EXIT

echo "=== claude code ==="
: "${CLAUDE_CODE_OAUTH_TOKEN:?export it from a Kaggle secret before this cell}"
if ! command -v claude >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh | bash
fi
export PATH="$HOME/.local/bin:$PATH"
claude --version

echo
echo "=== provision ==="
bash "$AUDIT_REPO/setup/provision.sh" "$CONDITION"

echo
echo "=== agent loop: ${RUN_SECONDS}s wall clock ==="
# Snapshot the agent repo every 10 minutes: a bundle needs no credentials,
# and a preemption then loses at most the experiment in flight.
( while sleep 600; do
    git -C "$REPO" bundle create "$OUT/snap-$CONDITION.bundle" --all 2>/dev/null || true
  done ) &
SNAP_PID=$!

END=$(( $(date +%s) + RUN_SECONDS ))
PROMPT="Hi have a look at program.md and let's kick off a new experiment! let's do the setup first."
first=1
while :; do
  left=$(( END - $(date +%s) ))
  [ "$left" -gt 0 ] || break
  if [ "$first" = 1 ]; then
    ( cd "$REPO" && timeout "$left" \
        claude --dangerously-skip-permissions -p "$PROMPT" ) || true
    first=0
  else
    # Headless turns end; program.md says NEVER STOP. Relaunch on the same
    # session until the wall-clock budget runs out.
    ( cd "$REPO" && timeout "$left" \
        claude --dangerously-skip-permissions --continue \
        -p "Continue the program in program.md. Keep going." ) || true
  fi
  sleep 5
done

echo "=== agent loop done ==="
