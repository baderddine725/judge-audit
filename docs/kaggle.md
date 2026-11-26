# Running the agent on Kaggle's free tier

Plan B, promoted to plan A on 2026-11-16: Lightning AI, the original host,
has been offline since its October 15 security incident
([notice](https://lightning.ai/security-incident)) and refuses new sign-ups;
its plan is retired to git history. Kaggle was already the named fallback, and the `--compat` shim was built for exactly its hardware — what
was missing was a way to run an *unattended agent loop* in a notebook, which
is what [`setup/kaggle_driver.sh`](../setup/kaggle_driver.sh) adds.

## What Kaggle gives (checked 2026-11-16)

| | Free tier |
|---|---|
| GPU | T4 ×2, 16 GB each (`sm_75`); P100 reportedly retired 2026-09-15 |
| Quota | ~30 GPU-hours/week, **renews weekly** |
| Session cap | 12 hours |
| Environment | Jupyter notebook VM, root, internet (after phone verification) |
| Unattended runs | "Save & Run All" executes the notebook with the browser closed |
| Persistence | none between sessions; `/kaggle/working` (~20 GB) kept as the version's output |
| Secrets | Add-ons → Secrets, injected per notebook |
| Shell | cells only; no SSH, no terminal |

T4 ×2 burns one quota-hour per wall-clock hour, and `train.py` uses one GPU
either way. The renewing quota is strictly better than Lightning's one-time
grant: a botched week costs nothing permanent.

## What changes vs the Lightning plan

**Hardware: T4 is `sm_75`.** No FA3 prebuilt and no bf16 tensor cores. The
first is solved — `--compat` swaps attention for the SDPA shim verified in
[`setup/test_fa3_compat.py`](../setup/test_fa3_compat.py). The second is the
real scientific caveat: the driver sets `AUTORESEARCH_COMPAT_DTYPE=fp16` per
the shim's own docstring, and fp16's narrow exponent range can need loss
scaling in real training. That is why the procedure below starts with a
**pilot run**: setup plus the honest baseline on the actual silicon for ~0.3
quota-hours, before any agent time is spent. NaNs or a wild baseline in
`baseline.log` mean more dtype work before proceeding.

**Drive mechanism: no shell.** Each condition is one batch notebook: a secrets
cell, then one `!` cell running the driver. The driver provisions via
[`setup/provision.sh`](../setup/provision.sh) unchanged (it runs on any Linux
GPU box), installs the Claude Code native binary, and runs the agent headless —
`claude -p --dangerously-skip-permissions`, relaunched with `--continue` until
the wall-clock budget (`RUN_SECONDS`) runs out.

**Persistence: version output only.** Nothing survives a session except
`/kaggle/working`. The driver bundles the agent repo there every 10 minutes
and again at exit, plus a tarball of the audit environment (stock
`prepare.py`, clean cache, hidden shard, tokenizer). Its `trap` stages
whatever exists even when a step fails, and a `!` cell ignores exit status, so
the version completes and keeps its output regardless.

## Credentials on the box

Claude Code needs the subscription token: run `claude setup-token` locally,
paste the result into a Kaggle secret named `CLAUDE_CODE_OAUTH_TOKEN`, and
revoke it after the runs. It is the agent's own credential, so it cannot be
kept from the agent — the same trade the Lightning plan made. Everything else
in lightning.md's sandboxing section stands: no `gh auth`, no SSH keys, no git
credential helper, and **no Kaggle API token in the notebook** — the
temptation is mid-run dataset uploads as backup, and the snapshot bundles make
that unnecessary. The judge's API key never enters Kaggle; judge and audit run
locally, unchanged.

## Procedure

1. Kaggle account, phone-verified. New notebook: Accelerator = **GPU T4 ×2**,
   Internet = on, secret attached. The cells below are also saved ready-made
   as [`notebooks/kaggle_condition.ipynb`](../notebooks/kaggle_condition.ipynb)
   (Kaggle: File → Import Notebook).
2. Cell 1 — pull the secret into the environment:

   ```python
   from kaggle_secrets import UserSecretsClient
   import os
   os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = \
       UserSecretsClient().get_secret("CLAUDE_CODE_OAUTH_TOKEN")
   ```

3. Cell 2 — pilot first:

   ```python
   !git clone https://github.com/<you>/judge-audit.git /kaggle/working/judge-audit
   !RUN_SECONDS=60 bash /kaggle/working/judge-audit/setup/kaggle_driver.sh A
   ```

4. Check `baseline-A.log` in the output: finite `val_bpb`, no NaNs, and note
   the baseline commit SHA from the cell log.
5. Set `RUN_SECONDS=7200`, **Save Version → Save & Run All**, close the tab.
6. Download `agent-A.bundle` and `audit-env-A.tgz` from the version's output.
7. Condition A's results set condition B's target; repeat with
   `TARGET_BPB=<x>` prepended to the driver line.

Locally, unpack and audit as in the README, with paths into the download:

```bash
git clone agent-A.bundle agent-A
tar xzf audit-env-A.tgz        # -> audit/refs, audit/home

python audit/reeval.py --agent-repo agent-A --branch autoresearch/<tag> \
    --base <baseline sha from the driver log> \
    --stock-prepare audit/refs/prepare.stock.py \
    --clean-cache audit/home/.cache/autoresearch \
    --hidden-shard audit/home/hidden/shard_06541.parquet \
    --tokenizer-dir audit/home/.cache/autoresearch/tokenizer \
    --out results/reeval_A.tsv
```

## Re-scoring on Kaggle

The re-score (`audit/reeval.py`) also runs on a Kaggle T4, not locally.
`train.py` stops on a 120 s wall clock, so a slower card trains fewer steps
and every audited model scores worse for reasons unrelated to the commit:
on run A, two runs of the identical baseline code differed by 14 steps and
0.03 bpb on the same T4 model. There is no agent and no credential in this
run, so nothing in the sandboxing section applies to it.

1. Upload the condition run's `agent-<C>.bundle` and `audit-env-<C>.tgz` as a
   **private** Kaggle dataset (it contains the hidden shard).
2. New notebook from [`notebooks/kaggle_audit.ipynb`](../notebooks/kaggle_audit.ipynb):
   GPU T4 ×2, Internet on, the dataset added as input, no secrets.
3. Set `BASE` (baseline commit) and `TIP` (last *kept* commit) from the run
   card, then Save & Run All. [`setup/kaggle_audit.sh`](../setup/kaggle_audit.sh)
   rebuilds the reference `prepare.py` from the base commit, builds one venv
   from the agent's unchanged `uv.lock`, re-runs the baseline `BASE_REPEATS`
   times (default 3) for the noise floor, then every kept commit.
4. Copy `reeval-<C>.tsv` from the output to `runs/<C>/reeval.tsv`.

Run A's `audit-env-A.tgz` packed the **unshrunk** upstream `prepare.py` as
`refs/prepare.stock.py` (fixed in `provision.sh` for later runs). The script
does not use that file, and `reeval.py` now refuses any reference whose
`MAX_SEQ_LEN`/`TIME_BUDGET`/`EVAL_TOKENS` differ from the base commit's.

## Budget

| Phase | Quota-hours |
|---|---|
| Pilot (setup + baseline only) | ~0.5 |
| Condition A | ~3 |
| Condition B | ~3 |
| Re-score per condition (3 baseline repeats + kept commits) | ~1 |
| Slack for one failed attempt | ~3 |

~9.5 of ~30 weekly hours, and the quota resets next week either way.

## Known risks

- **fp16 agent, bf16 audit.** The agent trains fp16 on T4; the local audit
  re-trains bf16. Deterministic replay across different GPUs is never
  bit-identical anyway, and the honest-commit score gap is the control that
  absorbs a systematic offset — but the write-up must say the precision
  differed between environments.
- **Failed batch versions may not keep output.** Kaggle promises output for
  completed versions; the driver's trap-and-stage plus the `!` cell's
  exit-status swallowing exist precisely so versions complete. Do not move the
  driver call into a `%%bash` cell, which fails the cell on a nonzero exit.
- **Quota exhaustion mid-run kills the session.** Check remaining GPU hours in
  account settings before launching a 2-hour run.
- **Interactive sessions idle out.** Babysitting a run in an open tab is the
  fragile path; Save & Run All is the reliable one.

Sources: [Kaggle notebook docs](https://www.kaggle.com/docs/notebooks) ·
[fastero on 2026 Kaggle limits](https://fastero.com/blog/kaggle-notebooks) ·
[gabrubuilds on the T4 ×2 tier](https://gabrubuilds.com/guides/kaggle-free-gpu/)
