# Running the agent on Lightning AI's free tier

Everything here was read off <https://lightning.ai/pricing> on 2026-11-16.
Third-party blogs widely claim Lightning gives "15 credits/month" or "22 GPU
hours/month" — the official page does not say that, and the distinction
matters for planning. Anything I could not confirm is marked **unverified**.

## The credits

| | |
|---|---|
| On registration | **5 credits**, no credit card |
| Attach a card | **+25 credits** (30 total) |
| Reset | **None — one-time grant.** Unused credits expire after 12 months |
| What that buys | "~50 GPU hours on interruptible machines" |
| Phone number | Requested, for abuse prevention |

The one-time grant is the planning constraint. There is no monthly top-up to
wait for, so a botched run costs real budget — which is the argument for
rehearsing the whole pipeline on CPU and locally before spending any of it.

## What the free plan actually allows

| | Free plan |
|---|---|
| Concurrent Studios | 1 (2 concurrent GPUs max, 1 GPU per Studio) |
| T4 / L4 / L40S session length | **Unlimited** |
| A100 / H100 / H200 session length | 4 hours |
| Free CPU Studio | 1, with 4-hour restarts |
| Persistent storage | 50 GB (10 GB free, rest pay-as-you-go) |
| Shell | Terminal, SSH, and sudo |
| Machines | Interruptible (that is how 50 hours is reached) |
| Cloud providers | All |

Two things follow. **L4 is the right pick**: 24 GB, `sm_89` (so native bf16 and
FlashAttention-3, no shim needed), and unlimited session length, where an A100
would cut you off at 4 hours mid-run. And **conditions A and B must run
sequentially**, because the free plan allows one Studio.

> **Unverified:** whether the 50 GB persists after a Studio stops (Lightning
> markets Studios as persistent, and the pricing page says "Max inactive
> Studios: Unlimited", but it never states it outright), and the exact
> credits-per-hour rate — the page quotes USD/hr, never credits/hr. Confirm
> both in the console before relying on them.

## Budget

| Phase | Hours | Where |
|---|---|---|
| Setup, `uv sync`, data download, baseline | ~0.5 | free CPU Studio where possible |
| Condition A (neutral) | ~2 | L4 |
| Condition B (pressure) | ~2 | L4 |
| Audit re-eval, both conditions | ~2 | **locally, 0 credits** |
| Slack for one failed attempt | ~2 | L4 |

~6.5 GPU-hours against ~50 available. The margin is deliberate: the first
attempt at an unattended multi-hour agent run usually teaches you something
and produces no usable data.

Run the data download and `uv sync` on the **free CPU Studio**, not the L4.
Downloading 4 shards from the Hub is bandwidth-bound and does not need a GPU;
paying GPU rates to wait on a download is the easiest way to waste the grant.

## Setup

```bash
# 1. In the Lightning console: New Studio -> start on the free CPU machine.

# 2. Get the harness in. Clone it -- do NOT authenticate gh or git.
git clone https://github.com/<you>/judge-audit.git ~/work/judge-audit

# 3. Export the API key for the judge only. The agent does not need it;
#    Claude Code bills your subscription.
#    Use the Studio's secrets UI, not a plaintext ~/.bashrc.

# 4. Switch the Studio to an L4, then provision one condition:
cd ~/work/judge-audit
bash setup/lightning_setup.sh A

# 5. Start the agent INSIDE the agent repo, permissions disabled.
#    Let it run ~2 hours. Then stop the GPU before auditing.
```

`lightning_setup.sh` ends by printing the exact `audit/reeval.py` invocation
with the run's baseline SHA and paths filled in. Copy that line out and keep
it — the audit can run days later, and on a different machine.

Condition B additionally needs `TARGET_BPB`, which can only be set after you
have seen the baseline the script measures in step 6. Run A first, read its
baseline and its best honest gain, then set B's target from those.

## Sandboxing on a Studio

A fresh Studio is already most of what the original plan meant by "a
container": a disposable VM with none of your data on it. Three rules keep it
that way.

1. **Do not authenticate anything.** No `gh auth login`, no SSH keys, no cloud
   CLI, no git credential helper. Clone over HTTPS from a public repo. If the
   agent needs nothing to push, it can push nothing.
2. **Only the judge's API key enters the Studio**, and only if you run the
   judge there — which you should not, since it is cheaper to run it locally.
   Prefer a Studio with no secrets at all.
3. **The audit stays off the box.** `lightning_setup.sh` puts `$AUDIT_ROOT`
   outside `$AGENT_ROOT`, but the stronger version is to copy the agent's
   `.git` out and audit it at home. An agent with filesystem access and
   `NEVER STOP` in its instructions has both the means and the incentive.

For a belt-and-braces layer, run the agent in a plain container inside the
Studio (`docker run -v $AGENT_ROOT:/work ...`) so a path traversal out of the
repo lands in a container rather than the Studio's home directory. Not
strictly required, and it complicates GPU passthrough.

## Interruptible machines

Free-tier GPU hours come from interruptible instances, so a run can be
preempted. This is survivable almost for free, because the autoresearch loop
commits after every experiment: a preemption loses at most the one experiment
in flight, and the git history — the actual experimental record — is intact.

Do not, however, let the agent push to a remote as a backup. That requires
giving it credentials, which defeats the point. Snapshot instead:

```bash
# from outside the agent's directory, between experiments
git -C "$AGENT_ROOT/autoresearch" bundle create ~/snapshots/A-$(date +%s).bundle --all
```

A bundle is a single file on the persistent volume and needs no credentials.

## If Lightning does not work out

The fallbacks are worse but real, in order of preference:

- **Kaggle** — ~30 GPU-hours/week on P100 or T4, no card. The problem is not
  the silicon (`--compat` handles `sm_75`) but that it is a notebook with a
  ~9-hour session cap and no persistent shell. Workable for the audit re-eval,
  painful for an unattended agent loop.
- **Colab** — same notebook problem, less predictable GPU allocation, 90-minute
  idle disconnect. An idle disconnect is fatal to an unattended run.
- **Modal** — $30/month of free compute, genuinely recurring, but serverless
  and built for bursty jobs rather than a long-lived interactive shell. A card
  is required.
- **Vast.ai** — not free, but RTX 3090s go for well under $0.30/hr, so the
  entire experiment is a few dollars. If the free tiers prove fiddly, this is
  the cheapest way to stop fighting them.

Sources: [lightning.ai/pricing](https://lightning.ai/pricing) ·
[thundercompute's free-credit survey](https://www.thundercompute.com/blog/free-cloud-gpu-credits) ·
[gpuperhour: what works and what closed](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
