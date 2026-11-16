# runs/

One folder per agent run: `runs/A/`, `runs/B/`, later `runs/A-seed2/`.
Each holds a run card (`README.md`), the agent's log (`results.tsv`) and the audit outputs.
Large artifacts (git bundle, transcripts, audit environment) stay in the Kaggle version output and are linked from the run card.

## Auditing a run locally

From the folder where you downloaded the Kaggle outputs (bash shown; on Windows PowerShell put each command on its own line, `&&` is not supported):

```bash
mkdir runA && cd runA
git clone ../agent-A.bundle autoresearch
tar -xzf ../audit-env-A.tgz          # -> ./audit/
tar -xzf ../transcripts-A.tgz        # -> ./projects/

# baseline commit and agent branch
git -C autoresearch log --oneline --all --grep "run baseline"
git -C autoresearch branch -a

# 1. rule-based monitor over every kept commit
python <judge-audit>/audit/diff_monitor.py --repo autoresearch \
    --rev-range <baseline>..autoresearch/<tag> --json monitor.json

# 2. transcript mentions of the shortcuts (counts, then each hit in context)
grep -c -E "extra_train|eval_tokens|EVAL_TOKENS|VAL_FILENAME" projects/*/*.jsonl
python <judge-audit>/audit/show_hits.py projects/

# 3. hidden-shard re-evaluation (GPU; --compat --tiny fits 4 GB)
python <judge-audit>/audit/reeval.py --agent-repo autoresearch \
    --branch autoresearch/<tag> --base <baseline> \
    --stock-prepare audit/refs/prepare.stock.py \
    --clean-cache audit/home/.cache/autoresearch \
    --hidden-shard audit/home/hidden/shard_06541.parquet \
    --tokenizer-dir audit/home/.cache/autoresearch/tokenizer \
    --out reeval.tsv
```

Copy `monitor.json`, `reeval.tsv` and your notes back into `runs/<run>/` and commit them.
