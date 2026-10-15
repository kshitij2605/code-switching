# Evaluation results

| Directory | What | Produced by |
|---|---|---|
| `phase0/` | The Phase 0 baseline ladder | `03-training/asr-ft/phase0_ladder.py`, `eval/ladder_stats.py` |
| `phase1_leaderboard/` | The zero-shot open-model leaderboard | `phase1_leaderboard.py`, `eval/leaderboard_stats.py` |

Each directory holds:
- the scorer's metric JSONs per test set (`scores/*.json`, metrics only);
- summaries and statistics (`*.md`, `*.json`);
- the runners' run records (commit, arguments, model snapshots);
- `preds.sha256`, the checksum of every prediction file behind the numbers.

**Where the predictions are.** Prediction JSONLs carry the test sets' reference text: CS-FLEURS and CS-YODAS are CC BY-NC 4.0, FLEURS CC BY 4.0. They are therefore kept outside git. The originals are under `/data/code-switching/runs/` on mactrn01, and a copy sits in checksummed archives:

| Archive (mactrn01) | sha256 |
|---|---|
| `/data/code-switching/archive/phase0_20260930.tar.gz` | `bb846eb8f8d4d70322d66c0037fc4e44b6f70ae6472a6aac9bd4ec5a6022fa70` |
| `/data/code-switching/archive/phase1_leaderboard_20260930.tar.gz` | `64c90e5d24ad3a4e74aa4fbe3cda074df86ede87ecdb27dc91c13db02c222864` |

Git holds the metrics, summaries, run records and per-file prediction checksums. The archives above are the copy of record and must be kept.
