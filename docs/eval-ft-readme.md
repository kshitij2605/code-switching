# EN-JP Code-Switching ASR — research project

On-device ASR for **English-Japanese code-switching** speech, built by
fine-tuning **LiquidAI/LFM2.5-Audio-1.5B-JP**. Cloud ASR mangles bilingual
speech (forces English into katakana, drops the minority language); an on-device
fine-tune fixes this for privacy-sensitive use.

👉 **Read [`CONTEXT.md`](CONTEXT.md) first** — it's the full handoff (status,
environment, verified API facts, how to resume on another GPU, next steps).

## Quickstart
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_score.py                 # 6/6 pass

# pull JA-EN slice of CS-FLEURS (real read speech)
.venv/bin/python data/load_csfleurs.py --method read --split test --limit 196

# baseline: LFM (or --model whisper) -> predictions -> metrics
.venv/bin/python eval/run_baseline.py --model lfm \
  --manifest data/csfleurs/read_test/manifest.jsonl --audio-root data/csfleurs \
  --out artifacts/preds/lfm_read.jsonl
.venv/bin/python eval/score.py lfm:artifacts/preds/lfm_read.jsonl
```

## Metrics (`eval/score.py`)
**MER**, **JA-CER** (Japanese chars), **EN-WER** (English words), and **Script
Accuracy** — the headline metric: % of English words kept in Latin script vs.
wrongly forced into katakana. Identical normalization is applied to references
and all hypotheses.

## First result (n=20 smoke test)
Base LFM-JP zero-shot keeps only **16.7%** of English words in Latin (forces ~56%
into katakana) — the gap the fine-tune targets. See `CONTEXT.md` §5.

## Layout
`eval/` scoring + baselines · `data/` dataset loaders + human-eval protocol ·
`tests/` unit tests · `artifacts/` run outputs (git-ignored).
