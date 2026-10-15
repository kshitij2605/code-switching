# 04 · Evaluation

## eval-ft/ — the shared benchmark (primary scoreboard)
The frozen, held-out evaluation harness every model is scored on. Training data is prepared
separately, so the whole CS-FLEURS slice stays a clean held-out test (no leakage).
- `eval/score.py` — metrics: **MER**, **JA-CER**, **EN-WER**, and the headline **Script Accuracy**
  (% of English words kept in Latin vs wrongly forced into katakana). Unit-tested (`tests/test_score.py`).
- `eval/run_baseline.py` — run LFM or Whisper over a manifest -> predictions.
- `eval/run_whisper_hf.py` — Whisper large-v3 baseline (transformers/GPU).
- `eval/leaderboard.py`, `eval/leaderboard_app.py` — aggregate scoreboard.
- `data/load_csfleurs.py`, `data/load_fleurs.py` — torchcodec-free loaders (HF-direct).
- `data/eval/*.jsonl` — three frozen subsets: `csfleurs_jaen_read196` (CS),
  `fleurs_ja_jp_test200` (JA control), `fleurs_en_us_test200` (EN control).
- `docs/human-eval-protocol.md` — protocol for the real human gold set.

**Standard protocol:** every model uses the `"Perform ASR."` system prompt (the base LFM is
strongly prompt-sensitive). See the result table in the top-level `README.md`.

## asr-ft/ — training-side evals
- `eval_lora.py` (CS-FLEURS), `eval_synthetic.py` (held-out synthetic), `eval_finetuned.py`,
  `sample_compare.py`, `sample_tts_check.py`, `lora_tts_check.py`.

## tts-ft/ — app / adapter evals
- `eval_asr.py`, `eval_translation.py` (chrF/BLEU on FLEURS), adapter/pipeline benchmarks
  (`benchmark_*`, `compare_*`), and the option-B/C/A probes.
