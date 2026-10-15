#!/usr/bin/env bash
# Full synthetic-CS corpus pipeline (run on mactrn01, from /data/code-switching).
# Renders the current transcript set across all 3 engines, then multi-augments.
# MeloTTS (CPU) runs in parallel with Kokoro (GPU) + edge-tts (cloud).
#
#   bash scripts/01-data-generation/asr-ft/run_full_pipeline.sh 300
#
set -u
ROOT=/data/code-switching
cd "$ROOT"
LIMIT="${1:-300}"
GPU="${2:-0}"
S=scripts/01-data-generation/asr-ft/engines
LOG=data/logs
mkdir -p "$LOG"
TR=data/transcripts/cs_transcripts.jsonl

echo "[$(date +%T)] pipeline start: limit=$LIMIT gpu=$GPU"

# clean old audio so no orphan wavs survive a re-render
rm -rf data/synth/edge_tts/audio data/synth/kokoro/audio data/synth/melo/audio data/augmented/audio

# MeloTTS on CPU, in the background (its own venv)
echo "[$(date +%T)] launching MeloTTS (CPU, background)"
.venv-melo/bin/python "$S/render_melo.py" --in "$TR" --out data/synth/melo --limit "$LIMIT" > "$LOG/render_melo.log" 2>&1 &
MELO_PID=$!

# Kokoro on GPU
echo "[$(date +%T)] Kokoro (GPU $GPU) ..."
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python "$S/render_kokoro.py" --in "$TR" --out data/synth/kokoro --limit "$LIMIT" > "$LOG/render_kokoro.log" 2>&1
echo "[$(date +%T)] Kokoro done -> $(ls data/synth/kokoro/audio | wc -l) clips"

# edge-tts (cloud, proxied)
echo "[$(date +%T)] edge-tts (cloud) ..."
.venv/bin/python "$S/render_edge_tts.py" --in "$TR" --out data/synth/edge_tts --limit "$LIMIT" > "$LOG/render_edge.log" 2>&1
echo "[$(date +%T)] edge-tts done -> $(ls data/synth/edge_tts/audio | wc -l) clips"

# wait for MeloTTS to finish
wait "$MELO_PID"
echo "[$(date +%T)] MeloTTS done -> $(ls data/synth/melo/audio | wc -l) clips"

# multi-type augmentation over all 3 clean manifests
echo "[$(date +%T)] augmenting ..."
.venv/bin/python scripts/02-preprocessing/asr-ft/augment_multi.py \
  --manifests data/synth/edge_tts/manifest.jsonl data/synth/kokoro/manifest.jsonl data/synth/melo/manifest.jsonl \
  --musan data/noise_sources/musan --rir data/noise_sources/RIRS_NOISES \
  --out data/augmented --seed 42 > "$LOG/augment.log" 2>&1

CLEAN=$(cat data/synth/*/manifest.jsonl | wc -l)
AUG=$(wc -l < data/augmented/manifest.jsonl)
echo "[$(date +%T)] PIPELINE_DONE  clean=$CLEAN augmented=$AUG total=$((CLEAN+AUG))"
