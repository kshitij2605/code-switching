#!/bin/bash
# Host runner: waits for CosyVoice generation, then mix -> preprocess -> LoRA (ms1/r8/500) -> eval JECS. the GPU host.
K=/data/lfm2.5/run-c
A=/data/lfm2.5/run-a
JP=LiquidAI/LFM2.5-Audio-1.5B-JP
AVPY=$K/audio-venv/bin/python
cd $K/repo/lfm
source $K/env.sh
export CUDA_VISIBLE_DEVICES=1

while pgrep -f "[g]enerate_cosyvoice_cs" >/dev/null; do sleep 20; done
echo GEN_FINISHED $(wc -l < data/cosy_cs/manifest.jsonl)

grep '"style": "mono' data/cs_mixed/train_lowdens.jsonl > data/cs_mixed/_mono_only.jsonl
cat data/cosy_cs/manifest.jsonl data/cs_mixed/_mono_only.jsonl > data/cs_mixed/train_cosy.jsonl
echo TRAIN_COSY_ROWS $(wc -l < data/cs_mixed/train_cosy.jsonl)

rm -rf data/cs_mixed/pp_jp_cosy
$AVPY scripts/preprocess_cs_asr.py --model-id $JP \
    --train-manifest data/cs_mixed/train_cosy.jsonl --eval-manifest data/cs_mixed/eval.jsonl \
    --output-dir data/cs_mixed/pp_jp_cosy --max-context 512 || { echo PREPROCESS_FAILED; exit 1; }
echo PREPROCESS_DONE

$AVPY $A/train_lora.py --model_id $JP \
    --data data/cs_mixed/pp_jp_cosy/train --val_data data/cs_mixed/pp_jp_cosy/eval \
    --output_dir runs/cs_asr_jp_cosy --max_steps 500 --lr 3e-5 --lora_r 8 --lora_alpha 16 \
    || { echo TRAIN_FAILED; exit 1; }
echo LORA_TRAIN_DONE

$AVPY scripts/eval_asr.py --model $JP --adapter runs/cs_asr_jp_cosy/final \
    --manifest data/eval/jecs_neutral_cs.jsonl --out runs/jecs_jp_cosy.json || { echo EVAL_FAILED; exit 1; }
echo JECS_EVAL_DONE
echo "=== JECS: base vs CosyVoice-CS ==="
$AVPY scripts/eval_asr.py --compare runs/jecs_jp_base.json runs/jecs_jp_cosy.json
echo "=== JECS: Kokoro-lowdens(best) vs CosyVoice-CS ==="
$AVPY scripts/eval_asr.py --compare runs/jecs_jp_lora.json runs/jecs_jp_cosy.json
echo POST_DONE
