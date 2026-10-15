#!/bin/bash
# Host runner: CosyVoice2 native single-switch CS -> mix with mono -> preprocess -> LoRA (ms1/r8/500) -> eval JECS.
# Two venvs: cosyvoice-venv (generation) + audio-venv (train/eval). Run on the GPU host.
K=/data/lfm2.5/run-c
A=/data/lfm2.5/run-a
JP=LiquidAI/LFM2.5-Audio-1.5B-JP
AVPY=$K/audio-venv/bin/python
CVPY=$K/cosyvoice-venv/bin/python
cd $K/repo/lfm
source $K/env.sh
export CUDA_VISIBLE_DEVICES=1

# 1. generate CS natively with CosyVoice2 (single-switch transcripts come from train_lowdens.jsonl)
PYTHONPATH=$K/cosyvoice_repo:$K/cosyvoice_repo/third_party/Matcha-TTS $CVPY scripts/generate_cosyvoice_cs.py \
    --in data/cs_mixed/train_lowdens.jsonl --out-dir data/cosy_cs \
    --model $K/cosyvoice_repo/pretrained_models/CosyVoice2-0.5B \
    --prompt-wav $K/data/cosy_prompt/prompt_ja.wav --prompt-text-file $K/data/cosy_prompt/prompt_ja.txt \
    || { echo GEN_FAILED; exit 1; }
echo GEN_DONE $(wc -l < data/cosy_cs/manifest.jsonl) CS clips

# 2. mix: CosyVoice CS + the mono rows from train_lowdens (same mono as the winning run)
grep '"style": "mono' data/cs_mixed/train_lowdens.jsonl > data/cs_mixed/_mono_only.jsonl
cat data/cosy_cs/manifest.jsonl data/cs_mixed/_mono_only.jsonl > data/cs_mixed/train_cosy.jsonl
echo TRAIN_COSY_ROWS $(wc -l < data/cs_mixed/train_cosy.jsonl)

# 3. preprocess (JP)
$AVPY scripts/preprocess_cs_asr.py --model-id $JP \
    --train-manifest data/cs_mixed/train_cosy.jsonl --eval-manifest data/cs_mixed/eval.jsonl \
    --output-dir data/cs_mixed/pp_jp_cosy --max-context 512 || { echo PREPROCESS_FAILED; exit 1; }
echo PREPROCESS_DONE

# 4. LoRA train (winning recipe: r8, 500 steps)
$AVPY $A/train_lora.py --model_id $JP \
    --data data/cs_mixed/pp_jp_cosy/train --val_data data/cs_mixed/pp_jp_cosy/eval \
    --output_dir runs/cs_asr_jp_cosy --max_steps 500 --lr 3e-5 --lora_r 8 --lora_alpha 16 \
    || { echo TRAIN_FAILED; exit 1; }
echo LORA_TRAIN_DONE

# 5. eval JECS + compare
$AVPY scripts/eval_asr.py --model $JP --adapter runs/cs_asr_jp_cosy/final \
    --manifest data/eval/jecs_neutral_cs.jsonl --out runs/jecs_jp_cosy.json || { echo EVAL_FAILED; exit 1; }
echo JECS_EVAL_DONE
echo "=== JECS: base vs CosyVoice-CS ==="
$AVPY scripts/eval_asr.py --compare runs/jecs_jp_base.json runs/jecs_jp_cosy.json
echo "=== JECS: Kokoro-lowdens(best) vs CosyVoice-CS ==="
$AVPY scripts/eval_asr.py --compare runs/jecs_jp_lora.json runs/jecs_jp_cosy.json
echo POST_DONE
