#!/bin/bash
# Host runner: low-density filter -> preprocess (JP) -> LoRA train -> eval JECS -> compare. Run on the GPU host.
K=/data/lfm2.5/run-c
A=/data/lfm2.5/run-a
VPY=$K/audio-venv/bin/python
source $K/env.sh
cd $K/repo/lfm
export CUDA_VISIBLE_DEVICES=1

# 1. low-density filter (use NON-augmented combined train; augmentation hurt)
$VPY scripts/filter_lowdensity.py --in data/cs_mixed/train.jsonl \
    --out data/cs_mixed/train_lowdens.jsonl --max-switches 1 || { echo FILTER_FAILED; exit 1; }
echo FILTER_DONE

# 2. preprocess (JP base)
$VPY scripts/preprocess_cs_asr.py --model-id LiquidAI/LFM2.5-Audio-1.5B-JP \
    --train-manifest data/cs_mixed/train_lowdens.jsonl --eval-manifest data/cs_mixed/eval.jsonl \
    --output-dir data/cs_mixed/pp_jp_lowdens --max-context 512 || { echo PREPROCESS_FAILED; exit 1; }
echo PREPROCESS_DONE

# 3. LoRA train (teammate's trainer; wandb auto-disabled without a key)
$VPY $A/train_lora.py --model_id LiquidAI/LFM2.5-Audio-1.5B-JP \
    --data data/cs_mixed/pp_jp_lowdens/train --val_data data/cs_mixed/pp_jp_lowdens/eval \
    --output_dir runs/cs_asr_jp_lora --max_steps 500 --lr 3e-5 --lora_r 8 --lora_alpha 16 \
    || { echo TRAIN_FAILED; exit 1; }
echo LORA_TRAIN_DONE

# 4. eval JECS with the adapter (base + LoRA)
$VPY scripts/eval_asr.py --model LiquidAI/LFM2.5-Audio-1.5B-JP --adapter runs/cs_asr_jp_lora/final \
    --manifest data/eval/jecs_neutral_cs.jsonl --out runs/jecs_jp_lora.json || { echo EVAL_FAILED; exit 1; }
echo JECS_EVAL_DONE

echo "=== JECS: base vs LoRA-lowdens ==="
$VPY scripts/eval_asr.py --compare runs/jecs_jp_base.json runs/jecs_jp_lora.json
echo "=== JECS: synthetic-FPFT vs LoRA-lowdens ==="
$VPY scripts/eval_asr.py --compare runs/jecs_jp_ft.json runs/jecs_jp_lora.json
echo POST_DONE
