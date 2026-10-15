#!/bin/bash
# Host runner: LoRA recipe sweep gated on JECS. Builds a max-switches=2 dataset, trains 3 configs in parallel
# (GPU 1/2/3), evals each on JECS, prints a comparison vs base + the existing ms1/r8 run. Run on the GPU host.
set -u
K=/data/lfm2.5/run-c
A=/data/lfm2.5/run-a
JP=LiquidAI/LFM2.5-Audio-1.5B-JP
VPY=$K/audio-venv/bin/python
source $K/env.sh
cd $K/repo/lfm

# ---- max-switches=2 dataset (ms1 = data/cs_mixed/pp_jp_lowdens already exists) ----
$VPY scripts/filter_lowdensity.py --in data/cs_mixed/train.jsonl \
    --out data/cs_mixed/train_ms2.jsonl --max-switches 2 || { echo FILTER_FAILED; exit 1; }
CUDA_VISIBLE_DEVICES=1 $VPY scripts/preprocess_cs_asr.py --model-id $JP \
    --train-manifest data/cs_mixed/train_ms2.jsonl --eval-manifest data/cs_mixed/eval.jsonl \
    --output-dir data/cs_mixed/pp_jp_ms2 --max-context 512 || { echo PP_MS2_FAILED; exit 1; }
echo PP_MS2_DONE

# ---- 3 parallel LoRA trainings ----
CUDA_VISIBLE_DEVICES=1 nohup $VPY $A/train_lora.py --model_id $JP \
    --data data/cs_mixed/pp_jp_lowdens/train --val_data data/cs_mixed/pp_jp_lowdens/eval \
    --output_dir runs/lora_ms1_r16 --max_steps 800 --lr 3e-5 --lora_r 16 --lora_alpha 32 > $K/sw_A.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=2 nohup $VPY $A/train_lora.py --model_id $JP \
    --data data/cs_mixed/pp_jp_ms2/train --val_data data/cs_mixed/pp_jp_ms2/eval \
    --output_dir runs/lora_ms2_r8 --max_steps 500 --lr 3e-5 --lora_r 8 --lora_alpha 16 > $K/sw_B.log 2>&1 &
PB=$!
CUDA_VISIBLE_DEVICES=3 nohup $VPY $A/train_lora.py --model_id $JP \
    --data data/cs_mixed/pp_jp_ms2/train --val_data data/cs_mixed/pp_jp_ms2/eval \
    --output_dir runs/lora_ms2_r16 --max_steps 800 --lr 3e-5 --lora_r 16 --lora_alpha 32 > $K/sw_C.log 2>&1 &
PC=$!
wait $PA $PB $PC
echo SWEEP_TRAIN_DONE

# ---- eval each on JECS in parallel ----
CUDA_VISIBLE_DEVICES=1 nohup $VPY scripts/eval_asr.py --model $JP --adapter runs/lora_ms1_r16/final \
    --manifest data/eval/jecs_neutral_cs.jsonl --out runs/jecs_lora_ms1_r16.json > $K/ev_A.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $VPY scripts/eval_asr.py --model $JP --adapter runs/lora_ms2_r8/final \
    --manifest data/eval/jecs_neutral_cs.jsonl --out runs/jecs_lora_ms2_r8.json > $K/ev_B.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $VPY scripts/eval_asr.py --model $JP --adapter runs/lora_ms2_r16/final \
    --manifest data/eval/jecs_neutral_cs.jsonl --out runs/jecs_lora_ms2_r16.json > $K/ev_C.log 2>&1 &
wait
echo SWEEP_EVAL_DONE

echo "===== JECS SWEEP RESULTS (vs base; existing ms1/r8 = runs/jecs_jp_lora.json) ====="
for j in jecs_jp_base jecs_jp_lora jecs_lora_ms1_r16 jecs_lora_ms2_r8 jecs_lora_ms2_r16; do
  $VPY - "$j" <<'PY'
import json, sys
p = f"runs/{sys.argv[1]}.json"
try:
    s = json.load(open(p))["summary"]["all"]
    print(f"{sys.argv[1]:22} CER={s['CER']:.4f}  WER={s['WER']:.4f}  PIER={s['PIER']}")
except Exception as e:
    print(f"{sys.argv[1]:22} (missing: {e})")
PY
done
echo POST_DONE
