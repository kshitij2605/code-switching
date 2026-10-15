#!/bin/bash
# Host runner: after augmented training finishes -> make final loadable -> eval on JECS -> compare. Run on the GPU host.
K=/data/lfm2.5/run-c
VPY=$K/audio-venv/bin/python
source $K/env.sh
cd $K/repo/lfm

while pgrep -f scripts/train_cs_asr.py >/dev/null; do sleep 20; done
echo TRAIN_AUG_FINISHED

JP_SNAP=$(ls -d $K/hf_cache/hub/models--LiquidAI--LFM2.5-Audio-1.5B-JP/snapshots/*/ | head -1)
for f in "$JP_SNAP"/*; do
  b=$(basename "$f")
  [ "$b" = model.safetensors ] && continue
  cp -rL "$f" runs/cs_asr_jp_aug/final/ 2>/dev/null
done
echo AUX_COPIED

export CUDA_VISIBLE_DEVICES=2
$VPY scripts/eval_asr.py --model runs/cs_asr_jp_aug/final \
    --manifest data/eval/jecs_neutral_cs.jsonl --out runs/jecs_jp_aug_ft.json
echo JECS_EVAL_DONE

echo "=== JECS: base vs aug-FT ==="
$VPY scripts/eval_asr.py --compare runs/jecs_jp_base.json runs/jecs_jp_aug_ft.json
echo "=== JECS: synthetic-only-FT vs aug-FT ==="
$VPY scripts/eval_asr.py --compare runs/jecs_jp_ft.json runs/jecs_jp_aug_ft.json
echo POST_DONE
