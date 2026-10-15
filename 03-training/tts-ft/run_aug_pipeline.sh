#!/bin/bash
# Host runner: wait for augmentation -> combine -> preprocess (JP) -> train augmented run. Run on the GPU host.
K=/data/lfm2.5/run-c
VPY=$K/audio-venv/bin/python
source $K/env.sh
cd $K/repo/lfm

while pgrep -f augment_audio.py >/dev/null; do sleep 15; done
echo "AUGMENT_DONE $(ls data/aug/audio | wc -l) clips"

cat data/cs_mixed/train.jsonl data/aug/manifest.jsonl > data/cs_mixed/train_aug.jsonl
echo "TRAIN_AUG_ROWS $(wc -l < data/cs_mixed/train_aug.jsonl)"

export CUDA_VISIBLE_DEVICES=2
$VPY scripts/preprocess_cs_asr.py --model-id LiquidAI/LFM2.5-Audio-1.5B-JP \
    --train-manifest data/cs_mixed/train_aug.jsonl --eval-manifest data/cs_mixed/eval.jsonl \
    --output-dir data/cs_mixed/pp_jp_aug --max-context 512 || { echo PREPROCESS_FAILED; exit 1; }
echo PREPROCESS_DONE

$VPY scripts/train_cs_asr.py --model-id LiquidAI/LFM2.5-Audio-1.5B-JP \
    --data data/cs_mixed/pp_jp_aug --output-dir runs/cs_asr_jp_aug --context-length 512 \
    --batch-size 8 --lr 3e-5 --max-steps 1000 --warmup-steps 80 --save-interval 250 --val-interval 100 \
    || { echo TRAIN_FAILED; exit 1; }
echo PIPELINE_DONE
