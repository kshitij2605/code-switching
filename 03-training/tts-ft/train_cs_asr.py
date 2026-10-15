#!/usr/bin/env python
"""Full fine-tune (FPFT) of LFM2.5-Audio for code-switch ASR, via liquid_audio.Trainer.

The liquid-audio Trainer is full-parameter only (AdamW over model.parameters() — no LoRA). We rely on the replay
mix + a low LR + val-based checkpoint selection to avoid catastrophic forgetting.

Pick the GPU with CUDA_VISIBLE_DEVICES (accelerate honors it):
  CUDA_VISIBLE_DEVICES=2 python scripts/train_cs_asr.py --model-id LiquidAI/LFM2.5-Audio-1.5B-JP \
      --data data/cs/preprocessed_jp --output-dir runs/cs_asr_jp
  CUDA_VISIBLE_DEVICES=1 python scripts/train_cs_asr.py --model-id LiquidAI/LFM2.5-Audio-1.5B \
      --data data/cs/preprocessed_en --output-dir runs/cs_asr_en --system-asr-en

context-length MUST match the value used in preprocessing (default 256).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="LiquidAI/LFM2.5-Audio-1.5B-JP")
    ap.add_argument("--data", required=True, help="preprocessed dir from preprocess_cs_asr.py (expects <dir>/train)")
    ap.add_argument("--output-dir", default="runs/cs_asr")
    ap.add_argument("--context-length", type=int, default=256, help="MUST match preprocessing --max-context")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5, help="low LR for a small dataset (trainer recipe uses 1e-4)")
    ap.add_argument("--max-steps", type=int, default=400, help="pipeline-proof default; scale up for a real run")
    ap.add_argument("--warmup-steps", type=int, default=40)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--val-interval", type=int, default=50)
    ap.add_argument("--logging-interval", type=int, default=10)
    args = ap.parse_args()

    from liquid_audio.data.dataloader import LFM2DataLoader
    from liquid_audio.trainer import Trainer

    data_root = Path(args.data)
    train_dir = data_root / "train"
    eval_dir = data_root / "eval"
    if not train_dir.exists():
        raise FileNotFoundError(f"No preprocessed train split at {train_dir}. Run scripts/preprocess_cs_asr.py first.")

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}  model={args.model_id}")
    train_data = LFM2DataLoader(dataset_path=str(train_dir), context_length=args.context_length)
    val_data = None
    if eval_dir.exists():
        val_data = LFM2DataLoader(dataset_path=str(eval_dir), context_length=args.context_length)
        print(f"val split: {eval_dir}")

    trainer = Trainer(
        model_id=args.model_id,
        train_data=train_data,
        val_data=val_data,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        dataloader_num_workers=args.num_workers,
        logging_interval=args.logging_interval,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        output_dir=args.output_dir,
    )
    trainer.train()
    print(f"\nDone. Final weights: {args.output_dir}/final  (checkpoints: {args.output_dir}/checkpoints)")


if __name__ == "__main__":
    main()
