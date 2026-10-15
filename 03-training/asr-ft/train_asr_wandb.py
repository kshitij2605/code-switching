"""
Fine-tunes LFM2.5-Audio-1.5B-JP for multilingual ASR using the official
liquid-audio Trainer, with Weights & Biases logging added.

Requires preprocessed data from preprocess_asr_data.py.

Usage:
  python train_asr_wandb.py \\
    --hf_token hf_xxx \\
    --wandb_key xxx \\
    [--model_id LiquidAI/LFM2.5-Audio-1.5B-JP] \\
    [--data data/preprocessed/train] \\
    [--val_data data/preprocessed/eval]
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import wandb
from huggingface_hub import login

from liquid_audio.data.dataloader import LFM2DataLoader
from liquid_audio.data.types import LFM2AudioModelInput
from liquid_audio.model.lfm2_audio import LFM2AudioModelOutput
from liquid_audio.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",      default="LiquidAI/LFM2.5-Audio-1.5B-JP")
    p.add_argument("--data",          default="data/preprocessed/train")
    p.add_argument("--val_data",      default="data/preprocessed/eval")
    p.add_argument("--output_dir",    default="checkpoints/lfm_asr")
    p.add_argument("--hf_token",      default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--wandb_key",     default=os.environ.get("WANDB_API_KEY", ""))
    p.add_argument("--wandb_project", default="lfm25-asr-jp")
    p.add_argument("--context_length", type=int,   default=512)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--max_steps",     type=int,   default=2000)
    p.add_argument("--warmup_steps",  type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--log_interval",  type=int,   default=10)
    p.add_argument("--save_interval", type=int,   default=200)
    p.add_argument("--val_interval",  type=int,   default=100)
    return p.parse_args()


class WandbTrainer(Trainer):
    """Trainer subclass that adds W&B logging on top of the official Trainer."""

    def __init__(self, *args, wandb_project: str = "lfm25-asr-jp", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wandb_project = wandb_project
        if self.accelerator.is_main_process:
            wandb.init(project=wandb_project, config=kwargs)

    def log(self, model_output: LFM2AudioModelOutput) -> None:
        super().log(model_output)
        if self.step > 0 and self.step % self.logging_interval == 0 and self.accelerator.is_main_process:
            train_loss = self.accelerator.reduce(
                model_output.loss.detach(), reduction="mean"
            ).item()
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/step": self.step,
                    "train/epoch": self.epoch,
                }
            )

    def validate(self) -> None:
        if self.val_loader is None:
            return

        loss_sum = torch.zeros(1, device=self.accelerator.device)
        loss_count = torch.zeros(1, device=self.accelerator.device)

        for batch in self.val_loader:
            batch = batch.to(self.accelerator.device)
            with self.accelerator.autocast():
                out = self.model(batch)
            loss_sum += out.loss.detach()
            loss_count += 1

        global_loss_sum = self.accelerator.reduce(loss_sum, reduction="sum")
        global_loss_count = self.accelerator.reduce(loss_count, reduction="sum")
        mean_val_loss = (global_loss_sum / global_loss_count.clamp_min(1)).item()

        total = int(time.monotonic() - self.time)
        mins, secs = divmod(total, 60)
        self.accelerator.print(
            f"[{mins:02d}:{secs:02d}] VALIDATION: step={self.step}/{self.max_steps} "
            f"val_loss={mean_val_loss:.4f}"
        )
        if self.accelerator.is_main_process:
            wandb.log({"eval/loss": mean_val_loss, "eval/step": self.step})


def main() -> None:
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    from pathlib import Path

    train_path = Path(args.data)
    val_path   = Path(args.val_data)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Preprocessed train dataset not found: {train_path}\n"
            "Run: python preprocess_asr_data.py first."
        )

    print(f"Loading train data: {train_path}")
    train_data = LFM2DataLoader(
        dataset_path=str(train_path),
        context_length=args.context_length,
    )

    val_data = None
    if val_path.exists():
        print(f"Loading eval data: {val_path}")
        val_data = LFM2DataLoader(
            dataset_path=str(val_path),
            context_length=args.context_length,
        )

    trainer = WandbTrainer(
        model_id          = args.model_id,
        train_data        = train_data,
        val_data          = val_data,
        lr                = args.lr,
        batch_size        = args.batch_size,
        max_steps         = args.max_steps,
        warmup_steps      = args.warmup_steps,
        dataloader_num_workers = args.num_workers,
        logging_interval  = args.log_interval,
        save_interval     = args.save_interval,
        val_interval      = args.val_interval,
        output_dir        = args.output_dir,
        wandb_project     = args.wandb_project,
    )

    print(f"\nStarting training for {args.max_steps} steps...")
    trainer.train()

    if trainer.accelerator.is_main_process:
        wandb.finish()
        print("Training complete.")


if __name__ == "__main__":
    main()
