"""
LoRA fine-tuning of LFM2.5-Audio-1.5B-JP — clean single-GPU implementation.

Uses lfm2_collator (from liquid-audio Trainer source) directly so batching
works identically to the official Trainer, but without the double-prepare bug.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_lora.py --hf_token hf_xxx --wandb_key xxx
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import wandb

BASE_MODEL = "LiquidAI/LFM2.5-Audio-1.5B-JP"

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "out_proj",  # lfm attention
    "in_proj",                                   # lfm SSM
    "w1", "w2", "w3",                            # lfm FFN
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",       default=BASE_MODEL)
    p.add_argument("--data",           default="data/preprocessed/train")
    p.add_argument("--val_data",       default="data/preprocessed/eval")
    p.add_argument("--output_dir",     default="checkpoints/lfm_lora")
    p.add_argument("--hf_token",       default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--wandb_key",      default=os.environ.get("WANDB_API_KEY", ""))
    p.add_argument("--wandb_project",  default="lfm25-asr-lora-jp")
    p.add_argument("--wandb_run_id",   default=None, help="Resume an existing W&B run")
    p.add_argument("--context_length", type=int,   default=512)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--max_steps",      type=int,   default=500)
    p.add_argument("--start_step",     type=int,   default=0,   help="Step to resume from")
    p.add_argument("--warmup_steps",   type=int,   default=50)
    p.add_argument("--lr",             type=float, default=3e-5)
    p.add_argument("--resume_adapter", default=None, help="Path to existing LoRA adapter to resume from")
    p.add_argument("--lora_r",         type=int,   default=8)
    p.add_argument("--lora_alpha",     type=int,   default=16)
    p.add_argument("--lora_dropout",   type=float, default=0.05)
    p.add_argument("--log_interval",   type=int,   default=10)
    p.add_argument("--val_interval",   type=int,   default=50)
    p.add_argument("--save_interval",  type=int,   default=100)
    return p.parse_args()


def get_lr(step: int, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def main() -> None:
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    # ── imports that need HF auth ─────────────────────────────────────────────
    from liquid_audio import LFM2AudioModel
    from liquid_audio.data.dataloader import LFM2DataLoader, lfm2_collator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── model ─────────────────────────────────────────────────────────────────
    print("Loading base model ...")
    model = LFM2AudioModel.from_pretrained(args.model_id, dtype=torch.bfloat16)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(device)

    if args.resume_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as sf_load
        print(f"Resuming adapter weights from: {args.resume_adapter}")
        adapter_sd = sf_load(f"{args.resume_adapter}/adapter_model.safetensors", device=str(device))
        incompatible = set_peft_model_state_dict(model, adapter_sd)
        if incompatible.unexpected_keys or incompatible.missing_keys:
            print(f"  unexpected={incompatible.unexpected_keys[:5]}  missing={incompatible.missing_keys[:5]}")
        else:
            print("  adapter weights loaded cleanly")

    # ── data ─────────────────────────────────────────────────────────────────
    print(f"Loading train data: {args.data}")
    train_ds = LFM2DataLoader(dataset_path=args.data, context_length=args.context_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lfm2_collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = None
    if Path(args.val_data).exists():
        print(f"Loading eval data:  {args.val_data}")
        val_ds = LFM2DataLoader(dataset_path=args.val_data, context_length=args.context_length)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lfm2_collator,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    # ── optimizer ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # ── W&B ──────────────────────────────────────────────────────────────────
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    wandb_kwargs: dict = dict(
        project=args.wandb_project,
        config=vars(args),
        mode="online" if args.wandb_key else "disabled",
    )
    if args.wandb_run_id:
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "must"
    wandb.init(**wandb_kwargs)

    # ── training loop ─────────────────────────────────────────────────────────
    min_lr = args.lr * 0.1
    step = args.start_step
    epoch = 0
    t0 = time.monotonic()

    resume_str = f" (resuming from step {args.start_step})" if args.start_step else ""
    print(f"\nLoRA training — steps {args.start_step}→{args.max_steps} on {device}{resume_str}\n")

    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.max_steps:
                break

            # LR schedule
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch = batch.to(device)
            model.train()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch)

            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            step += 1

            # log
            if step % args.log_interval == 0:
                m, s = divmod(int(time.monotonic() - t0), 60)
                loss_v = loss.item()
                print(f"[{m:02d}:{s:02d}] TRAIN epoch={epoch} step={step}/{args.max_steps}  loss={loss_v:.4f}  lr={lr:.2e}")
                wandb.log({"train/loss": loss_v, "train/lr": lr}, step=step)

            # validate
            if val_loader and step % args.val_interval == 0:
                model.eval()
                v_sum, v_n = 0.0, 0
                with torch.no_grad():
                    for vb in val_loader:
                        vb = vb.to(device)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            vo = model(vb)
                        v_sum += vo.loss.item()
                        v_n   += 1
                val_loss = v_sum / max(v_n, 1)
                m, s = divmod(int(time.monotonic() - t0), 60)
                print(f"[{m:02d}:{s:02d}] VALIDATION step={step}/{args.max_steps}  val_loss={val_loss:.4f}")
                wandb.log({"eval/loss": val_loss}, step=step)

            # save adapter checkpoint
            if step % args.save_interval == 0:
                ckpt = Path(args.output_dir) / f"step_{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt))
                print(f"  → adapter saved: {ckpt}")

    # ── final save ────────────────────────────────────────────────────────────
    final = Path(args.output_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    m, s = divmod(int(time.monotonic() - t0), 60)
    print(f"\n[{m:02d}:{s:02d}] LoRA adapter saved: {final}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
