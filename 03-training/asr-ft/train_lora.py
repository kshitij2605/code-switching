"""
LoRA fine-tuning of LFM2.5-Audio-1.5B-JP — clean single-GPU implementation.

Uses lfm2_collator (from liquid-audio Trainer source) directly so batching
works identically to the official Trainer, but without the double-prepare bug.

Reproducibility:
  * --seed seeds python/numpy/torch and the data order.
  * <output_dir>/run_record.json records the git SHA, arguments, library
    versions and the preprocessing build records of --data and --val_data
    (these hold the hashes of the source JSONL files).
  * Whenever the loss on --val_data (a dev set, never test) improves, the
    adapter is saved to <output_dir>/best/, with the step recorded in
    best/selection.json.
  * --lora_scope both|lm|encoder places LoRA on the whole model, the decoder
    only, or the conformer only.

Multi-GPU (data parallel):
  NCCL_SOCKET_IFNAME=lo torchrun --nproc_per_node=4 train_lora.py ...
(On mactrn01 the login environment sets NCCL_SOCKET_IFNAME=eno1, an interface that
does not exist there; NCCL then fails with "Bootstrap : no socket interface found".
Use the loopback interface for single-node runs.)
With torchrun each rank trains on a DistributedSampler shard and the dev loss
is all-reduced; rank 0 logs, writes the run record and saves adapters. The
effective batch is --batch_size x world size, and the LR is NOT rescaled
automatically. A single process runs exactly the single-GPU code path.
Duration bucketing buys nothing here, because liquid-audio pads every item to
--context_length.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_lora.py --hf_token hf_xxx --wandb_key xxx
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from huggingface_hub import login
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.run_registry import set_seed, write_run_record  # noqa: E402

BASE_MODEL = "LiquidAI/LFM2.5-Audio-1.5B-JP"

# PEFT matches these as module-name suffixes. In LFM2.5-Audio-1.5B-JP they hit
# lfm (6 attention + 10 conv layers, 16 FFNs), conformer (17 layers) and the
# audio-output depthformer (out_proj/w1-3); the depthformer adapters get no
# gradient from ASR-only targets. audio_adapter (the connector) is not adapted.
LM_TARGETS = [
    # LM / depthformer side
    "q_proj", "k_proj", "v_proj", "out_proj",  # attention
    "in_proj",                                   # SSM
    "w1", "w2", "w3",                            # FFN
]
ENCODER_TARGETS = [
    # Conformer encoder side (RelPositionMultiHeadAttention + FFN)
    "linear_q", "linear_k", "linear_v", "linear_out",  # encoder self-attn
    "linear1", "linear2",                               # encoder FFN
]
LORA_TARGETS = LM_TARGETS + ENCODER_TARGETS
SCOPE_TARGETS = {"both": LORA_TARGETS, "lm": LM_TARGETS, "encoder": ENCODER_TARGETS}


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
    p.add_argument("--encoder_lora_r", type=int,   default=None,
                   help="LoRA rank for conformer encoder modules (default: same as lora_r)")
    p.add_argument("--encoder_lr",     type=float, default=None,
                   help="LR for conformer encoder LoRA params (default: lr/5)")
    p.add_argument("--log_interval",   type=int,   default=10)
    p.add_argument("--val_interval",   type=int,   default=50)
    p.add_argument("--save_interval",  type=int,   default=100)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--lora_scope",     choices=sorted(SCOPE_TARGETS), default="both",
                   help="Where to put LoRA: both (default), lm (decoder only) or encoder (conformer only)")
    return p.parse_args()


def build_records(dataset_path: str) -> dict | None:
    """The <split>.build.json that preprocess_asr_data.py wrote next to the dataset."""
    p = Path(dataset_path)
    rec = p.parent / f"{p.name}.build.json"
    return json.loads(rec.read_text(encoding="utf-8")) if rec.exists() else None


def get_lr(step: int, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.hf_token:
        login(token=args.hf_token)
    import wandb

    # ── imports that need HF auth ─────────────────────────────────────────────
    from liquid_audio import LFM2AudioModel
    from liquid_audio.data.dataloader import LFM2DataLoader, lfm2_collator

    # ── optional multi-GPU (torchrun) ─────────────────────────────────────────
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        if rank != 0:
            sys.stdout = open(os.devnull, "w")      # rank 0 logs, records and saves
    device = (torch.device(f"cuda:{local_rank}") if distributed
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}" + (f" (DDP, world size {world}; effective batch "
                                 f"{args.batch_size * world})" if distributed else ""))

    # ── model ─────────────────────────────────────────────────────────────────
    print("Loading base model ...")
    model = LFM2AudioModel.from_pretrained(args.model_id, dtype=torch.bfloat16)

    enc_r = args.encoder_lora_r if args.encoder_lora_r is not None else args.lora_r
    ENCODER_MODULES = ENCODER_TARGETS
    rank_pattern  = {m: enc_r for m in ENCODER_MODULES} if enc_r != args.lora_r else {}
    alpha_pattern = {m: enc_r * 2 for m in ENCODER_MODULES} if enc_r != args.lora_r else {}
    print(f"LoRA scope: {args.lora_scope} — rank LM: {args.lora_r}, encoder: {enc_r}")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=SCOPE_TARGETS[args.lora_scope],
        bias="none",
        rank_pattern=rank_pattern if rank_pattern else None,
        alpha_pattern=alpha_pattern if alpha_pattern else None,
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

    core = model                     # the PEFT model; `model` becomes the DDP wrapper if distributed
    if distributed:
        # find_unused_parameters: the depthformer's LoRA adapters get no gradient from ASR targets
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)

    # ── data ─────────────────────────────────────────────────────────────────
    from torch.utils.data.distributed import DistributedSampler
    print(f"Loading train data: {args.data}")
    train_ds = LFM2DataLoader(dataset_path=args.data, context_length=args.context_length)
    train_sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                        seed=args.seed) if distributed else None)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=None if distributed else torch.Generator().manual_seed(args.seed),
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
            sampler=(DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
                     if distributed else None),
            collate_fn=lfm2_collator,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    # ── optimizer — two param groups: lower LR for encoder LoRA ──────────────
    enc_lr = args.encoder_lr if args.encoder_lr is not None else args.lr / 5
    lm_params, enc_params = [], []
    for name, param in core.named_parameters():
        if not param.requires_grad:
            continue
        if "conformer" in name:
            enc_params.append(param)
        else:
            lm_params.append(param)
    print(f"Param groups — LM LoRA: {len(lm_params)}, encoder LoRA: {len(enc_params)} (enc_lr={enc_lr:.2e})")
    need_lm, need_enc = args.lora_scope in ("both", "lm"), args.lora_scope in ("both", "encoder")
    if bool(lm_params) != need_lm or bool(enc_params) != need_enc:
        raise RuntimeError(f"LoRA scope {args.lora_scope!r} but LM tensors={len(lm_params)}, "
                           f"encoder tensors={len(enc_params)}: the 'conformer' name split no "
                           f"longer matches this model's module names")
    if rank == 0:
        write_run_record(args.output_dir, vars(args),
                         extra={"lm_lora_tensors": len(lm_params), "encoder_lora_tensors": len(enc_params),
                                "trainable_params": sum(p.numel() for p in lm_params + enc_params),
                                "world_size": world, "effective_batch": args.batch_size * world,
                                "data_build": build_records(args.data),
                                "val_data_build": build_records(args.val_data)})
    optimizer = torch.optim.AdamW(
        [{"params": lm_params, "lr": args.lr},
         {"params": enc_params, "lr": enc_lr}],
        weight_decay=0.01,
    )
    trainable = lm_params + enc_params

    # ── W&B ──────────────────────────────────────────────────────────────────
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    wandb_kwargs: dict = dict(
        project=args.wandb_project,
        config=vars(args),
        mode="online" if args.wandb_key and rank == 0 else "disabled",
    )
    if args.wandb_run_id:
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "must"
    wandb.init(**wandb_kwargs)

    # ── training loop ─────────────────────────────────────────────────────────
    min_lr = args.lr * 0.1
    step = args.start_step
    epoch = 0
    best_val = math.inf     # checkpoint selection on --val_data (a dev set, never test)
    t0 = time.monotonic()

    resume_str = f" (resuming from step {args.start_step})" if args.start_step else ""
    print(f"\nLoRA training — steps {args.start_step}→{args.max_steps} on {device}{resume_str}\n")

    while step < args.max_steps:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= args.max_steps:
                break

            # LR schedule — cosine for both groups, encoder scaled proportionally
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr)
            lr_scale = lr / args.lr
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = enc_lr * lr_scale

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
                wandb.log({"train/loss": loss_v, "train/lr": lr,
                           "train/enc_lr": optimizer.param_groups[1]["lr"]}, step=step)

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
                if distributed:     # every rank saw a shard of the dev set: pool the sums
                    t = torch.tensor([v_sum, float(v_n)], device=device)
                    torch.distributed.all_reduce(t)
                    v_sum, v_n = t[0].item(), int(t[1].item())
                val_loss = v_sum / max(v_n, 1)
                m, s = divmod(int(time.monotonic() - t0), 60)
                print(f"[{m:02d}:{s:02d}] VALIDATION step={step}/{args.max_steps}  val_loss={val_loss:.4f}")
                wandb.log({"eval/loss": val_loss}, step=step)
                if val_loss < best_val:
                    best_val = val_loss
                    if rank == 0:
                        best_dir = Path(args.output_dir) / "best"
                        core.save_pretrained(str(best_dir))
                        (best_dir / "selection.json").write_text(json.dumps(
                            {"step": step, "val_loss": val_loss, "selected_on": args.val_data}),
                            encoding="utf-8")
                        print(f"  → new best adapter (val_loss={val_loss:.4f}): {best_dir}")

            # save adapter checkpoint
            if step % args.save_interval == 0 and rank == 0:
                ckpt = Path(args.output_dir) / f"step_{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                core.save_pretrained(str(ckpt))
                print(f"  → adapter saved: {ckpt}")

    # ── final save ────────────────────────────────────────────────────────────
    if rank == 0:
        final = Path(args.output_dir) / "final"
        final.mkdir(parents=True, exist_ok=True)
        core.save_pretrained(str(final))
        m, s = divmod(int(time.monotonic() - t0), 60)
        print(f"\n[{m:02d}:{s:02d}] LoRA adapter saved: {final}")

    wandb.finish()
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print("Done.")


if __name__ == "__main__":
    main()
