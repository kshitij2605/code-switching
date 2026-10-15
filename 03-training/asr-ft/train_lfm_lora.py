"""
LoRA fine-tuning of LFM2.5-Audio-1.5B-JP for multilingual ASR + translation.

Tasks trained simultaneously:
  - "Transcribe."             → native language transcription
  - "Transcribe in Japanese." → forced Japanese output
  - "Transcribe in English."  → forced English output

Usage:
  python train_lfm_lora.py --hf_token hf_xxx --wandb_key xxx [--model_id LiquidAI/LFM2.5-Audio-1.5B-JP]
"""

import argparse
import json
import math
import os
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import wandb
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",    default="LiquidAI/LFM2.5-Audio-1.5B-JP")
    p.add_argument("--train_data",  default="data/training/train.jsonl")
    p.add_argument("--eval_data",   default="data/training/eval.jsonl")
    p.add_argument("--output_dir",  default="checkpoints/lfm_lora")
    p.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--wandb_key",   default=os.environ.get("WANDB_API_KEY", ""))
    p.add_argument("--wandb_project", default="lfm25-asr-lora")
    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Training
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--warmup_steps", type=int,   default=50)
    p.add_argument("--save_steps",   type=int,   default=200)
    p.add_argument("--eval_steps",   type=int,   default=100)
    p.add_argument("--max_audio_s",  type=float, default=30.0)
    p.add_argument("--resume",       action="store_true")
    return p.parse_args()


# ── Audio loading ─────────────────────────────────────────────────────────────

TARGET_SR = 16000

def load_wav(path: str, max_seconds: float = 30.0) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    max_samples = int(max_seconds * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio, sr


# ── Dataset ───────────────────────────────────────────────────────────────────

class ASRDataset(Dataset):
    def __init__(self, jsonl_path: str, max_audio_s: float = 30.0):
        self.examples = []
        self.max_audio_s = max_audio_s
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    if Path(ex["audio_file"]).exists():
                        self.examples.append(ex)
        print(f"  Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        audio, sr = load_wav(ex["audio_file"], self.max_audio_s)
        return {
            "audio":   torch.from_numpy(audio).unsqueeze(0),  # (1, T)
            "sr":      sr,
            "system":  ex["system"],
            "target":  ex["target"],
            "type":    ex.get("type", "unknown"),
        }


# ── Model introspection helpers ───────────────────────────────────────────────

def find_lora_targets(model) -> list[str]:
    """Auto-detect linear layer names suitable for LoRA."""
    import re
    candidates = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last = name.split(".")[-1]
            # skip very small projection layers (embed, lm_head)
            if module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                candidates.add(last)
    # prefer well-known SSM / attention names
    preferred = [
        "in_proj", "out_proj", "x_proj", "dt_proj",
        "q_proj",  "k_proj",  "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    targets = [n for n in preferred if n in candidates]
    if not targets:
        targets = list(candidates)[:8]
    print(f"  LoRA targets: {targets}")
    return targets


# ── Training helpers ──────────────────────────────────────────────────────────

def build_chat_input(processor, model, batch, device, max_new_tokens):
    """
    Build input tensors for one example using ChatState.
    Returns (input_ids, audio_features, assistant_start_idx) or None on error.
    """
    from liquid_audio import ChatState, LFMModality
    try:
        chat = ChatState(processor)

        # system turn
        chat.new_turn("system")
        chat.add_text(batch["system"])
        chat.end_turn()

        # user turn — audio
        chat.new_turn("user")
        chat.add_audio(batch["audio"].to(device), batch["sr"])
        chat.end_turn()

        # assistant turn header (empty — we supply the target as labels)
        chat.new_turn("assistant")

        return chat
    except Exception as e:
        print(f"  [WARN] build_chat_input failed: {e}")
        return None


def compute_loss(model, processor, batch, device, max_new_tokens):
    """
    Teacher-forced forward pass.
    Encode target tokens, append to chat sequence, compute CE loss on target.
    """
    from liquid_audio import ChatState

    chat = build_chat_input(processor, model, batch, device, max_new_tokens)
    if chat is None:
        return None

    # Encode target text to token ids
    target_ids = processor.text.encode(batch["target"], return_tensors="pt").input_ids.to(device)
    if target_ids.shape[1] == 0:
        return None

    # Try to run a forward pass with teacher forcing
    # liquid-audio exposes **chat as kwargs to model.forward
    try:
        # Build full sequence: [system | audio | assistant | target]
        # We pass target_ids as decoder_input_ids if the model is encoder-decoder,
        # or append to input_ids for decoder-only.
        chat_kwargs = dict(**chat)

        # Attempt standard causal LM forward
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model.forward(
                **chat_kwargs,
                decoder_input_ids=target_ids,
                labels=target_ids,
            )
        if hasattr(out, "loss") and out.loss is not None:
            return out.loss

        # Fallback: manual CE on logits
        if hasattr(out, "logits"):
            logits = out.logits[:, -target_ids.shape[1]:, :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=processor.text.pad_token_id or -100,
            )
            return loss

    except TypeError:
        # Try without decoder_input_ids (decoder-only model)
        try:
            # Append target tokens to input and create shifted labels
            input_ids  = chat_kwargs.get("input_ids", None)
            if input_ids is None:
                return None
            full_ids   = torch.cat([input_ids, target_ids], dim=1)
            labels     = torch.full_like(full_ids, -100)
            labels[:, -target_ids.shape[1]:] = target_ids
            chat_kwargs["input_ids"] = full_ids

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model.forward(**chat_kwargs, labels=labels)
            if hasattr(out, "loss") and out.loss is not None:
                return out.loss
        except Exception as e2:
            print(f"  [WARN] forward fallback failed: {e2}")

    except Exception as e:
        print(f"  [WARN] forward failed: {e}")

    return None


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, processor, eval_loader, device, max_new_tokens, n_batches=50):
    from jiwer import cer
    model.eval()
    total_loss = 0.0
    n = 0
    refs, hyps = [], []

    for batch in list(eval_loader)[:n_batches]:
        loss = compute_loss(model, processor, batch, device, max_new_tokens)
        if loss is not None:
            total_loss += loss.item()
            n += 1

        # Generate one sample for CER
        if len(refs) < 10:
            try:
                from liquid_audio import ChatState
                chat = build_chat_input(processor, model, batch, device, max_new_tokens)
                if chat:
                    tokens = []
                    for t in model.generate_sequential(**chat, max_new_tokens=max_new_tokens):
                        if t.numel() == 1:
                            tokens.append(processor.text.decode(t))
                    hyp = "".join(tokens).strip()
                    refs.append(batch["target"])
                    hyps.append(hyp)
            except Exception:
                pass

    avg_loss = total_loss / max(n, 1)
    avg_cer  = cer(refs, hyps) if refs else 1.0
    model.train()
    return avg_loss, avg_cer


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Auth ─────────────────────────────────────────────────────────────────
    if args.hf_token:
        from huggingface_hub import login
        login(args.hf_token)
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"lora_r{args.lora_r}_lr{args.lr}",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Load model + processor ───────────────────────────────────────────────
    print(f"\nLoading {args.model_id}...")
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    processor = LFM2AudioProcessor.from_pretrained(args.model_id)
    model     = LFM2AudioModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.train()

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    print("\nApplying LoRA...")
    lora_targets = find_lora_targets(model)
    lora_cfg = LoraConfig(
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        target_modules = lora_targets,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = ASRDataset(args.train_data,  max_audio_s=args.max_audio_s)
    eval_ds  = ASRDataset(args.eval_data,   max_audio_s=args.max_audio_s)

    # batch_size=1 because audio lengths vary — collation is handled per-sample
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0)
    eval_loader  = DataLoader(eval_ds,  batch_size=1, shuffle=False, num_workers=0)

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    out_path   = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if args.resume and (out_path / "adapter_config.json").exists():
        print(f"Resuming from {out_path}")
        model.load_adapter(str(out_path))

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs, {total_steps} optimizer steps...")
    global_step  = start_step
    accum_loss   = 0.0
    accum_count  = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        print(f"\n── Epoch {epoch+1}/{args.epochs} ──")
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):

            # Squeeze batch dim added by DataLoader
            sample = {
                "audio":  batch["audio"].squeeze(0),
                "sr":     batch["sr"][0].item() if isinstance(batch["sr"], torch.Tensor) else batch["sr"][0],
                "system": batch["system"][0],
                "target": batch["target"][0],
                "type":   batch["type"][0],
            }

            loss = compute_loss(model, processor, sample, device, args.max_new_tokens)
            if loss is None:
                continue

            loss = loss / args.grad_accum
            loss.backward()

            accum_loss  += loss.item() * args.grad_accum
            accum_count += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg = accum_loss / accum_count
                wandb.log({
                    "train/loss": avg,
                    "train/lr":   scheduler.get_last_lr()[0],
                    "train/step": global_step,
                    "train/epoch": epoch + 1,
                })
                accum_loss  = 0.0
                accum_count = 0

                # ── Eval ─────────────────────────────────────────────────────
                if global_step % args.eval_steps == 0:
                    eval_loss, eval_cer = run_eval(
                        model, processor, eval_loader, device, args.max_new_tokens
                    )
                    print(f"  step {global_step} | eval_loss={eval_loss:.4f} | eval_cer={eval_cer:.4f}")
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/cer":  eval_cer,
                        "eval/step": global_step,
                    })

                # ── Save ──────────────────────────────────────────────────────
                if global_step % args.save_steps == 0:
                    ckpt = out_path / f"step_{global_step}"
                    model.save_pretrained(str(ckpt))
                    processor.save_pretrained(str(ckpt))
                    print(f"  Saved checkpoint → {ckpt}")

    # ── Final save ────────────────────────────────────────────────────────────
    model.save_pretrained(str(out_path / "final"))
    processor.save_pretrained(str(out_path / "final"))
    print(f"\nTraining complete. Final adapter → {out_path}/final")
    wandb.finish()


if __name__ == "__main__":
    main()
