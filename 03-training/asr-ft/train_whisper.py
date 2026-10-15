#!/usr/bin/env python3
"""LoRA fine-tune a Whisper checkpoint on the synthetic CS corpus.

Parameterised by model size so the same script trains whisper-tiny / base /
small / large-v3. Reads our engine + augmentation manifests (clean + augmented),
extracts log-mel features on the fly (no big feature cache -> disk-friendly),
and fine-tunes q_proj/v_proj LoRA with a Seq2SeqTrainer. Saves a PEFT adapter.

Details:
  * Decoder prefix. Labels are
      <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|> text <|endoftext|>
    and the collator strips the leading decoder_start_token_id, so the training
    prefix matches the inference prompt. --label-language sets <|lang|>:
      auto (default)  per row: its "lang" field, else ja if the transcript
                      contains Japanese, else en (CS rows are Japanese-matrix);
      ja | en         force one language;
      none            no language token (prefix SOT transcribe notimestamps).
  * Validation. The 5% validation split is group-aware (common.leak_guard):
    all engines and augmented copies of a transcript stay on one side, and
    check_disjoint verifies it. Or pass --dev-manifests to select checkpoints on
    an external dev set (e.g. CS-FLEURS xtts/test1 minus test overlap, plus FLEURS
    validation); training then uses every training row.
  * Training rows must pass the leak guard (test + dev roles), dev rows the test
    role; --license-scope commercial refuses non-commercial rows.
  * Seeds: python/numpy/torch plus the Trainer's seed and data_seed.
    <out>/run_record.json records git SHA, args, input hashes and versions.

Multi-GPU: `NCCL_SOCKET_IFNAME=lo torchrun --nproc_per_node=4 train_whisper.py ...` (the HF Trainer
runs DDP; rank 0 writes the run record and the final adapter). The effective
batch is --batch-size x --grad-accum x world size. Duration bucketing buys
nothing for Whisper, because the feature extractor always pads to 30 s.

Run (on mactrn01, on a free GPU):
  CUDA_VISIBLE_DEVICES=0 $PY 03-training/asr-ft/train_whisper.py \
     --model-id openai/whisper-small \
     --manifests data/synth/edge_tts/manifest.jsonl data/synth/kokoro/manifest.jsonl \
                 data/synth/melo/manifest.jsonl data/augmented/manifest.jsonl \
     --out models/trained/whisper-small --epochs 3 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.leak_guard import LeakGuard, check_disjoint, group_split, row_group_key  # noqa: E402
from common.licenses import enforce_scope, ensure_tags  # noqa: E402
from common.run_registry import set_seed, write_run_record  # noqa: E402

_JA = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
MAX_LABEL_TOKENS = 448          # Whisper decoder max_target_positions


def load_rows(manifests: list[str]) -> list[dict]:
    rows = []
    for m in manifests:
        for line in Path(m).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_language(row: dict, mode: str) -> str | None:
    if mode == "none":
        return None
    if mode in ("ja", "en"):
        return mode
    lang = str(row.get("lang") or "").lower()
    if lang.startswith(("ja", "jp")):
        return "ja"
    if lang.startswith("en"):
        return "en"
    return "ja" if _JA.search(row["transcript"]) else "en"


def label_ids(tokenizer, text: str, lang: str | None) -> list[int]:
    """<|startoftranscript|> [<|lang|>] <|transcribe|> <|notimestamps|> text <|endoftext|>"""
    specials = ["<|startoftranscript|>"] + ([f"<|{lang}|>"] if lang else []) + \
               ["<|transcribe|>", "<|notimestamps|>"]
    prefix = tokenizer.convert_tokens_to_ids(specials)
    if tokenizer.unk_token_id in prefix and tokenizer.unk_token_id != tokenizer.eos_token_id:
        raise ValueError(f"unknown special token in {specials}")
    return prefix + tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"{path}: {sr} Hz, expected {target_sr} (resample the corpus first)")
    return a


@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        feats = [{"input_features": b["input_features"]} for b in batch]
        out = self.processor.feature_extractor.pad(feats, return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
        lab = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        # Labels start with <|startoftranscript|>; the model prepends it again as
        # decoder_start_token_id when it shifts labels right, so drop it here.
        if (lab[:, 0] == self.decoder_start_token_id).all().cpu().item():
            lab = lab[:, 1:]
        out["labels"] = lab
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--dev-manifests", nargs="*", default=[],
                    help="External dev set for checkpoint selection (replaces the internal split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--label-language", choices=["auto", "ja", "en", "none"], default="auto")
    ap.add_argument("--license-scope", choices=["research", "commercial"], default="research")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="Keep every epoch checkpoint, linked as OUT/epoch-<k>; final/ is still "
                         "the dev-selected one")
    args = ap.parse_args()

    set_seed(args.seed)

    # ── data checks first: cheap, and they must pass before any GPU work ──────
    rows = enforce_scope(ensure_tags(load_rows(args.manifests)), args.license_scope, "train")
    LeakGuard().filter_rows(rows, on_leak="raise", context="train manifests")
    if args.dev_manifests:
        val_rows = enforce_scope(ensure_tags(load_rows(args.dev_manifests)), args.license_scope, "dev")
        LeakGuard(roles=("test",)).filter_rows(val_rows, on_leak="raise", context="dev manifests")
        train_rows = rows
    else:
        train_rows, val_rows = group_split(rows, args.val_frac, random.Random(args.seed),
                                           row_group_key)
    check_disjoint(train_rows, val_rows, context="train vs validation")
    print(f"[train] {len(train_rows)} train / {len(val_rows)} val clips")

    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from peft import LoraConfig, get_peft_model

    print(f"[train] model={args.model_id}")
    processor = WhisperProcessor.from_pretrained(args.model_id, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    tok = processor.tokenizer

    def to_examples(split_rows: list[dict], name: str) -> list[dict]:
        ex, langs, too_long = [], Counter(), 0
        for r in split_rows:
            lang = row_language(r, args.label_language)
            ids = label_ids(tok, r["transcript"], lang)
            if len(ids) > MAX_LABEL_TOKENS:
                too_long += 1
                continue
            langs[lang or "none"] += 1
            ex.append({"audio_path": r["audio_path"], "labels": ids})
        print(f"[train] {name}: label languages {dict(langs)}; dropped {too_long} over "
              f"{MAX_LABEL_TOKENS} label tokens")
        return ex

    train_ex, val_ex = to_examples(train_rows, "train"), to_examples(val_rows, "val")
    first = train_ex[0]["labels"]
    print(f"[train] label prefix: {tok.convert_ids_to_tokens(first[:4])} (SOT is stripped by the "
          f"collator; decoder_start_token_id={model.config.decoder_start_token_id})")

    class FeatureDataset(torch.utils.data.Dataset):
        """Log-mel features computed on access, in the Trainer's DataLoader workers.
        Same features as the former up-front Dataset.map pass (checked bit-identical),
        without a single-process pass that left the GPU idle for ~25 min per run."""

        def __init__(self, items: list[dict]):
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int) -> dict:
            ex = self.items[i]
            feats = processor.feature_extractor(load_audio(ex["audio_path"]),
                                                sampling_rate=16000).input_features[0]
            return {"input_features": feats, "labels": ex["labels"]}

    train_ds, val_ds = FeatureDataset(train_ex), FeatureDataset(val_ex)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none"))
    model.print_trainable_parameters()

    out = Path(args.out)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        write_run_record(out, vars(args), inputs=args.manifests + args.dev_manifests,
                         extra={"train_clips": len(train_ex), "val_clips": len(val_ex),
                                "world_size": world,
                                "effective_batch": args.batch_size * args.grad_accum * world,
                                "by_use": dict(Counter(r.get("license_use", "untagged")
                                                       for r in train_rows))})
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio, num_train_epochs=args.epochs,
        bf16=True, gradient_checkpointing=False,
        eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=None if args.keep_checkpoints else 1,
        logging_steps=25, report_to=[], remove_unused_columns=False,
        label_names=["labels"], dataloader_num_workers=4,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        predict_with_generate=False, seed=args.seed, data_seed=args.seed,
        ddp_find_unused_parameters=False,      # every q/v LoRA parameter gets a gradient
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        processing_class=processor.feature_extractor)

    trainer.train()
    if trainer.is_world_process_zero():
        final = out / "final"
        model.save_pretrained(str(final))
        processor.save_pretrained(str(final))
        (final / "run_record.json").write_text((out / "run_record.json").read_text(encoding="utf-8"),
                                               encoding="utf-8")
        print(f"[train] adapter saved -> {final}")
        if args.keep_checkpoints:
            # one checkpoint per epoch (save_strategy="epoch"), in step order
            cks = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
            for k, ck in enumerate(cks, 1):
                processor.save_pretrained(str(ck))   # checkpoints hold only the adapter
                link = out / f"epoch-{k}"
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(ck.name)
            print(f"[train] epoch checkpoints: {[c.name for c in cks]}")


if __name__ == "__main__":
    main()
