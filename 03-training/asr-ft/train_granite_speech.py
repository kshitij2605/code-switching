#!/usr/bin/env python3
"""Fine-tune Granite-4.0-1b-speech on our CS corpus, under the same data contract
as train_whisper.py.

Recipe base: IBM's notebook ibm-granite/granite-speech-models
notebooks/fine_tuning_granite_speech.ipynb (@3306e51, Apache-2.0):
  * collator: the prompt (with <|audio|>) left-padded, the target + eos
    right-padded, concatenated, with labels only on the target;
  * the speech encoder frozen; the projector (q-former) trained; lr 3e-5, warmup
    0.2, batch 16 x grad-accum 2, bf16.
The notebook also trains parameters whose name contains "lora". The released 4.0-1b
checkpoint has has_lora_adapter = false (its LoRA is merged), so that rule matches
nothing, and the notebook's recipe is projector-only (--llm-lora-r 0, the default).
--llm-lora-r R adds a PEFT LoRA of rank R (alpha 2R) on the LLM's q_proj/v_proj,
the analogue of our Whisper LoRA. It is merged into the weights before saving.

The prompt is the model card's ("<|audio|>can you transcribe the speech into a
written format?"), as in eval/run_audio_llm.py, so OUT/final evaluates with:
  run_audio_llm.py --family granite --model-id OUT/final

Added for our protocol: the leak guard and licence scope, --dev-manifests with
per-epoch evaluation and lowest-eval-loss selection, seeds, OUT/run_record.json,
and OUT/final with the processor and selection.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.leak_guard import LeakGuard, check_disjoint, group_split, row_group_key  # noqa: E402
from common.licenses import enforce_scope, ensure_tags  # noqa: E402
from common.run_registry import set_seed, write_run_record  # noqa: E402

PROMPT = "<|audio|>can you transcribe the speech into a written format?"


class ListDataset(torch.utils.data.Dataset):
    """A list of example dicts as a torch Dataset (the Trainer needs no `datasets`)."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def load_rows(manifests: list[str]) -> list[dict]:
    rows = []
    for m in manifests:
        for line in Path(m).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    a, s = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if s != sr:
        import torchaudio
        a = torchaudio.functional.resample(torch.from_numpy(a), s, sr).numpy()
    return a


class Collator:
    """IBM notebook collator: left-padded prompt + right-padded target; loss on target only."""

    def __init__(self, processor, prompt_text: str):
        self.processor, self.prompt_text = processor, prompt_text

    def __call__(self, feats: list[dict]) -> dict:
        from transformers.feature_extraction_utils import BatchFeature
        tok = self.processor.tokenizer
        audios = [torch.from_numpy(load_audio(f["audio"])) for f in feats]
        p = self.processor([self.prompt_text] * len(feats), audios, return_tensors="pt",
                           padding=True, padding_side="left")
        t = tok([f["target"] + tok.eos_token for f in feats], return_tensors="pt", padding=True,
                padding_side="right", add_special_tokens=False)
        labels = t.input_ids.clone()
        labels[~t.attention_mask.bool()] = -100
        return BatchFeature(data={
            "input_ids": torch.cat([p.input_ids, t.input_ids], dim=1),
            "attention_mask": torch.cat([p.attention_mask, t.attention_mask], dim=1),
            "labels": torch.cat([torch.full_like(p.input_ids, -100), labels], dim=1),
            "input_features": p.input_features, "input_features_mask": p.input_features_mask})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="ibm-granite/granite-4.0-1b-speech")
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--dev-manifests", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--license-scope", choices=["research", "commercial"], default="research")
    ap.add_argument("--llm-lora-r", type=int, default=0, help="0 = projector only (IBM notebook recipe)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-projector", action="store_true",
                    help="With --llm-lora-r: train the LLM LoRA only (projector frozen)")
    args = ap.parse_args()

    set_seed(args.seed)
    rows = enforce_scope(ensure_tags(load_rows(args.manifests)), args.license_scope, "train")
    LeakGuard().filter_rows(rows, on_leak="raise", context="train manifests")
    if args.dev_manifests:
        val_rows = enforce_scope(ensure_tags(load_rows(args.dev_manifests)), args.license_scope, "dev")
        LeakGuard(roles=("test",)).filter_rows(val_rows, on_leak="raise", context="dev manifests")
        train_rows = rows
    else:
        train_rows, val_rows = group_split(rows, args.val_frac, random.Random(args.seed), row_group_key)
    check_disjoint(train_rows, val_rows, context="train vs validation")
    print(f"[train] {len(train_rows)} train / {len(val_rows)} val clips; llm lora r={args.llm_lora_r}")

    Dataset = ListDataset
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Trainer, TrainingArguments

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, dtype=torch.bfloat16)
    prompt_text = processor.tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
                                                           tokenize=False, add_generation_prompt=True)
    for n, p in model.named_parameters():
        p.requires_grad = "projector" in n and not args.freeze_projector
    if args.llm_lora_r:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.llm_lora_r, lora_alpha=2 * args.llm_lora_r, lora_dropout=0.05, bias="none",
            target_modules=r".*language_model.*\.(q_proj|v_proj)",
            modules_to_save=None if args.freeze_projector else ["projector"]))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable parameters: {n_train / 1e6:.1f}M")

    ds = lambda rs: Dataset([{"audio": r["audio_path"], "target": r["transcript"]} for r in rs])
    train_ds, val_ds = ds(train_rows), ds(val_rows)

    out = Path(args.out)
    if int(os.environ.get("RANK", "0")) == 0:
        write_run_record(out, vars(args), inputs=args.manifests + args.dev_manifests,
                         extra={"train_clips": len(train_ds), "val_clips": len(val_ds), "trainable": n_train,
                                "recipe": "ibm-granite/granite-speech-models fine_tuning notebook @3306e51 (adapted)",
                                "by_use": dict(Counter(r.get("license_use", "untagged") for r in train_rows))})
    targs = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, warmup_ratio=args.warmup_ratio,
        logging_steps=25, bf16=True, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", dataloader_num_workers=4,
        remove_unused_columns=False, report_to="none", seed=args.seed, data_seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(processor, prompt_text))
    trainer.train()
    if trainer.is_world_process_zero():
        final = out / "final"
        best = trainer.model
        if args.llm_lora_r:
            best = best.merge_and_unload()
        best.save_pretrained(str(final))
        processor.save_pretrained(str(final))
        (final / "run_record.json").write_text((out / "run_record.json").read_text(encoding="utf-8"),
                                               encoding="utf-8")
        (final / "selection.json").write_text(json.dumps({"best_checkpoint": trainer.state.best_model_checkpoint,
                                                          "best_eval_loss": trainer.state.best_metric}),
                                              encoding="utf-8")
        print(f"[train] model saved -> {final} (best eval_loss {trainer.state.best_metric})")


if __name__ == "__main__":
    main()
