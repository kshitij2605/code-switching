#!/usr/bin/env python3
"""Fine-tune Qwen3-ASR (0.6B / 1.7B) on our CS corpus, under the same data
contract as train_whisper.py.

The model code path follows the official recipe, QwenLM/Qwen3-ASR
finetuning/qwen3_asr_sft.py (Apache-2.0, Alibaba Qwen team; vendored at commit
7c6daf7, adapted): full fine-tuning; text "language <Lang><asr_text><transcript>";
the chat-template prefix masked from the loss; librosa loading at 16 kHz;
CastFloatInputsTrainer.

What this script adds for our protocol:
  * the leak guard (train: test and dev roles; dev: test role), licence scope and
    disjointness checks of train_whisper.py;
  * --dev-manifests: evaluation once per epoch on an external dev set, keeping the
    lowest-eval-loss checkpoint (load_best_model_at_end), saved to OUT/final with
    every processor file, so OUT/final loads with Qwen3ASRModel.from_pretrained;
  * seeds (python, numpy, torch, Trainer seed and data_seed) and OUT/run_record.json;
  * --label-language auto|none: auto gives "language Japanese" to CS and Japanese
    rows and "language English" to English rows (as train_whisper's auto labels);
    none gives "language None", the official option that drops language
    supervision. The official script's checkpoint callback copies processor files
    only from a local model path; the model id is resolved to its local snapshot.

Run (mactrn01, qwen3-asr environment, one GPU):
  CUDA_VISIBLE_DEVICES=2 /data/code-switching/envs/qwen3-asr/bin/python 03-training/asr-ft/train_qwen3_asr.py \
     --model-id Qwen/Qwen3-ASR-0.6B --manifests ... --dev-manifests ... --out OUT --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.leak_guard import LeakGuard, check_disjoint, group_split, row_group_key  # noqa: E402
from common.licenses import enforce_scope, ensure_tags  # noqa: E402
from common.run_registry import set_seed, write_run_record  # noqa: E402

_JA = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
PROCESSOR_FILES = ["config.json", "generation_config.json", "preprocessor_config.json",
                   "processor_config.json", "tokenizer_config.json", "tokenizer.json",
                   "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json"]


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


def row_language(row: dict, mode: str) -> str:
    if mode == "none":
        return "None"
    lang = str(row.get("lang") or "").lower()
    if lang.startswith(("ja", "jp")):
        return "Japanese"
    if lang.startswith("en"):
        return "English"
    return "Japanese" if _JA.search(row["transcript"]) else "English"


def to_examples(rows: list[dict], mode: str) -> list[dict]:
    return [{"audio": r["audio_path"], "text": f"language {row_language(r, mode)}<asr_text>{r['transcript']}",
             "prompt": ""} for r in rows]


# ── official recipe (QwenLM/Qwen3-ASR finetuning/qwen3_asr_sft.py, Apache-2.0) ──

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: model has no `.thinker.forward`.")

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(input_ids=input_ids, attention_mask=attention_mask,
                                    input_features=input_features,
                                    feature_attention_mask=feature_attention_mask, labels=labels, **kwargs)

    cls.forward = forward
    cls._forward_patched = True


def build_prefix_messages(prompt: str, audio_array):
    return [{"role": "system", "content": prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_array}]}]


@dataclass
class Collator:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        import librosa
        prefix_texts = [f["prefix_text"] for f in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + f["target"] + eos for p, f in zip(prefix_texts, features)]
        audios = [librosa.load(f["audio"], sr=self.sampling_rate, mono=True)[0] for f in features]
        full = self.processor(text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)
        prefix = self.processor(text=prefix_texts, audio=audios, return_tensors="pt", padding=True,
                                truncation=False)
        # Supervise exactly the target (+ eos): the last (full - prefix) real tokens.
        # The official recipe masks labels[i, :prefix_len], which assumes right padding;
        # the Qwen3-ASR processor pads on the LEFT, so every example shorter than the
        # longest in its batch had the tail of its prompt (audio placeholders) in the loss.
        # Masking by pad id is also dropped: it would mask an eos that equals the pad token.
        labels = torch.full_like(full["input_ids"], -100)
        n_target = (full["attention_mask"].sum(dim=1) - prefix["attention_mask"].sum(dim=1)).tolist()
        for i, t in enumerate(n_target):
            pos = full["attention_mask"][i].nonzero().flatten()
            keep = pos[len(pos) - t:]
            labels[i, keep] = full["input_ids"][i, keep]
        full["labels"] = labels
        return full


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--dev-manifests", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--label-language", choices=["auto", "none"], default="auto")
    ap.add_argument("--license-scope", choices=["research", "commercial"], default="research")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=0,
                    help="0 = full fine-tuning (official recipe); R > 0 = LoRA of rank R (alpha 2R) on the "
                         "LLM's q_proj/v_proj only, everything else frozen, merged before saving")
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
    print(f"[train] {len(train_rows)} train / {len(val_rows)} val clips; labels {args.label_language}")

    Dataset = ListDataset
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig, Trainer, TrainingArguments

    model_dir = args.model_id if os.path.isdir(args.model_id) else snapshot_download(args.model_id)
    wrapper = Qwen3ASRModel.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=None)
    model, processor = wrapper.model, wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    if args.lora_r:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
            target_modules=r"thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable parameters: {n_train / 1e6:.1f}M")

    prefix_text = processor.apply_chat_template([build_prefix_messages("", None)],
                                                add_generation_prompt=True, tokenize=False)[0]

    def dataset(rs):
        ex = to_examples(rs, args.label_language)
        for e in ex:
            e["prefix_text"], e["target"] = prefix_text, e.pop("text")
        return Dataset(ex)

    train_ds, val_ds = dataset(train_rows), dataset(val_rows)
    print(f"[train] example target: {train_ds[0]['target'][:80]!r}")

    out = Path(args.out)
    if int(os.environ.get("RANK", "0")) == 0:
        write_run_record(out, vars(args), inputs=args.manifests + args.dev_manifests,
                         extra={"train_clips": len(train_ds), "val_clips": len(val_ds), "model_dir": model_dir,
                                "recipe": "QwenLM/Qwen3-ASR finetuning/qwen3_asr_sft.py @7c6daf7 (adapted)",
                                "by_use": dict(Counter(r.get("license_use", "untagged") for r in train_rows))})

    class CastFloatInputsTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=self.model.dtype)
            return inputs

    targs = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio, logging_steps=25, bf16=True,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_safetensors=True,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        dataloader_num_workers=4, remove_unused_columns=False, report_to="none",
        seed=args.seed, data_seed=args.seed, ddp_find_unused_parameters=False)
    trainer = CastFloatInputsTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                                     data_collator=Collator(processor), processing_class=processor.tokenizer)
    trainer.train()
    if trainer.is_world_process_zero():
        final = out / "final"
        if args.lora_r:
            trainer.model.merge_and_unload().save_pretrained(str(final))
        else:
            trainer.save_model(str(final))
        for fn in PROCESSOR_FILES:
            src = Path(model_dir) / fn
            if src.exists():
                shutil.copy2(src, final / fn)
        (final / "run_record.json").write_text((out / "run_record.json").read_text(encoding="utf-8"),
                                               encoding="utf-8")
        best = trainer.state.best_model_checkpoint
        (final / "selection.json").write_text(json.dumps({"best_checkpoint": best,
                                                          "best_eval_loss": trainer.state.best_metric}),
                                              encoding="utf-8")
        print(f"[train] model saved -> {final} (best {best}, eval_loss {trainer.state.best_metric})")


if __name__ == "__main__":
    main()
