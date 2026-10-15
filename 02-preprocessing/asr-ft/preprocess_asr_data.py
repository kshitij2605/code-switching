"""
Converts our ASR JSONL dataset into liquid-audio's preprocessed format.
Reads data/training/train.jsonl and data/training/eval.jsonl.
Outputs preprocessed HuggingFace datasets to data/preprocessed/{train,eval}.

This is the last step before LFM training, so it enforces these checks:
  * train rows must not overlap any protected test or dev data; eval rows (used
    for checkpoint selection) must not overlap test data (common/leak_guard.py);
  * train and eval must not share a group or an audio file;
  * rows whose audio is missing are an error (--allow-missing-audio to skip them);
  * an existing output split is an error (--overwrite rebuilds it);
  * --license-scope commercial refuses rows that are not commercial-clean;
  * <output_dir>/<split>.build.json records the input JSONL hashes.

Usage:
  python preprocess_asr_data.py --hf_token hf_xxx [--model_id LiquidAI/LFM2.5-Audio-1.5B-JP]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

from huggingface_hub import login

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.leak_guard import LeakGuard, check_disjoint  # noqa: E402
from common.licenses import enforce_scope, ensure_tags  # noqa: E402
from common.run_registry import write_run_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_id",    default="LiquidAI/LFM2.5-Audio-1.5B-JP")
    p.add_argument("--train_jsonl", default="data/training/train.jsonl")
    p.add_argument("--eval_jsonl",  default="data/training/eval.jsonl")
    p.add_argument("--output_dir",  default="data/preprocessed")
    p.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--max_context", type=int, default=512)
    p.add_argument("--license_scope", choices=["research", "commercial"], default="research")
    p.add_argument("--allow_missing_audio", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Rebuild existing output splits")
    return p.parse_args()


def read_rows(jsonl_path: str, allow_missing: bool) -> list[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    missing = [r["audio_file"] for r in rows if not Path(r["audio_file"]).exists()]
    if missing:
        msg = f"{jsonl_path}: {len(missing)} of {len(rows)} audio files missing (e.g. {missing[:3]})"
        if not allow_missing:
            sys.exit(msg + "; fix the paths or pass --allow_missing_audio")
        print(f"  WARNING: {msg} -> skipped")
        rows = [r for r in rows if Path(r["audio_file"]).exists()]
    print(f"  Loaded {len(rows)} examples from {jsonl_path}")
    return rows


class ASRIterator:
    """Yields list[ChatMessage] from our JSONL rows."""

    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    def __iter__(self) -> Iterator[list]:
        from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment
        for ex in self.examples:
            audio_bytes = Path(ex["audio_file"]).read_bytes()
            yield [
                ChatMessage(
                    role="system",
                    content=[TextSegment(text=ex["system"])],
                ),
                ChatMessage(
                    role="user",
                    content=[AudioSegment(audio=audio_bytes)],
                ),
                ChatMessage(
                    role="assistant",
                    content=[TextSegment(text=ex["target"])],
                ),
            ]


def main() -> None:
    args = parse_args()

    # ── checks first: they are cheap and must pass before any GPU work ─────────
    splits = {"train": read_rows(args.train_jsonl, args.allow_missing_audio),
              "eval": read_rows(args.eval_jsonl, args.allow_missing_audio)}
    for split, rows in splits.items():
        splits[split] = enforce_scope(ensure_tags(rows), args.license_scope, context=split)
    LeakGuard().filter_rows(splits["train"], on_leak="raise", context="train")
    LeakGuard(roles=("test",)).filter_rows(splits["eval"], on_leak="raise", context="eval")
    check_disjoint(splits["train"], splits["eval"], context="train vs eval")

    out_root = Path(args.output_dir)
    todo = []
    for split in splits:
        out_path = out_root / split
        if out_path.exists():
            if not args.overwrite:
                sys.exit(f"[{split}] {out_path} already exists; pass --overwrite to rebuild it "
                         f"(it may have been built from different JSONL)")
            shutil.rmtree(out_path)
        todo.append(split)

    if args.hf_token:
        login(token=args.hf_token)
    from liquid_audio import LFM2AudioProcessor
    from liquid_audio.data.mapper import LFM2AudioChatMapper
    from liquid_audio.data.preprocess import preprocess_dataset

    print(f"Loading processor: {args.model_id}")
    processor = LFM2AudioProcessor.from_pretrained(args.model_id, device="cuda").eval()
    mapper = LFM2AudioChatMapper(processor)

    for split in todo:
        out_path = out_root / split
        jsonl = args.train_jsonl if split == "train" else args.eval_jsonl
        print(f"\n[{split}] Preprocessing {jsonl} → {out_path}")
        preprocess_dataset(
            data=ASRIterator(splits[split]),
            output_path=out_path,
            mapper=mapper,
            max_context_length=args.max_context,
        )
        # next to (not inside) the dataset directory, so the loader never sees it
        write_run_record(out_root, vars(args), inputs=[jsonl],
                         extra={"split": split, "rows": len(splits[split])},
                         filename=f"{split}.build.json")
        print(f"[{split}] Done.")

    print("\nPreprocessing complete. Dataset ready at:", args.output_dir)


if __name__ == "__main__":
    main()
