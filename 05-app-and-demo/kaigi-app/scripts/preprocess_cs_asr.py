#!/usr/bin/env python
"""Preprocess a code-switch ASR manifest into liquid-audio's on-disk training format.

Pipeline: manifest.jsonl --(CodeSwitchASRIterator: system=ASR prompt / user=audio / assistant=transcript)-->
          liquid_audio.data.preprocess.preprocess_dataset(mapper=LFM2AudioChatMapper(processor)) --> LFM2DataLoader.

Each preprocessed dataset is processor-specific, so run this once per base model (the JP base and the EN base may
tokenize differently). Mirrors the host's optionB_preprocess.py but for ASR (audio in / text out).

  python scripts/preprocess_cs_asr.py \
      --model-id LiquidAI/LFM2.5-Audio-1.5B-JP \
      --train-manifest data/cs/train.jsonl --eval-manifest data/cs/eval.jsonl \
      --output-dir data/cs/preprocessed_jp --system-prompt "Perform ASR in japanese."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.cs_asr_iterator import CodeSwitchASRIterator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="LiquidAI/LFM2.5-Audio-1.5B-JP",
                    help="base model whose processor/tokenizer encodes the data (JP base or EN base)")
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--eval-manifest", help="held-out manifest (optional but recommended for val_loss)")
    ap.add_argument("--output-dir", required=True, help="writes <dir>/train and <dir>/eval")
    ap.add_argument("--system-prompt", default=None,
                    help='force one ASR prompt for all rows; default = per-row by language '
                         '(JA/CS -> "Perform ASR in japanese.", EN replay -> "Perform ASR.")')
    ap.add_argument("--max-context", type=int, default=256,
                    help="tokens; samples longer than this are DROPPED (kit default 256). Raise if many are skipped.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true", help="re-preprocess even if the output split exists")
    args = ap.parse_args()

    from liquid_audio import LFM2AudioProcessor
    from liquid_audio.data.mapper import LFM2AudioChatMapper
    from liquid_audio.data.preprocess import preprocess_dataset

    print(f"Loading processor: {args.model_id} (device={args.device})")
    processor = LFM2AudioProcessor.from_pretrained(args.model_id, device=args.device).eval()
    mapper = LFM2AudioChatMapper(processor)
    out_root = Path(args.output_dir)

    splits = [("train", args.train_manifest)]
    if args.eval_manifest:
        splits.append(("eval", args.eval_manifest))

    for split, manifest in splits:
        out_path = out_root / split
        if out_path.exists() and not args.overwrite:
            print(f"[{split}] exists at {out_path}, skipping (use --overwrite to redo).")
            continue
        sys_desc = repr(args.system_prompt) if args.system_prompt else "per-row by language"
        print(f"\n[{split}] preprocessing {manifest} -> {out_path}  (system={sys_desc})")
        data = CodeSwitchASRIterator(manifest, system_prompt=args.system_prompt)
        preprocess_dataset(data=data, output_path=out_path, mapper=mapper, max_context_length=args.max_context)
        print(f"[{split}] done.")

    print(f"\nPreprocessed dataset at: {out_root}")
    print("Next: scripts/train_cs_asr.py --data", out_root)


if __name__ == "__main__":
    main()
