#!/usr/bin/env python3
"""Run Qwen3-ASR (Qwen/Qwen3-ASR-1.7B or -0.6B; Apache-2.0) over an eval manifest.

Phase 1 leaderboard baseline and candidate backbone (backbone
bake-off). Qwen3-ASR needs the `qwen-asr` package, whose pins differ from the
research lock, so it runs in its own environment:
    /data/code-switching/envs/qwen3-asr   (torch 2.11.0+cu126, qwen-asr;
                               frozen in /data/code-switching/envs/qwen3-asr.freeze.txt)

Decoding follows the model's own transcribe() (greedy). --language None =
automatic language identification; "Japanese" forces Japanese. Output JSONL
matches the shared contract ({id, reference, hypothesis}) plus the detected
language, so predictions drop straight into eval/score.py.

Example:
  CUDA_VISIBLE_DEVICES=0 /data/code-switching/envs/qwen3-asr/bin/python eval/run_qwen3_asr.py \
      --model-id Qwen/Qwen3-ASR-1.7B --manifest M.jsonl --audio-root ROOT --out preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline import read_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default=None, help="None = auto language ID; e.g. Japanese")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    import torch
    from qwen_asr import Qwen3ASRModel

    rows = read_manifest(args.manifest, args.audio_root, args.limit)
    if not rows:
        raise SystemExit("Empty manifest — nothing to run.")
    print(f"[qwen3-asr] {args.model_id}, language={args.language}, {len(rows)} utterances")
    model = Qwen3ASRModel.from_pretrained(args.model_id, dtype=torch.bfloat16,
                                          device_map="cuda:0", max_new_tokens=args.max_new_tokens)

    t0, out = time.time(), []
    for b in range(0, len(rows), args.batch_size):
        batch = rows[b:b + args.batch_size]
        audio = [r["_audio_path"] for r in batch]   # paths: qwen-asr decodes and resamples
        res = model.transcribe(audio=audio, language=args.language)
        if len(res) != len(batch):
            raise SystemExit(f"transcribe returned {len(res)} results for {len(batch)} inputs")
        for r, x in zip(batch, res):
            out.append({"id": r["id"], "reference": r["reference"],
                        "hypothesis": (x.text or "").strip(), "detected_language": x.language})
        print(f"  [{min(b + args.batch_size, len(rows))}/{len(rows)}] {batch[0]['id']}: "
              f"{out[-len(batch)]['hypothesis'][:80]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"\nWrote {len(out)} predictions -> {args.out}  ({dt:.1f}s, {dt / len(out):.2f}s/utt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
