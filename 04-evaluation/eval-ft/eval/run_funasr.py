#!/usr/bin/env python3
"""Run Fun-ASR (FunAudioLLM/Fun-ASR-Nano-2512, Apache-2.0) over an eval manifest.

Phase 1 leaderboard baseline. Our landscape report rates it strong in Japanese,
and its card claims "free language switching and mixed recognition". It needs
the `funasr` package and the model's own code (trust_remote_code, from the
official FunAudioLLM Hugging Face repo), so it runs in its own environment:
    /data/code-switching/envs/funasr   (torch 2.11.0+cu126 + Fun-ASR requirements;
                            frozen in envs/funasr.freeze.txt)

--language is passed to generate() as given; the model uses Chinese names, and
the default example is "中文". Omit it for the model's automatic behaviour.
itn=True, the card's default, writes numbers as digits, like our references.
Output JSONL matches the shared contract ({id, reference, hypothesis}).

Example:
  CUDA_VISIBLE_DEVICES=0 /data/code-switching/envs/funasr/bin/python eval/run_funasr.py \
      --manifest M.jsonl --audio-root ROOT --out preds.jsonl
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
    ap.add_argument("--model-id", default="FunAudioLLM/Fun-ASR-Nano-2512")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default=None)
    ap.add_argument("--no-itn", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from funasr import AutoModel

    rows = read_manifest(args.manifest, args.audio_root, args.limit)
    if not rows:
        raise SystemExit("Empty manifest — nothing to run.")
    print(f"[fun-asr] {args.model_id}, language={args.language}, {len(rows)} utterances")
    model = AutoModel(model=args.model_id, hub="hf", trust_remote_code=True, device="cuda:0",
                      disable_update=True)
    kw = {"itn": not args.no_itn}
    if args.language:
        kw["language"] = args.language

    t0, out = time.time(), []
    for b in range(0, len(rows), args.batch_size):
        batch = rows[b:b + args.batch_size]
        res = model.generate(input=[r["_audio_path"] for r in batch], cache={},
                             batch_size=len(batch), **kw)
        if len(res) != len(batch):
            raise SystemExit(f"generate returned {len(res)} results for {len(batch)} inputs")
        for r, x in zip(batch, res):
            out.append({"id": r["id"], "reference": r["reference"],
                        "hypothesis": (x.get("text") or "").strip()})
        print(f"  [{min(b + args.batch_size, len(rows))}/{len(rows)}] {out[-len(batch)]['hypothesis'][:80]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"\nWrote {len(out)} predictions -> {args.out}  ({dt:.1f}s, {dt / len(out):.2f}s/utt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
