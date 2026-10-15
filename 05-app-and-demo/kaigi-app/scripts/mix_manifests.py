#!/usr/bin/env python
"""Combine several manifests (CS synth + mono JA + EN replay) into one train/eval split for fine-tuning.

Rewrites each row's audio_path to an ABSOLUTE path (rows come from different dirs), shuffles deterministically,
holds out an eval fraction (stratified by source so every bucket appears in both splits), and writes train.jsonl +
eval.jsonl. The data-plan target mix is ~30% synthetic CS / ~55% mono JA / ~15% EN replay — control it with --cap.

  python scripts/mix_manifests.py \
      --manifest data/cs/manifest.jsonl --manifest data/mono/manifest_ja.jsonl --manifest data/mono/manifest_en.jsonl \
      --out-dir data/cs_mixed --eval-frac 0.05
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.manifest import read_jsonl, write_jsonl  # noqa: E402


def _abspath(audio_path: str, manifest: Path) -> str:
    p = Path(audio_path)
    return str((p if p.is_absolute() else manifest.parent / p).resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True, help="repeatable: each input manifest.jsonl")
    ap.add_argument("--out-dir", default="data/cs_mixed")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--cap", type=int, default=0, help="max rows per input manifest (0 = all)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for m in args.manifest:
        mp = Path(m) if Path(m).is_absolute() else ROOT / m
        these = list(read_jsonl(mp))
        rng.shuffle(these)
        if args.cap > 0:
            these = these[:args.cap]
        for r in these:
            r["audio_path"] = _abspath(r["audio_path"], mp)
        rows.extend(these)
        print(f"  + {len(these):5d} rows from {mp}")

    # stratified train/eval split by source
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r.get("source", "unknown")].append(r)
    train, eval_ = [], []
    for src, group in by_src.items():
        rng.shuffle(group)
        k = max(1, int(len(group) * args.eval_frac)) if len(group) > 1 else 0
        eval_.extend(group[:k])
        train.extend(group[k:])
    rng.shuffle(train); rng.shuffle(eval_)

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    n_tr = write_jsonl(out_dir / "train.jsonl", train)
    n_ev = write_jsonl(out_dir / "eval.jsonl", eval_)
    print(f"\ntrain={n_tr}  eval={n_ev}  -> {out_dir}")
    print("train by source:", dict(Counter(r.get("source") for r in train)))
    print("eval  by source:", dict(Counter(r.get("source") for r in eval_)))


if __name__ == "__main__":
    main()
