#!/usr/bin/env python
"""Reduce code-switch density to match real speech (JECS averages ~1 English insertion/utterance).

Our synthetic CS over-represents switches (multiple English spans/utt), which biased FPFT to hallucinate English on
real audio (JECS WER >1.0). This keeps all mono rows and only the CS rows with <= --max-switches English spans.

  python scripts/filter_lowdensity.py --in data/cs_mixed/train.jsonl --out data/cs_mixed/train_lowdens.jsonl --max-switches 1
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.manifest import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-switches", type=int, default=1, help="max English spans allowed in a CS utterance")
    args = ap.parse_args()

    inp = Path(args.inp) if Path(args.inp).is_absolute() else ROOT / args.inp
    kept, dropped = [], 0
    for row in read_jsonl(inp):
        style = str(row.get("style", ""))
        if style.startswith("mono"):
            kept.append(row)
            continue
        en = sum(1 for s in row.get("spans", []) if s.get("lang") == "en")
        if en <= args.max_switches:
            kept.append(row)
        else:
            dropped += 1

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    n = write_jsonl(out, kept)
    print(f"kept {n} (dropped {dropped} high-density CS) -> {out}")
    print("by source:", dict(Counter(r.get("source") for r in kept)))


if __name__ == "__main__":
    main()
