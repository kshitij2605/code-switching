#!/usr/bin/env python
"""Build commercial-safe JA->EN SFT data (FLEURS + Tatoeba + JESC) for the OPTIONAL MT fine-tune.

Only run this if scripts/eval_translation.py shows the base translator underperforms on your domain.
Output is the kit's text-track `messages` format.

  pip install -e ".[mt]"
  python scripts/prep_mt_data.py --out data/mt/train.jsonl --jesc 20000 --tatoeba 20000 --fleurs 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.mt.prep_data import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/mt/train.jsonl")
    ap.add_argument("--jesc", type=int, default=20000)
    ap.add_argument("--tatoeba", type=int, default=20000)
    ap.add_argument("--fleurs", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    n = build(out, jesc=args.jesc, tatoeba=args.tatoeba, fleurs=args.fleurs)
    print(f"wrote {n} JA->EN SFT pairs -> {args.out}")
    print("Fine-tune via the kit's text track (TRL+LoRA) or the cookbook cpt_translation_with_unsloth.ipynb. "
          "See docs/translation.md.")


if __name__ == "__main__":
    main()
