#!/usr/bin/env python
"""Measure base JA->EN translation quality to decide if the MT stage needs fine-tuning.

  pip install -e ".[mt]"
  python scripts/eval_translation.py --n 200                 # base LFM2.5-1.2B-JP on FLEURS ja->en
  python scripts/eval_translation.py --pairs data/mt/indomain.jsonl   # your own {ja,en} set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.mt.eval import evaluate, load_fleurs_pairs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--translator", default="LiquidAI/LFM2.5-1.2B-JP")
    ap.add_argument("--adapter", help="LoRA adapter to evaluate the fine-tuned MT against the base")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--pairs", help="jsonl with {ja, en} rows for an in-domain eval instead of FLEURS")
    ap.add_argument("--mock", action="store_true", help="identity translator (wiring sanity only)")
    args = ap.parse_args()

    if args.pairs:
        from csmeeting.manifest import read_jsonl
        pairs = [(r["ja"], r["en"]) for r in read_jsonl(args.pairs)]
        if args.n:
            pairs = pairs[: args.n]
    else:
        print("Loading FLEURS ja_jp / en_us pairs (joined by FLoRes id)...")
        pairs = load_fleurs_pairs(args.n)
    print(f"{len(pairs)} parallel pairs")

    if args.mock:
        translate = lambda s: s  # noqa: E731
    else:
        from csmeeting.live.config import LiveConfig
        from csmeeting.live.translate import LFMTranslator
        cfg = LiveConfig(device=args.device, translator_model_id=args.translator, translator_adapter=args.adapter)
        translate = LFMTranslator(cfg).translate

    res = evaluate(translate, pairs)
    print(json.dumps({k: res[k] for k in ("n", "bleu", "chrf")}, indent=2))
    print("\nsamples:")
    for s in res["samples"]:
        print(f"  JA : {s['ja']}\n  REF: {s['ref']}\n  HYP: {s['hyp']}\n")
    print("Decision: if chrF is healthy AND samples read well -> ship the base translator; "
          "else fine-tune the MT stage (scripts/prep_mt_data.py).")


if __name__ == "__main__":
    main()
