#!/usr/bin/env python3
"""Synthetic twins of a real test set (Phase 1, T-TWINS): the *same* text,
rendered by TTS engines, so that synthetic-vs-real comparisons isolate the
acoustics (P-1b ranking fidelity). The legacy synthetic sets use different text.

  prepare   split every reference into Japanese / English spans, in the format
            of the engine renderers (01-data-generation/asr-ft/engines/render_*.py):
            {id, transcript, segments: [{lang, text}], en_words, ja_text}
  finalize  turn a renderer manifest back into an eval-contract manifest
            {id, reference, audio, lang, engine, twin_of, source}

Twins are TEST data: their text is the protected test text, so the leak guard
already keeps them out of training.

Span rule: Latin words are English. Digits and number punctuation belong to
English only between two English words ("magnitude 6.5 earthquake"); otherwise
they are read in Japanese ("2種類"). Spaces and ASCII punctuation attach to the
current span.

Usage:
  python data/make_twins.py prepare --manifest READ/manifest.jsonl --out twins_in.jsonl
  <python> ../../01-data-generation/asr-ft/engines/render_kokoro.py --in twins_in.jsonl --out DIR
  python data/make_twins.py finalize --rendered DIR/manifest.jsonl --engine kokoro --out DIR/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.cs_segments import segments  # noqa: E402  (shared span splitter)



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--rendered", required=True)
    f.add_argument("--engine", required=True)
    f.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "prepare":
        rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
        with open(args.out, "w", encoding="utf-8") as fo:
            for r in rows:
                segs = segments(r["reference"])
                fo.write(json.dumps({
                    "id": r["id"], "transcript": r["reference"], "segments": segs,
                    "en_words": [w for s in segs if s["lang"] == "en" for w in s["text"].split()],
                    "ja_text": "".join(s["text"] for s in segs if s["lang"] == "ja"),
                    "n_switches": max(0, len(segs) - 1)}, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} render inputs -> {args.out}")
        return 0

    rows = [json.loads(l) for l in open(args.rendered, encoding="utf-8") if l.strip()]
    with open(args.out, "w", encoding="utf-8") as fo:
        for r in rows:
            twin_of = re.sub(rf"^{re.escape(args.engine)}_", "", r["id"])
            fo.write(json.dumps({
                "id": twin_of, "reference": r["transcript"], "hypothesis": "",
                "audio": r["audio_path"], "lang": "jpn-eng", "engine": args.engine,
                "twin_of": twin_of, "speaker": r.get("speaker") or r.get("voice"),
                "source": f"twins/{args.engine}"}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} eval rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
