#!/usr/bin/env python
"""Convert JECS (real JP↔EN code-switch speech, NAIST/Takamichi) into our manifest schema for honest real-audio eval.

JECS layout: <root>/<emotion>/transcripts_cs.txt  ("<id>: <text>", English switch wrapped in *asterisks*) +
<root>/<emotion>/wav24kHz16bit/<id>.wav. The *...* markers give exact switch spans (great for PIER).

LICENSE: text CC BY 3.0; **audio = academic / non-commercial / personal research only, no redistribution** →
EVAL ONLY. Never train on or ship this. (JECS is evaluation-only gold data.)

  python scripts/jecs_to_manifest.py --jecs-root /data/lfm2.5/run-c/data/jecs/jecs_ver1 \
      --emotion neutral --out data/eval/jecs_neutral_cs.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.manifest import write_jsonl  # noqa: E402

_SWITCH = re.compile(r"\*(.+?)\*")   # *english* switch markers


def _parse(text: str) -> tuple[str, list[dict]]:
    """'Ｊ*first goal*を決めた' -> ('Ｊfirst goalを決めた', [ja 'Ｊ', en 'first goal', ja 'を決めた'])."""
    parts = _SWITCH.split(text)        # even idx = JA (matrix), odd idx = EN (switch)
    spans, clean = [], ""
    for i, p in enumerate(parts):
        if not p:
            continue
        lang = "en" if i % 2 == 1 else "ja"
        spans.append({"text": p, "lang": lang})
        clean += p
    return clean, spans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jecs-root", default="/data/lfm2.5/run-c/data/jecs/jecs_ver1")
    ap.add_argument("--emotion", default="neutral", help="neutral | joy | sad | anger")
    ap.add_argument("--out", default="data/eval/jecs_neutral_cs.jsonl")
    args = ap.parse_args()

    root = Path(args.jecs_root) / args.emotion
    cs_txt = root / "transcripts_cs.txt"
    wav_dir = root / "wav24kHz16bit"
    rows, missing = [], 0
    for line in cs_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        uid, text = line.split(":", 1)
        uid, text = uid.strip(), text.strip()
        wav = wav_dir / f"{uid}.wav"
        if not wav.exists():
            missing += 1
            continue
        clean, spans = _parse(text)
        rows.append({
            "id": uid,
            "audio_path": str(wav.resolve()),
            "transcript": clean,
            "spans": spans,
            "style": "cs",
            "voice": "real_jecs",
            "domain": "general",
            "source": f"jecs_{args.emotion}",
            "duration_s": 0.0,
            "sr": 24000,
        })
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    n = write_jsonl(out, rows)
    print(f"JECS {args.emotion}: {n} CS utts -> {out}  ({missing} missing wavs)")
    if rows:
        print("sample:", rows[0]["transcript"], "|", [(s['lang'], s['text']) for s in rows[0]['spans']])


if __name__ == "__main__":
    main()
