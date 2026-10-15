#!/usr/bin/env python3
"""Backfill style/density/span_type into already-rendered manifests.

The renderers were fixed to carry these fields, but the v2 corpus was rendered
before that. Rather than re-render, join each manifest row to the transcript set
by transcript text (unique) and add the three fields in place.
"""
import json
import sys
from pathlib import Path

TR = sys.argv[1] if len(sys.argv) > 1 else "data/transcripts/cs_transcripts.jsonl"
MANIFESTS = sys.argv[2:] or [
    "data/synth/edge_tts/manifest.jsonl",
    "data/synth/kokoro/manifest.jsonl",
    "data/synth/melo/manifest.jsonl",
    "data/augmented/manifest.jsonl",
]

meta = {}
for l in Path(TR).read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l)
        meta[r["transcript"]] = (r["style"], r["density"], r["span_type"])

for man in MANIFESTS:
    p = Path(man)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    patched = miss = 0
    for r in rows:
        m = meta.get(r.get("transcript"))
        if m is None:
            miss += 1
            continue
        if r.get("style") != m[0] or "density" not in r:
            r["style"], r["density"], r["span_type"] = m
            patched += 1
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{man}: {len(rows)} rows, patched={patched}, unmatched={miss}")
