#!/usr/bin/env python3
"""Convert training-contract manifests ({id, audio_path, transcript, ...}) into
the LFM chat JSONL that preprocess_asr_data.py reads
({audio_file, system, target, type, group, licence fields}).

Whisper and LFM runs can then train on the *same* rows. Fields the guards
need are kept: id, group, augmentation and licence tags (legacy rows are tagged
from their engine / type). Transcripts are not changed.

Usage:
  python manifest_to_lfm_jsonl.py --out train_lfm.jsonl \
      data/synth/edge_tts/manifest.jsonl data/augmented_v2/manifest.jsonl data/mono/mono_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.licenses import ensure_tags  # noqa: E402

KEEP = ("id", "group", "augmentation", "clean_id", "engine", "lang",
        "license_source", "license", "license_use")


def convert(row: dict, system: str, manifest: str) -> dict:
    out = {"audio_file": row["audio_path"], "system": system, "target": row["transcript"],
           "type": row.get("type") or Path(manifest).parent.name}
    out.update({k: row[k] for k in KEEP if row.get(k) not in (None, "")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", default="Perform ASR.",
                    help="System prompt (the standard eval prompt; keep train and eval identical)")
    args = ap.parse_args()
    rows = []
    for m in args.manifests:
        for line in Path(m).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(convert(json.loads(line), args.system, m))
    rows = ensure_tags(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}; by type {dict(Counter(r['type'] for r in rows))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
