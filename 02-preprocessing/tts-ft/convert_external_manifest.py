#!/usr/bin/env python
"""Convert an external CS/JA/EN dataset collection (data/) into our manifest schema, so mix_manifests.py can
combine them with our Kokoro CS + FLEURS mono.

Their metadata (transcriptions.json) per bucket:
  code_switching: {id, audio_file, full_transcription, segments:[{lang,text,...}], duration_seconds}
  japanese_only : {id, audio_file, japanese_transcription, english_translation, duration_seconds}
  english_only  : {id, audio_file, english_transcription, japanese_translation, duration_seconds}

  python scripts/convert_external_manifest.py --root /data/lfm2.5/run-a --out-dir data/external
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "05-app-and-demo" / "kaigi-app"  # stage-folder copy of kaigi-app/scripts
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.manifest import write_jsonl  # noqa: E402

# bucket -> (metadata path rel to --root, transcript field, style, source, single-span lang or None=use segments)
# NB: metadata lives under <root>/data/<bucket>/metadata; audio_file inside is also "data/<bucket>/audio/..".
_BUCKETS = {
    "cs": ("data/code_switching/metadata/transcriptions.json", "full_transcription", "cs", "edge_cs", None),
    "ja": ("data/japanese_only/metadata/transcriptions.json", "japanese_transcription", "mono_ja", "edge_ja", "ja"),
    "en": ("data/english_only/metadata/transcriptions.json", "english_transcription", "mono_en", "edge_en", "en"),
}


def _spans(ex: dict, text: str, lang: str | None) -> list[dict]:
    if lang is not None:
        return [{"text": text, "lang": lang}]
    out = []
    for seg in ex.get("segments", []):
        t = (seg.get("text") or "").strip()
        if t and seg.get("lang") in ("ja", "en"):
            out.append({"text": t, "lang": seg["lang"]})
    return out or [{"text": text, "lang": "ja"}]


def convert(bucket: str, root: Path, out_dir: Path) -> int:
    meta_rel, text_field, style, source, lang = _BUCKETS[bucket]
    meta_path = root / meta_rel
    if not meta_path.exists():
        print(f"[{bucket}] SKIP — no {meta_path}")
        return 0
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = []
    for ex in data:
        text = (ex.get(text_field) or "").strip()
        audio = root / ex["audio_file"]
        if not text or not audio.exists():
            continue
        rows.append({
            "id": ex.get("id", f"{source}_{len(rows):05d}"),
            "audio_path": str(audio.resolve()),
            "transcript": text,
            "spans": _spans(ex, text, lang),
            "style": style,
            "voice": "edge",
            "domain": ex.get("topic", "meeting" if bucket == "cs" else "general"),
            "source": source,
            "duration_s": round(float(ex.get("duration_seconds", 0.0)), 3),
            "sr": 24000,
        })
    out_path = out_dir / f"external_{bucket}.jsonl"
    n = write_jsonl(out_path, rows)
    print(f"[{bucket}] {n} rows -> {out_path}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/data/lfm2.5/run-a", help="teammate data root (parent of data/)")
    ap.add_argument("--out-dir", default="data/external")
    ap.add_argument("--buckets", nargs="*", default=["cs", "ja", "en"], choices=["cs", "ja", "en"])
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(convert(b, root, out_dir) for b in args.buckets)
    print(f"\nconverted {total} rows -> {out_dir}  (feed these to scripts/mix_manifests.py)")


if __name__ == "__main__":
    main()
