#!/usr/bin/env python
"""Synthesize audio for transcripts.jsonl -> wav files + manifest.jsonl (script-matched bilingual voices)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.synthesize import synthesize  # noqa: E402


def load_cfg(path: str) -> dict:
    try:
        import yaml
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else ROOT / pp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "pipeline.yaml"))
    ap.add_argument("--in", dest="inp", help="transcripts.jsonl (default from config)")
    ap.add_argument("--out-dir", help="output dir; writes audio/ + manifest.jsonl (default = transcripts' dir)")
    ap.add_argument("--backend", choices=["edge", "kokoro"], help="TTS backend (default from config or 'edge')")
    ap.add_argument("--limit", type=int, default=0, help="only synthesize first N (0 = all)")
    ap.add_argument("--placeholder", action="store_true", help="write SILENT wavs (offline smoke, no TTS/network)")
    ap.add_argument("--sr", type=int, help="target sample rate")
    ap.add_argument("--gap-ms", type=int, help="silence between spliced spans")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths, tts = cfg.get("paths", {}), cfg.get("tts", {})
    inp = resolve(args.inp or paths.get("transcripts", "data/synth/transcripts.jsonl"))
    out_dir = resolve(args.out_dir) if args.out_dir else inp.parent

    backend = args.backend or tts.get("backend", "edge")
    # speakers are backend-specific (edge voice ids != Kokoro voice ids). Prefer tts.speakers_<backend>;
    # fall back to tts.speakers only for edge; otherwise None -> the backend's built-in defaults.
    speakers = tts.get(f"speakers_{backend}") or (tts.get("speakers") if backend == "edge" else None)

    synthesize(
        transcripts_path=inp,
        out_dir=out_dir,
        backend=backend,
        speakers=speakers,
        target_sr=args.sr or tts.get("sample_rate", 24000),
        gap_ms=args.gap_ms if args.gap_ms is not None else tts.get("gap_ms", 60),
        limit=args.limit,
        placeholder=args.placeholder,
    )


if __name__ == "__main__":
    main()
