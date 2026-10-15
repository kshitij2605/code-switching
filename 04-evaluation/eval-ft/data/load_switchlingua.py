#!/usr/bin/env python3
"""Load the Japanese part of SwitchLingua (HF: Shelton1013/SwitchLingua_audio; NeurIPS
2025) as a real JA-EN code-switching TEST set (Phase 1, T-MULTI).

Licence: CC BY-NC 4.0, plus the authors' licensing agreement on the dataset card, which
binds by download: research and non-commercial use only, no redistribution,
attribution. Use here is evaluation only, in research and papers. It never trains a
model and never feeds the commercial product decision (bake-off). The set is
registered as protected test data in common/eval_blocklist.json.

Contents: Japanese.csv (1,335 rows) plus Japanese/<session>_<k>.m4a. The text was
generated with the authors' LinguaMaster framework and read by bilingual speakers;
the CSV carries speaker metadata (gender, age, education, L1/L2) and labels
(cs_type, cs_function, cs_ratio, topic). One session (the file-name prefix) shares
one speaker's metadata; it is the clustering unit (`speaker`) for speaker-level
bootstrap. Audio is converted with ffmpeg to 16 kHz mono WAV.

Example:
  $PY data/load_switchlingua.py --out /data/code-switching/data/eval-real/switchlingua
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO = "Shelton1013/SwitchLingua_audio"
META = ("gender", "age", "education Level", "first Language", "second Language",
        "cs_type", "cs_function", "cs_ratio", "topic", "conversation_type")


def fetch_throttled(files: list[str], min_interval: float) -> None:
    """Download files one by one: the Hub allows 1,000 API calls per 5 minutes, and
    each file costs about two. Cached files are skipped; 429s back off and retry."""
    import time
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    for i, f in enumerate(files):
        try:
            hf_hub_download(REPO, f, repo_type="dataset", local_files_only=True)
            continue
        except Exception:
            pass
        for attempt in range(12):
            try:
                t0 = time.time()
                hf_hub_download(REPO, f, repo_type="dataset")
                time.sleep(max(0.0, min_interval - (time.time() - t0)))
                break
            except HfHubHTTPError as e:
                if "429" not in str(e) or attempt == 11:
                    raise
                print(f"  rate-limited at file {i}; waiting 60 s", flush=True)
                time.sleep(60)
        if (i + 1) % 100 == 0:
            print(f"  fetched {i + 1}/{len(files)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output dir (manifest.jsonl + audio/)")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--min-interval", type=float, default=0.7, help="Seconds per file download (Hub quota)")
    args = ap.parse_args(argv)

    import soundfile as sf
    from huggingface_hub import snapshot_download
    from common.licenses import tag_rows

    snap = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=["Japanese.csv", "README.md"]))
    recs = list(csv.DictReader(open(snap / "Japanese.csv", encoding="utf-8-sig")))
    fetch_throttled([f"Japanese/{r['file_name']}" for r in recs], args.min_interval)
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows, total, missing = [], 0.0, []
    for r in recs:
        src = snap / "Japanese" / r["file_name"]
        if not src.exists():
            missing.append(r["file_name"])
            continue
        stem = Path(r["file_name"]).stem
        wav = out / "audio" / f"{stem}.wav"
        if not wav.exists():
            subprocess.run([args.ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        info = sf.info(str(wav))
        dur = info.frames / info.samplerate
        total += dur
        session = stem.split("_")[0]
        rows.append({"id": f"switchlingua_ja_{stem}", "reference": r["text"].strip(), "hypothesis": "",
                     "audio": f"audio/{stem}.wav", "lang": "jpn-eng", "duration_sec": round(dur, 3),
                     "speaker": f"sl:{session}", "source": "switchlingua/ja",
                     "group": f"switchlingua:{session}", "audio_path": str(wav.resolve()),
                     "transcript": r["text"].strip(), **{f"meta_{k.split()[0].lower()}": r[k] for k in META}})
    if missing:
        raise SystemExit(f"{len(missing)} audio files missing from the snapshot, e.g. {missing[:3]}")
    rows = tag_rows(rows, REPO)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[switchlingua/ja] {len(rows)} utterances, {total / 3600:.2f} h, "
          f"{len({r['speaker'] for r in rows})} sessions -> {out / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
