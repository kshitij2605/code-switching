#!/usr/bin/env python
"""Download a small monolingual replay slice (JA + EN) -> wav + manifest rows, for the anti-forgetting mix.

Default source = FLEURS (`google/fleurs`, CC-BY-4.0, parquet → works with current `datasets`).
The CC0 Common Voice mirror is script-based and rejected by recent `datasets` ("dataset scripts no longer
supported"), so it's left as a non-default option. Streaming, so it only pulls the N rows you ask for. Output schema
matches synthesize.py's manifest (consumed by cs_asr_iterator + mix_manifests).

  python scripts/download_mono.py --ja 1500 --en 500 --out-dir data/mono            # FLEURS
  python scripts/download_mono.py --source cv --ja 1500 --en 500                     # Common Voice (needs script support)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "05-app-and-demo" / "kaigi-app"  # stage-folder copy of kaigi-app/scripts
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.manifest import append_jsonl  # noqa: E402
from csmeeting.tts_backends import resample_linear, save_wav  # noqa: E402

# dataset source -> repo, per-lang config name, and the text fields to try (in order)
_SOURCES = {
    "fleurs": {"repo": "google/fleurs", "configs": {"ja": "ja_jp", "en": "en_us"},
               "text_fields": ("raw_transcription", "transcription")},
    "cv": {"repo": "fsicoli/common_voice_17_0", "configs": {"ja": "ja", "en": "en"},
           "text_fields": ("sentence",)},
}


def _resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x.astype(np.float32)
    try:
        import librosa
        return librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    except Exception:
        return resample_linear(x, sr, target_sr)


def _text_of(row: dict, fields: tuple[str, ...]) -> str:
    for f in fields:
        v = row.get(f)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def pull(lang: str, n: int, audio_dir: Path, manifest_path: Path, target_sr: int,
         src: str, split: str) -> int:
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    spec = _SOURCES[src]
    config = spec["configs"][lang]
    prefix = f"{src[:2]}{lang}"
    print(f"[{lang}] streaming {n} rows from {spec['repo']}:{config} ({split}) ...")
    ds = load_dataset(spec["repo"], config, split=split, streaming=True)
    # decode=False -> raw bytes (avoids the torchcodec dependency the datasets Audio decoder now requires);
    # we decode with soundfile ourselves.
    ds = ds.cast_column("audio", Audio(decode=False))
    n_ok = 0
    for row in ds:
        if n_ok >= n:
            break
        sentence = _text_of(row, spec["text_fields"])
        audio = row.get("audio") or {}
        data = audio.get("bytes")
        if data is None and audio.get("path"):
            data = Path(audio["path"]).read_bytes()
        if not sentence or not data:
            continue
        arr, src_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        wav = _resample(arr, int(src_sr), target_sr)
        if len(wav) < target_sr * 0.3:        # skip <0.3 s clips
            continue
        uid = f"{prefix}_{n_ok:05d}"
        save_wav(str(audio_dir / f"{uid}.wav"), wav, target_sr)
        append_jsonl(manifest_path, {
            "id": uid,
            "audio_path": f"audio/{uid}.wav",
            "transcript": sentence,
            "spans": [{"text": sentence, "lang": lang}],
            "style": f"mono_{lang}",
            "voice": "real",
            "domain": "general",
            "source": f"{src}_{lang}",
            "duration_s": round(len(wav) / target_sr, 3),
            "sr": target_sr,
        })
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"[{lang}] {n_ok}/{n} ...")
    print(f"[{lang}] done: {n_ok} rows -> {manifest_path}")
    return n_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ja", type=int, default=1500, help="# Japanese replay utterances")
    ap.add_argument("--en", type=int, default=500, help="# English replay utterances")
    ap.add_argument("--out-dir", default="data/mono")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--source", default="fleurs", choices=list(_SOURCES), help="dataset source (default FLEURS)")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for lang, n in [("ja", args.ja), ("en", args.en)]:
        if n <= 0:
            continue
        manifest_path = out_dir / f"manifest_{lang}.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()
        pull(lang, n, audio_dir, manifest_path, args.sr, args.source, args.split)

    print(f"\nmono replay at {out_dir} (manifest_ja.jsonl / manifest_en.jsonl). Next: scripts/mix_manifests.py")


if __name__ == "__main__":
    main()
