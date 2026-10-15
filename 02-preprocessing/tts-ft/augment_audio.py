#!/usr/bin/env python
"""Acoustic augmentation to close the clean-synthetic -> real-audio gap (JECS).

Takes a manifest (our schema), and for each row writes augmented copies with a random chain of: additive MUSAN
noise (random SNR), synthetic reverb (decaying-noise RIR), and gain. Transcript/spans are unchanged (label-preserving),
so the output is a drop-in manifest you mix back in with the originals before preprocessing.

  python scripts/augment_audio.py --manifest data/cs_mixed/train.jsonl --out-dir data/aug \
      --musan-noise /data/lfm2.5/musan/musan/noise --n 1
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "05-app-and-demo" / "kaigi-app"  # stage-folder copy of kaigi-app/scripts
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.manifest import append_jsonl, read_jsonl  # noqa: E402
from csmeeting.tts_backends import resample_linear, save_wav  # noqa: E402


def _load_mono(path: str, target_sr: int) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return resample_linear(y, sr, target_sr) if sr != target_sr else y


def _add_noise(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(y):                       # tile to cover
        noise = np.tile(noise, int(np.ceil(len(y) / max(1, len(noise)))))
    start = 0 if len(noise) <= len(y) else np.random.randint(0, len(noise) - len(y))
    noise = noise[start:start + len(y)]
    sig_p = float(np.mean(y ** 2)) + 1e-9
    noi_p = float(np.mean(noise ** 2)) + 1e-9
    scale = float(np.sqrt(sig_p / (10 ** (snr_db / 10) * noi_p)))
    return y + scale * noise


def _reverb(y: np.ndarray, sr: int, rt60: float) -> np.ndarray:
    from scipy.signal import fftconvolve
    n = max(1, int(sr * rt60))
    rir = np.random.randn(n).astype(np.float32) * np.exp(-3.0 * np.arange(n) / n).astype(np.float32)
    rir[0] = 1.0                                   # direct path
    rir /= np.sqrt(np.sum(rir ** 2)) + 1e-9
    wet = fftconvolve(y, rir)[:len(y)].astype(np.float32)
    return wet


def _peak_norm(y: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(y))) + 1e-9
    return (y * (peak / m)).astype(np.float32) if m > peak else y.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="data/aug")
    ap.add_argument("--musan-noise", default="/data/lfm2.5/musan/musan/noise")
    ap.add_argument("--n", type=int, default=1, help="augmented copies per source clip")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--p-noise", type=float, default=0.8)
    ap.add_argument("--p-reverb", type=float, default=0.5)
    ap.add_argument("--p-gain", type=float, default=0.5)
    ap.add_argument("--snr-min", type=float, default=5.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    noise_files = sorted(glob.glob(str(Path(args.musan_noise) / "**" / "*.wav"), recursive=True))
    if not noise_files and args.p_noise > 0:
        print(f"WARN: no MUSAN noise wavs under {args.musan_noise} — noise disabled")
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    rows = list(read_jsonl(args.manifest))
    if args.limit:
        rows = rows[:args.limit]
    n_ok = 0
    for i, row in enumerate(rows):
        src = Path(row["audio_path"])
        src = src if src.is_absolute() else Path(args.manifest).resolve().parent / src
        try:
            y0 = _load_mono(str(src), args.sr)
        except Exception as e:  # noqa: BLE001
            print(f"skip {row['id']}: {e}")
            continue
        for k in range(args.n):
            y = y0.copy()
            if noise_files and rng.random() < args.p_noise:
                noise = _load_mono(rng.choice(noise_files), args.sr)
                y = _add_noise(y, noise, rng.uniform(args.snr_min, args.snr_max))
            if rng.random() < args.p_reverb:
                y = _reverb(y, args.sr, rng.uniform(0.15, 0.5))
            if rng.random() < args.p_gain:
                y = y * (10 ** (rng.uniform(-5, 4) / 20))
            y = _peak_norm(y)
            uid = f"{row['id']}_aug{k}"
            save_wav(str(audio_dir / f"{uid}.wav"), y, args.sr)
            new = dict(row)
            new["id"] = uid
            new["audio_path"] = str((audio_dir / f"{uid}.wav").resolve())   # absolute → mixes with any manifest
            new["source"] = f"{row.get('source', 'unknown')}_aug"
            new["duration_s"] = round(len(y) / args.sr, 3)
            new["sr"] = args.sr
            append_jsonl(manifest_path, new)
            n_ok += 1
        if (i + 1) % 200 == 0:
            print(f"[augment] {i + 1}/{len(rows)} ...")
    print(f"[augment] wrote {n_ok} augmented clips -> {manifest_path}")


if __name__ == "__main__":
    main()
