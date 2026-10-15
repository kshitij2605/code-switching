"""
Augment CS/EN/JA synthetic audio with MUSAN noise and music.
Creates 3 noisy copies per original sample:
  _aug1 — MUSAN noise,  SNR ~20 dB
  _aug2 — MUSAN noise,  SNR ~13 dB
  _aug3 — MUSAN music,  SNR ~20 dB

Appends new rows to train.jsonl with augmented audio paths. Each augmented row
keeps its source row's "group", so group-aware splits keep a clip and its noisy
copies together, and inherits the MUSAN licence (common.licenses.add_derivation).
eval.jsonl stays clean unless --augment-eval is given.

Usage:
  python augment_data.py \
    --data-root /data/lfm2.5/run-a \
    --musan-dir /data/lfm2.5/musan_extracted/musan
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.licenses import add_derivation, augmentation_sources  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   default="/data/lfm2.5/run-a")
    p.add_argument("--musan-dir",   default="/data/lfm2.5/musan_extracted/musan")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--train-jsonl", default="data/training/train.jsonl")
    p.add_argument("--eval-jsonl",  default="data/training/eval.jsonl")
    p.add_argument("--augment-eval", action="store_true",
                   help="Also add noisy copies of eval clips to eval.jsonl")
    return p.parse_args()


def load_wav(path: str) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def save_wav(path: str, arr: np.ndarray, sr: int):
    sf.write(path, arr, sr, subtype="PCM_16")


def mix_at_snr(signal: np.ndarray, noise_pool: list[str], sr: int,
               snr_db: float, rng: random.Random) -> np.ndarray:
    noise_path = rng.choice(noise_pool)
    noise, _ = load_wav(noise_path)

    # Tile noise to cover signal length
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)
    if len(noise) > len(signal):
        start = rng.randint(0, len(noise) - len(signal))
        noise = noise[start: start + len(signal)]

    sig_rms   = np.sqrt(np.mean(signal ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise  ** 2)) + 1e-9
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = signal + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def collect_wav_files(musan_dir: str, subset: str) -> list[str]:
    pattern = os.path.join(musan_dir, subset, "**", "*.wav")
    files = glob.glob(pattern, recursive=True)
    assert files, f"No WAV files found under {musan_dir}/{subset}/"
    return files


def augment_category(category: str, data_root: str,
                     noise_pool: list[str], music_pool: list[str],
                     rng: random.Random) -> list[tuple[str, str]]:
    """
    Create augmented WAV files for every original in the category.
    Returns list of (original_audio_relpath, aug_audio_relpath) pairs.
    """
    audio_root = os.path.join(data_root, "data", category, "audio")
    aug_root   = os.path.join(data_root, "data", category, "audio_aug")
    os.makedirs(aug_root, exist_ok=True)

    orig_wavs = sorted(glob.glob(os.path.join(audio_root, "*.wav")))
    aug_configs = [
        ("aug1", noise_pool, 20.0),
        ("aug2", noise_pool, 13.0),
        ("aug3", music_pool, 20.0),
    ]

    pairs: list[tuple[str, str]] = []  # (orig_rel, aug_rel)
    for orig_path in orig_wavs:
        signal, sr = load_wav(orig_path)
        stem = Path(orig_path).stem  # e.g. "cs_001"

        orig_rel = os.path.join("data", category, "audio", Path(orig_path).name)

        for suffix, pool, snr in aug_configs:
            aug_name = f"{stem}_{suffix}.wav"
            aug_path = os.path.join(aug_root, aug_name)
            if not os.path.exists(aug_path):
                mixed = mix_at_snr(signal, pool, sr, snr, rng)
                save_wav(aug_path, mixed, sr)
            aug_rel = os.path.join("data", category, "audio_aug", aug_name)
            pairs.append((orig_rel, aug_rel))

    print(f"  {category}: {len(orig_wavs)} originals → {len(pairs)} augmented")
    return pairs


def update_jsonl(jsonl_path: str, orig_to_augs: dict[str, list[str]]):
    """
    For every row in jsonl_path whose audio_file has augmented versions,
    append 3 new rows (one per augmented file) with the same system/target/type.
    Skip if augmented rows already exist.
    """
    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    existing_audio = {r["audio_file"] for r in rows}

    new_rows = []
    for row in rows:
        af = row["audio_file"].replace("\\", "/")
        # Normalize to forward slash for matching
        for orig_rel, aug_rels in orig_to_augs.items():
            orig_rel_norm = orig_rel.replace("\\", "/")
            if af == orig_rel_norm or af.endswith("/" + Path(orig_rel_norm).name):
                for aug_rel in aug_rels:
                    aug_rel_norm = aug_rel.replace("\\", "/")
                    if aug_rel_norm not in existing_audio:
                        suffix = Path(aug_rel_norm).stem.rsplit("_", 1)[-1]   # aug1/aug2/aug3
                        new_row = dict(row)
                        new_row["audio_file"] = aug_rel_norm
                        new_row["augmentation"] = suffix
                        if row.get("id"):
                            new_row["id"] = f"{row['id']}_{suffix}"
                        # same "group" (or, without one, same transcript) as the source row
                        new_rows.append(add_derivation(new_row, augmentation_sources(suffix)))
                        existing_audio.add(aug_rel_norm)
                break

    if new_rows:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {jsonl_path}: +{len(new_rows)} augmented rows (total now {len(rows)+len(new_rows)})")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    root   = args.data_root
    musan  = args.musan_dir

    print("Collecting MUSAN noise and music files...")
    noise_pool = collect_wav_files(musan, "noise")
    music_pool = collect_wav_files(musan, "music")
    print(f"  noise: {len(noise_pool)} files")
    print(f"  music: {len(music_pool)} files")

    categories = ["code_switching", "english_only", "japanese_only"]

    # Build orig→aug mapping
    # orig_to_augs: { orig_rel_path: [aug1_rel, aug2_rel, aug3_rel] }
    orig_to_augs: dict[str, list[str]] = {}
    for cat in categories:
        print(f"\n[{cat}] Augmenting audio...")
        pairs = augment_category(cat, root, noise_pool, music_pool, rng)
        for orig_rel, aug_rel in pairs:
            orig_to_augs.setdefault(orig_rel, []).append(aug_rel)

    # Update JSONL files
    print("\nUpdating JSONL files...")
    update_jsonl(os.path.join(root, args.train_jsonl), orig_to_augs)
    if args.augment_eval:
        update_jsonl(os.path.join(root, args.eval_jsonl), orig_to_augs)
    else:
        print(f"  {args.eval_jsonl}: left clean (pass --augment-eval to add noisy copies)")

    total_new_audio = sum(len(v) for v in orig_to_augs.values())
    print(f"\nDone. {total_new_audio} augmented WAV files created.")
    print("Next: re-run preprocess_asr_data.py to build an augmented preprocessed dataset.")


if __name__ == "__main__":
    main()
