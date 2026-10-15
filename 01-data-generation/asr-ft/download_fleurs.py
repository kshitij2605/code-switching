"""
Download N samples each from FLEURS EN (en_us) and FLEURS JA (ja_jp), train split,
save WAV files and append the rows to the native_asr JSONL files.
All use prompt: "Perform ASR."

Notes:
  * Monolingual only: real CS training data comes from download_cs_fleurs_byan.py.
  * The train/eval split keeps all recordings of a FLEURS sentence together
    (the FLEURS sentence id is stored as "group").
  * Rows are licence-tagged (CC BY 4.0) and must pass the leak guard.

Usage:
  python download_fleurs.py --data-root /data/lfm2.5/run-a --hf-token hf_xxx
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import io
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.leak_guard import LeakGuard, check_disjoint, group_split, row_group_key  # noqa: E402
from common.licenses import tag_rows  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root",  default="/data/lfm2.5/run-a")
    p.add_argument("--hf-token",   default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--n-samples",  type=int, default=200)
    p.add_argument("--seed",       type=int, default=99)
    return p.parse_args()


def save_wav(path: str, array: np.ndarray, sr: int):
    arr = array.astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    # Normalize
    peak = np.max(np.abs(arr))
    if peak > 0:
        arr = arr / peak * 0.9
    sf.write(path, arr, sr, subtype="PCM_16")


def download_fleurs_mono(dataset_name: str, config: str, split: str,
                         out_audio_dir: str, label_prefix: str,
                         n: int, transcription_key: str = "transcription") -> list[dict]:
    """Download n samples from a FLEURS-style dataset, return JSONL rows."""
    from datasets import load_dataset

    print(f"\nLoading {dataset_name} ({config}, {split})...")
    # Disable torchcodec — use soundfile backend via decode=False
    import datasets as hf_datasets
    hf_datasets.config.IN_MEMORY_MAX_SIZE = 0
    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    # Cast audio column to avoid torchcodec; get raw bytes instead
    from datasets import Audio as HfAudio
    ds = ds.cast_column("audio", HfAudio(decode=False))

    os.makedirs(out_audio_dir, exist_ok=True)
    rows = []
    seen = 0

    for i, ex in enumerate(ds):
        if seen >= n:
            break

        # Decode audio manually with soundfile (bypasses torchcodec)
        audio_data = ex["audio"]
        raw_bytes = audio_data.get("bytes") or audio_data.get("path")
        if raw_bytes is None:
            continue
        try:
            if isinstance(raw_bytes, bytes):
                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
            else:
                arr, sr = sf.read(raw_bytes, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"  skip {i}: audio decode error: {e}")
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        # Get transcription
        text = ex.get(transcription_key) or ex.get("sentence") or ex.get("raw_transcription", "")
        text = text.strip()
        if not text:
            continue

        # Skip very short clips (<1s)
        if len(arr) / sr < 1.0:
            continue

        fname = f"{label_prefix}_{seen:03d}.wav"
        fpath = os.path.join(out_audio_dir, fname)
        save_wav(fpath, arr, sr)

        rows.append({
            "id": f"fleurs_{label_prefix}_{seen:03d}",
            "group": f"fleurs:{ex.get('id')}" if ex.get("id") is not None else None,
            "audio_file": os.path.join("data", "fleurs", label_prefix, fname),
            "system": "Perform ASR.",
            "target": text,
            "type": f"fleurs_{label_prefix}",
        })
        seen += 1
        if seen % 50 == 0:
            print(f"  {seen}/{n} saved...")

    print(f"  Done: {len(rows)} samples saved to {out_audio_dir}")
    return rows


def append_to_jsonl(path: str, rows: list[dict]):
    existing = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                existing.add(r["audio_file"])
    new = [r for r in rows if r["audio_file"] not in existing]
    with open(path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Appended {len(new)} rows to {path}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    guard = LeakGuard()  # fails fast if the blocklist is missing
    root = args.data_root
    n    = args.n_samples

    train_jsonl = os.path.join(root, "data", "training", "native_asr_train.jsonl")
    eval_jsonl  = os.path.join(root, "data", "training", "native_asr_eval.jsonl")

    en_rows = download_fleurs_mono("google/fleurs", "en_us", "train",
                                   os.path.join(root, "data", "fleurs", "en"), "en", n)
    ja_rows = download_fleurs_mono("google/fleurs", "ja_jp", "train",
                                   os.path.join(root, "data", "fleurs", "ja"), "ja", n)
    all_rows = tag_rows(en_rows + ja_rows, "google/fleurs")
    guard.filter_rows(all_rows, on_leak="raise", context="download_fleurs")

    # ── Split 90/10 train/eval by FLEURS sentence and append to JSONL ────────
    train_rows, eval_rows = group_split(all_rows, 0.1, rng, row_group_key)
    check_disjoint(train_rows, eval_rows, context="download_fleurs")

    print(f"\nTotal: {len(all_rows)} rows → {len(train_rows)} train / {len(eval_rows)} eval")
    append_to_jsonl(train_jsonl, train_rows)
    append_to_jsonl(eval_jsonl,  eval_rows)

    print("\nDone. Summary:")
    print(f"  EN: {len(en_rows)}")
    print(f"  JA: {len(ja_rows)}")
    print(f"  Total new rows: {len(all_rows)}")
    print(f"\nNext: re-run preprocess_asr_data.py on native_asr_train/eval.jsonl")


if __name__ == "__main__":
    main()
