#!/usr/bin/env python3
"""Load monolingual FLEURS (google/fleurs) utterances for the eval benchmark.

FLEURS is the parent corpus of CS-FLEURS. The CS-FLEURS READ text was generated
by GPT-4o from parallel FLEURS *test* sentences and recorded separately by
CS-FLEURS's own bilingual readers. So the `ja_jp` and `en_us` test slices are a
source-sentence-matched control for forgetting (same sentence domain and
read-speech style, but different speakers and recordings). Only part of each
control set are source sentences of CS items.

`--split dev` (FLEURS validation) builds the mono *dev* sets used for
checkpoint selection. Rows also carry the training-contract fields
(`audio_path`, `transcript`, `group` = FLEURS sentence id), so train_whisper.py
--dev-manifests can read them directly.

We read the **parquet** form (`parquet-data/<config>/test-*.parquet`), which embeds
audio bytes + transcription, so we bypass the `datasets` Audio/torchcodec path
(same gotcha as data/load_csfleurs.py). Output manifest matches the shared
contract consumed by eval/run_baseline.py and eval/score.py.

Example:
    python data/load_fleurs.py --config ja_jp --split test --limit 200
    python data/load_fleurs.py --config en_us --split test --limit 200
    python data/load_fleurs.py --config ja_jp --split dev --limit 100000 --exclude-protected test
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

REPO = "google/fleurs"
_SPLIT2PARQUET = {"test": "test", "dev": "validation", "validation": "validation", "train": "train"}


def _to_16k_mono(wav_bytes: bytes):
    """Decode embedded audio bytes -> (np.float32 mono @ 16 kHz, 16000)."""
    import librosa
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        sr = 16000
    return np.asarray(arr, dtype="float32"), sr


def extract(config: str, split: str, limit: int, out_dir: str,
            exclude_roles: list[str] | None = None) -> str:
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    pslot = _SPLIT2PARQUET.get(split, split)
    rel = f"parquet-data/{config}/{pslot}-00000-of-00001.parquet"
    print(f"Downloading parquet: {rel}")
    pf = hf_hub_download(REPO, rel, repo_type="dataset")
    rows = pq.read_table(pf).to_pylist()
    # FLEURS reuses the same numeric id for one sentence read by multiple speakers.
    # Dedupe to one recording per sentence (distinct sentences = broad coverage,
    # unique ids), deterministically: sort by id, keep first occurrence, take limit.
    rows.sort(key=lambda r: int(r["id"]))
    seen, uniq = set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        uniq.append(r)
    rows = uniq[:limit]
    print(f"[{config}/{split}] {len(rows)} distinct-sentence rows selected (of {len(uniq)} unique)")

    tag = f"{config}_{split}"
    audio_dir = os.path.join(out_dir, tag, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, tag, "manifest.jsonl")

    out_rows = []
    total_dur = 0.0
    for i, r in enumerate(rows):
        uid = f"{config}_{r['id']}"
        try:
            arr, sr = _to_16k_mono(r["audio"]["bytes"])
        except Exception as e:  # noqa: BLE001
            print(f"  skip {uid} ({type(e).__name__}: {e})")
            continue
        wav_path = os.path.join(audio_dir, f"{uid}.wav")
        sf.write(wav_path, arr, sr)
        dur = len(arr) / sr
        total_dur += dur
        reference = r.get("raw_transcription") or r.get("transcription") or ""
        out_rows.append({
            "id": uid,
            "reference": reference,
            "hypothesis": "",
            "audio": os.path.relpath(wav_path, out_dir),
            "lang": r.get("language", config),
            "duration_sec": round(dur, 3),
            "gender": {0: "male", 1: "female"}.get(r.get("gender"), r.get("gender")),
            "source": f"fleurs/{config}/{split}",
            # training-contract fields (train_whisper.py --dev-manifests)
            "audio_path": os.path.abspath(wav_path),
            "transcript": reference,
            "group": f"fleurs:{r['id']}",
        })
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1} written")

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from common.licenses import tag_rows
    out_rows = tag_rows(out_rows, "google/fleurs")
    if exclude_roles:
        from common.leak_guard import LeakGuard
        n0 = len(out_rows)
        out_rows = LeakGuard(roles=exclude_roles).filter_rows(
            out_rows, on_leak="drop", context=f"{tag} vs {'+'.join(exclude_roles)}")
        print(f"[{tag}] dropped {n0 - len(out_rows)} rows overlapping protected "
              f"{'+'.join(exclude_roles)} data")

    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    spk = len({r["audio"]["path"] for r in rows if r.get("audio")})
    print(f"\n[{tag}] kept={len(out_rows)} total_audio={total_dur / 60:.1f} min "
          f"| genders={ {x['gender'] for x in out_rows} }")
    print(f"[{tag}] manifest: {manifest_path}")
    print("\n--- sample references ---")
    for r in out_rows[:5]:
        print(f"  {r['reference'][:90]}")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load monolingual FLEURS test utts")
    ap.add_argument("--config", required=True, help="FLEURS config, e.g. ja_jp or en_us")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="data/fleurs")
    ap.add_argument("--exclude-protected", default="",
                    help="Comma-separated leak-guard roles whose overlapping rows are dropped "
                         "(use 'test' when building a dev set)")
    args = ap.parse_args(argv)
    roles = [r for r in args.exclude_protected.split(",") if r]
    extract(args.config, args.split, args.limit, args.out, roles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
