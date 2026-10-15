#!/usr/bin/env python3
"""Load the Japanese-English slice of CS-FLEURS (HF: byan/cs-fleurs).

CS-FLEURS is stored as raw WAVs + a `metadata.jsonl` per generation-method/split:

    read/test/metadata.jsonl   <- REAL read speech (most credible eval slice)
    mms/test/metadata.jsonl    <- MMS-TTS synthetic
    xtts/{train,test1,test2}/metadata.jsonl  <- XTTS synthetic (bulk of train)

The HF `datasets` auto-loader merges these into train/test splits with an Audio
feature and then tries to re-encode the decoded arrays via `torchcodec` on every
read (ImportError if torchcodec/FFmpeg absent). We sidestep `datasets` entirely:
read the metadata.jsonl directly and pull only the JA-EN ('jpn-eng') WAVs.

Each metadata row: {id, file_name, text, duration, fluency, language, speaker}.
We resample to 16 kHz mono and emit a manifest JSONL of
{id, reference, hypothesis:"", audio, lang, duration_sec, fluency, speaker, source}
— the exact shape eval/run_baseline.py and eval/score.py consume.

Rows also carry training-contract fields (`audio_path`, `transcript`, `group`),
so a dev manifest can go straight to train_whisper.py --dev-manifests. XTTS/MMS
text is cleaned of its ** markers and MeCab spacing (common/csfleurs.py); the raw
text is kept in `reference_raw`.

Examples:
    python data/load_csfleurs.py --inspect                 # schema + lang counts
    python data/load_csfleurs.py --method read --split test --limit 50
    # CS dev set for checkpoint selection: xtts/test1 minus rows overlapping test data
    python data/load_csfleurs.py --method xtts --split test1 --limit 100000 --exclude-protected test

NOTE: CS-FLEURS switch points are machine-generated in every subset. The READ
text was written by GPT-4o from parallel FLEURS test sentences, then
human-validated and read aloud. The XTTS/MMS text comes from align-then-swap.
It is a credibility baseline, NOT the spontaneous gold set. Evaluate the final
model on natural speech as well (Phase 2). The data is
CC BY-NC 4.0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO = "byan/cs-fleurs"

JA_HINTS = ["jpn", "japanese", "ja"]
EN_HINTS = ["eng", "english", "en"]


def _is_ja_en(language: str) -> bool:
    parts = [p for p in str(language).lower().replace("_", "-").split("-") if p]
    has_ja = any(p in JA_HINTS for p in parts) or any(h in str(language).lower() for h in ("jpn", "japanese"))
    has_en = any(p in EN_HINTS for p in parts) or "eng" in str(language).lower()
    return has_ja and has_en


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _meta_repo_path(method: str, split: str) -> str:
    return f"{method}/{split}/metadata.jsonl"


def inspect() -> None:
    from collections import Counter
    from huggingface_hub import hf_hub_download

    for method, split in [("read", "test"), ("mms", "test"), ("xtts", "train")]:
        rel = _meta_repo_path(method, split)
        try:
            p = hf_hub_download(REPO, rel, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print(f"[{rel}] unavailable: {type(e).__name__}: {e}")
            continue
        rows = _read_jsonl(p)
        langs = Counter(r.get("language", "?") for r in rows)
        ja_en = sum(1 for r in rows if _is_ja_en(r.get("language", "")))
        print(f"\n=== {rel} : {len(rows)} rows, {len(langs)} language pairs, JA-EN={ja_en} ===")
        print("  top pairs:", langs.most_common(8))
        print("  fields:", list(rows[0].keys()))
        for r in rows:
            if _is_ja_en(r.get("language", "")):
                print(f"  JA-EN example id={r.get('id')} lang={r.get('language')}")
                print(f"    text: {r.get('text')}")
                break


def _to_16k_mono(src_wav: str):
    import librosa
    import numpy as np
    import soundfile as sf
    arr, sr = sf.read(src_wav, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        sr = 16000
    return np.asarray(arr, dtype="float32"), sr


def extract(method: str, split: str, limit: int, out_dir: str,
            exclude_roles: list[str] | None = None) -> str:
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from common.csfleurs import clean_xtts_text, is_garbled, sentence_id

    rel = _meta_repo_path(method, split)
    print(f"Downloading metadata: {rel}")
    meta_path = hf_hub_download(REPO, rel, repo_type="dataset")
    rows = _read_jsonl(meta_path)
    ja_en = [r for r in rows if _is_ja_en(r.get("language", ""))]
    print(f"[{rel}] {len(rows)} total rows, {len(ja_en)} JA-EN rows; taking up to {limit}")

    tag = f"{method}_{split}"
    audio_dir = os.path.join(out_dir, tag, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, tag, "manifest.jsonl")

    # XTTS/MMS text is MeCab-spaced with **English** markers: clean it (the raw
    # text is kept as reference_raw). READ text is already natural text.
    cands = []
    for i, r in enumerate(ja_en):
        uid = str(r.get("id", f"{tag}_{i:05d}")).replace("/", "_")
        ref = r["text"] if method == "read" else clean_xtts_text(r["text"])
        cands.append((r, {"id": uid, "reference": ref}))
    if exclude_roles:
        from common.leak_guard import LeakGuard
        g = LeakGuard(roles=exclude_roles)
        n0 = len(cands)
        cands = [(r, c) for r, c in cands if not g.check_row(c)]
        print(f"[{tag}] dropped {n0 - len(cands)} of {n0} rows overlapping protected "
              f"{'+'.join(exclude_roles)} data")

    out_rows = []
    total_dur = 0.0
    for i, (r, c) in enumerate(cands[:limit]):
        repo_audio = f"{method}/{split}/{r['file_name']}"
        try:
            src = hf_hub_download(REPO, repo_audio, repo_type="dataset")
            arr, sr = _to_16k_mono(src)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {r.get('id')} ({type(e).__name__}: {e})")
            continue
        uid = c["id"]
        wav_path = os.path.join(audio_dir, f"{uid}.wav")
        sf.write(wav_path, arr, sr)
        dur = len(arr) / sr
        total_dur += dur
        row = {
            "id": uid,
            "reference": c["reference"],
            "hypothesis": "",
            "audio": os.path.relpath(wav_path, out_dir),
            "lang": r.get("language", ""),
            "duration_sec": round(dur, 3),
            "fluency": r.get("fluency"),
            "speaker": r.get("speaker"),
            "source": f"cs-fleurs/{method}/{split}",
            # training-contract fields (train_whisper.py --dev-manifests)
            "audio_path": os.path.abspath(wav_path),
            "transcript": c["reference"],
            "group": f"csf-{method}:{sentence_id(uid) if method != 'read' else uid}",
        }
        if method != "read":
            row["reference_raw"] = r["text"]
            row["garbled"] = is_garbled(r["text"])
        from common.licenses import tag_row
        out_rows.append(tag_row(row, "byan/cs-fleurs"))
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1} fetched")

    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[{tag}] kept={len(out_rows)} total_audio={total_dur / 60:.1f} min")
    print(f"[{tag}] manifest: {manifest_path}")
    print(f"[{tag}] wavs dir: {audio_dir}/")
    print("\n--- sample references (eyeball the switching style) ---")
    for r in out_rows[:8]:
        print(f"  [{r['lang']}] {r['reference']}")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load JA-EN slice of CS-FLEURS")
    ap.add_argument("--inspect", action="store_true",
                    help="Print metadata schema + language counts, then exit")
    ap.add_argument("--method", default="read", choices=["read", "mms", "xtts"],
                    help="Generation method (read = real read-speech, best for eval)")
    ap.add_argument("--split", default="test", help="Split within the method (e.g. test, train)")
    ap.add_argument("--limit", type=int, default=50, help="Max JA-EN examples to fetch")
    ap.add_argument("--out", default="data/csfleurs", help="Output dir")
    ap.add_argument("--exclude-protected", default="",
                    help="Comma-separated leak-guard roles whose overlapping rows are dropped "
                         "(use 'test' when building the xtts/test1 dev set)")
    args = ap.parse_args(argv)

    if args.inspect:
        inspect()
        return 0
    roles = [r for r in args.exclude_protected.split(",") if r]
    extract(args.method, args.split, args.limit, args.out, roles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
