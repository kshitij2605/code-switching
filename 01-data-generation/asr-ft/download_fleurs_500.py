"""
Download N monolingual FLEURS clips each for EN and JA (train split).
Saves WAV files + a meta.jsonl per language so transcriptions survive restarts.

Notes:
  * Monolingual only: CS training data comes only from download_cs_fleurs_byan.py
    (CS-FLEURS xtts/train).
  * FLEURS validation is reserved as the dev set for checkpoint selection, so
    it is not added to training unless --include-validation is given.
  * Rows carry the FLEURS sentence id as "group". FLEURS is n-way parallel, so
    that id groups every recording of a sentence in both languages, and
    train/val splits keep them together.
  * Default target text is raw_transcription (cased, punctuated), matching
    the CS data; --text-field transcription uses the lower-cased text.
  * Rows are licence-tagged (CC BY 4.0) and must pass common/leak_guard.py.

Usage:
  python download_fleurs_500.py --data-root /data/lfm2.5/run-a --n 500
"""
from __future__ import annotations

import argparse, io, json, os, sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.leak_guard import LeakGuard  # noqa: E402
from common.licenses import tag_rows  # noqa: E402
from common.run_registry import write_run_record  # noqa: E402

OUT_NAME = "fleurs500_rows.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="/data/lfm2.5/run-a")
    p.add_argument("--hf-token",  default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--n",         type=int, default=500)
    p.add_argument("--text-field", default="raw_transcription",
                   choices=["raw_transcription", "transcription"])
    p.add_argument("--include-validation", action="store_true",
                   help="Also add FLEURS validation clips to training (then FLEURS "
                        "validation cannot be used for checkpoint selection)")
    return p.parse_args()


def save_wav(path, arr, sr):
    arr = arr.astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    peak = np.max(np.abs(arr))
    if peak > 0:
        arr = arr / peak * 0.9
    sf.write(path, arr, sr, subtype="PCM_16")


def _row(prefix, rec):
    return {"id": f"fleurs_{prefix}_{rec['fname'][:-4]}",
            "group": f"fleurs:{rec['fleurs_id']}" if rec.get("fleurs_id") is not None else None,
            "audio_file": os.path.join("data", "fleurs_500", prefix, rec["fname"]),
            "system": "Perform ASR.", "target": rec["text"], "type": f"fleurs_{prefix}"}


def download_lang(dataset_name, config, split, out_dir, prefix, n, text_field):
    from datasets import load_dataset, Audio as HfAudio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"

    # Load already-saved transcriptions
    saved = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("text_field", "transcription") != text_field:
                    raise SystemExit(f"{meta_path} was built with text_field="
                                     f"{r.get('text_field', 'transcription')}; delete it or pass "
                                     f"--text-field {r.get('text_field', 'transcription')}")
                saved[r["fname"]] = r

    print(f"\n{dataset_name} ({config}/{split}) — have {len(saved)}/{n} cached")
    if len(saved) < n:
        ds = load_dataset(dataset_name, config, split=split, streaming=True)
        ds = ds.cast_column("audio", HfAudio(decode=False))

        meta_f = open(meta_path, "a", buffering=1, encoding="utf-8")  # line-buffered
        seen = len(saved)
        for ex in ds:
            if seen >= n:
                break
            audio_data = ex["audio"]
            raw = audio_data.get("bytes") or audio_data.get("path")
            if raw is None:
                continue
            try:
                if isinstance(raw, bytes):
                    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                else:
                    arr, sr = sf.read(raw, dtype="float32", always_2d=False)
            except Exception as e:
                print(f"  skip: {e}")
                continue
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if len(arr) / sr < 1.0:
                continue
            text = (ex.get(text_field) or "").strip()
            if not text:
                continue

            fname = f"{prefix}_{seen:03d}.wav"
            wav_path = out_dir / fname
            if not wav_path.exists():
                save_wav(str(wav_path), arr, sr)

            # Save transcription immediately
            rec = {"fname": fname, "text": text, "fleurs_id": ex.get("id"),
                   "text_field": text_field}
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved[fname] = rec
            seen += 1
            if seen % 100 == 0:
                print(f"  {seen}/{n}", flush=True)
        meta_f.close()
    print(f"  Done: {len(saved)} rows")
    return [_row(prefix, rec) for rec in list(saved.values())[:n]]


def main():
    args = parse_args()
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    guard = LeakGuard()  # fails fast if the blocklist is missing
    root = Path(args.data_root)
    base = root / "data" / "fleurs_500"

    all_rows = []
    for cfg, prefix in (("en_us", "en"), ("ja_jp", "ja")):
        all_rows += download_lang("google/fleurs", cfg, "train", base / prefix, prefix,
                                  args.n, args.text_field)
    if args.include_validation:
        print("WARNING: adding FLEURS validation to training; select checkpoints on another dev set")
        for cfg, prefix in (("en_us", "en_val"), ("ja_jp", "ja_val")):
            all_rows += download_lang("google/fleurs", cfg, "validation", base / prefix, prefix,
                                      args.n, args.text_field)

    all_rows = tag_rows(all_rows, "google/fleurs")
    guard.filter_rows(all_rows, on_leak="raise", context=OUT_NAME)
    print("\nFLEURS download complete:", {t: sum(r["type"] == t for r in all_rows)
                                         for t in sorted({r["type"] for r in all_rows})})

    out = root / "data" / "training" / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(all_rows)} rows → {out}")
    write_run_record(out.parent, vars(args), inputs=[str(out)], extra={"rows": len(all_rows)},
                     filename=OUT_NAME.replace(".jsonl", ".build.json"))


if __name__ == "__main__":
    main()
