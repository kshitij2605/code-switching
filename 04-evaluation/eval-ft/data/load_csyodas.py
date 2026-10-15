#!/usr/bin/env python3
"""Load the Japanese slice of CS-YODAS (HF: byan/cs-yodas; CC BY-NC 4.0) as an
in-the-wild JA-EN code-switching TEST set (Phase 1, T-MULTI).

Each jpn.jsonl record is one human-verified code-switched utterance. All 242
records have consistent=1, Q1-Q4=Yes and Q5=No. The audio in jpn.tar is a
~60 s YouTube context clip (`wav_path`), and the utterance sits inside it:

    kaldi_uttid  '<video>_jpn_<ctx_start_ms>_<ctx_end_ms>'   context clip
    id           '<video>_<seg_start_ms>_<seg_end_ms>_jpn_asr_<a_10ms>_<b_10ms>'
    utterance    [seg_start + a, seg_start + b] - ctx_start   seconds within the clip

For every record the computed span agrees with the record's own
context_text_list entry for the same text, within 0.1 s (checked 2026-09-30).
The loader uses the context_text_list boundaries and falls back to the id
arithmetic.

The output manifest follows the shared eval contract
({id, reference, audio, lang, duration_sec, speaker, source}). `speaker` is the
YouTube video id, the clustering unit for speaker-level bootstrap (214 videos
for 242 utterances). The set is registered as protected test data in
common/eval_blocklist.json, so it can never enter training.

Example:
    python data/load_csyodas.py --out /data/code-switching/data/eval-real/csyodas
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO = "byan/cs-yodas"
_ID = re.compile(r"^(?P<video>.+?)_(?P<s0>\d{9})_(?P<s1>\d{9})_jpn_asr_(?P<a>\d{9})_(?P<b>\d{9})$")
_CTX = re.compile(r"_jpn_(?P<c0>\d{9})_(?P<c1>\d{9})$")


def span(rec: dict) -> tuple[float, float]:
    """(start, end) in seconds within the context clip."""
    segs = rec.get("context_text_list")
    if isinstance(segs, str):
        segs = ast.literal_eval(segs)
    key = rec["text"].replace(" ", "")
    for text, s, e in segs or []:
        if text.replace(" ", "") == key:
            return float(s), float(e)
    m, c = _ID.match(rec["id"]), _CTX.search(rec["kaldi_uttid"])
    seg0, ctx0 = int(m["s0"]) / 1000, int(c["c0"]) / 1000
    return seg0 + int(m["a"]) / 100 - ctx0, seg0 + int(m["b"]) / 100 - ctx0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/csyodas", help="Output dir (manifest + audio/)")
    args = ap.parse_args(argv)

    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from common.licenses import tag_rows

    meta = hf_hub_download(REPO, "jpn.jsonl", repo_type="dataset")
    recs = [json.loads(l) for l in open(meta, encoding="utf-8") if l.strip()]
    tar_path = hf_hub_download(REPO, "jpn.tar", repo_type="dataset")
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    wanted = {r["wav_path"] for r in recs}
    clips = {}
    with tarfile.open(tar_path) as tar:
        for m in tar:
            name = m.name.lstrip("./")
            hit = next((w for w in wanted if name.endswith(w)), None)
            if hit and m.isfile():
                clips[hit] = tar.extractfile(m).read()
    missing = sorted(wanted - set(clips))
    if missing:
        raise SystemExit(f"{len(missing)} context clips missing from jpn.tar, e.g. {missing[:3]}")

    rows, total = [], 0.0
    for r in recs:
        a, sr = sf.read(io.BytesIO(clips[r["wav_path"]]), dtype="float32", always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            import librosa
            a, sr = librosa.resample(a, orig_sr=sr, target_sr=16000), 16000
        s, e = span(r)
        seg = np.asarray(a[max(0, int(s * sr)):min(len(a), int(e * sr))], dtype="float32")
        if seg.size < sr // 10:
            raise SystemExit(f"{r['id']}: empty or <0.1 s segment at [{s:.2f}, {e:.2f}]")
        uid = r["id"]
        wav = out / "audio" / f"{uid}.wav"
        sf.write(wav, seg, sr)
        dur = len(seg) / sr
        total += dur
        video = _ID.match(uid)["video"]
        rows.append({"id": uid, "reference": r["text"], "hypothesis": "",
                     "audio": os.path.relpath(wav, out), "lang": "jpn-eng",
                     "duration_sec": round(dur, 3), "speaker": video,
                     "source": "cs-yodas/jpn", "group": f"csyodas:{video}",
                     "audio_path": str(wav.resolve()), "transcript": r["text"]})
    rows = tag_rows(rows, REPO)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[cs-yodas/jpn] {len(rows)} utterances, {total / 60:.1f} min, "
          f"{len({r['speaker'] for r in rows})} videos -> {out / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
