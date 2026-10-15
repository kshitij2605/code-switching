#!/usr/bin/env python3
"""Engine #2 — render CS transcripts with Kokoro (local GPU).

Kokoro is language-locked per pipeline (separate JA and EN G2P), so it cannot
code-switch inside one voice. We splice: each transcript's segments are rendered
with the pipeline+voice matching their language, then concatenated. To keep a
consistent "speaker" per utterance we use gender-matched (JA voice, EN voice)
pairs and rotate the pair across the corpus for multi-speaker diversity.

This is deliberately the *splice* engine in the comparison (vs edge-tts's single
multilingual voice and CosyVoice2's native in-utterance switching).

Kokoro emits 24 kHz; we resample to 16 kHz to match the corpus / eval.

Run (on mactrn01, in the venv, on a free GPU):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
     scripts/01-data-generation/asr-ft/engines/render_kokoro.py \
     --in data/transcripts/cs_transcripts.jsonl --out data/synth/kokoro --limit 100
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from kokoro import KPipeline

KOKORO_SR = 24000
SR = 16000
GAP_MS = 60                       # silence between spliced segments

# gender-matched (JA voice, EN voice) speaker pairs -> one pair == one speaker
PAIRS = [
    ("jf_alpha", "af_bella", "F"),
    ("jf_gongitsune", "af_heart", "F"),
    ("jf_nezumi", "af_nicole", "F"),
    ("jf_tebukuro", "af_sarah", "F"),
    ("jm_kumo", "am_adam", "M"),
]


def trim_silence(a: np.ndarray, thr: float = 0.01, win: int = 200,
                 pad: int = 800) -> np.ndarray:
    """Strip Kokoro's leading/trailing near-silence, leaving a small pad.

    Without this the splice stacks each segment's padding and clips end up
    ~60% silence. pad (~33 ms at 24 kHz) keeps a natural breath at edges.
    """
    if a.size == 0:
        return a
    n = len(a) // win
    if n == 0:
        return a
    e = np.array([np.sqrt(np.mean(a[i * win:(i + 1) * win] ** 2)) for i in range(n)])
    idx = np.where(e > thr)[0]
    if idx.size == 0:
        return a
    s = max(0, idx[0] * win - pad)
    t = min(len(a), (idx[-1] + 1) * win + pad)
    return a[s:t]


def synth(pipe: KPipeline, text: str, voice: str) -> np.ndarray:
    chunks = [au.detach().cpu().numpy() if torch.is_tensor(au) else np.asarray(au)
              for _, _, au in pipe(text, voice=voice)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return trim_silence(np.concatenate(chunks).astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/synth/kokoro")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"

    print("loading Kokoro JA + EN pipelines ...")
    en_pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    ja_pipe = KPipeline(lang_code="j", repo_id="hexgrad/Kokoro-82M")
    gap = np.zeros(int(KOKORO_SR * GAP_MS / 1000), dtype=np.float32)

    print(f"kokoro: rendering {len(rows)} utterances across {len(PAIRS)} speaker pairs")
    written, errors, total_dur = [], 0, 0.0
    for i, row in enumerate(rows):
        ja_voice, en_voice, gender = PAIRS[i % len(PAIRS)]
        try:
            parts = []
            for seg in row.get("segments", []):
                if seg["lang"] == "ja":
                    parts.append(synth(ja_pipe, seg["text"], ja_voice))
                else:
                    parts.append(synth(en_pipe, seg["text"], en_voice))
                parts.append(gap)
            audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if audio.size == 0:
                raise RuntimeError("empty audio")
            # 24k -> 16k  (2/3 polyphase resample)
            audio16 = resample_poly(audio, SR, KOKORO_SR).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  ERR {row.get('id')}: {type(e).__name__}: {e}")
            continue

        uid = f"kokoro_{row['id']}"
        wav_path = audio_dir / f"{uid}.wav"
        sf.write(str(wav_path), audio16, SR)
        dur = len(audio16) / SR
        total_dur += dur
        written.append({
            "id": uid,
            "audio_path": str(wav_path.resolve()),
            "transcript": row["transcript"],
            "en_words": row.get("en_words", []),
            "ja_text": row.get("ja_text", ""),
            "segments": row.get("segments", []),
            "domain": row.get("domain", ""),
            "n_switches": row.get("n_switches", 0),
            "style": row.get("style", ""),
            "density": row.get("density", ""),
            "span_type": row.get("span_type", ""),
            "engine": "kokoro",
            "voice": f"{ja_voice}+{en_voice}",
            "speaker": f"{ja_voice}+{en_voice}",
            "gender": gender,
            "duration_s": round(dur, 3),
            "sr": SR,
            "source": "synth/kokoro",
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} ...")

    with manifest.open("w", encoding="utf-8") as f:
        for m in written:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    from collections import Counter
    by_spk = Counter(m["voice"] for m in written)
    print(f"\ndone: {len(written)} ok, {errors} err")
    print(f"  total audio : {total_dur / 60:.1f} min ({total_dur / max(len(written),1):.1f} s/clip)")
    print(f"  per speaker : {dict(by_spk)}")
    print(f"  manifest    : {manifest}")


if __name__ == "__main__":
    main()
