#!/usr/bin/env python3
"""Engine #3 — render CS transcripts with MeloTTS (CPU).

MeloTTS is language-locked, so — like Kokoro — we splice: JA segments via the JP
voice, EN segments via one of MeloTTS's 5 English accents (US / BR / India / AU /
Default). The English accent is rotated across the corpus for multi-speaker
diversity (the JP voice is single). Runs on CPU: MeloTTS ships a CPU torch here
and the model is a light VITS; 96 cores make 100 short clips quick.

MeloTTS emits 44.1 kHz; we resample to 16 kHz and trim splice silence.

Run (on mactrn01, in the melo venv):
  .venv-melo/bin/python scripts/01-data-generation/asr-ft/engines/render_melo.py \
     --in data/transcripts/cs_transcripts.jsonl --out data/synth/melo --limit 100
"""
from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MELO_SR = 44100
SR = 16000
GAP_MS = 60

# EN accents rotated across clips -> speaker diversity (JP voice is single)
EN_ACCENTS = ["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default"]


def trim_silence(a: np.ndarray, thr: float = 0.01, win: int = 400,
                 pad: int = 1500) -> np.ndarray:
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


def synth(model, text: str, sid: int, tmp: Path) -> np.ndarray:
    """Return a 44.1 kHz float array for `text` in speaker `sid`."""
    out = model.tts_to_file(text, sid, None, speed=1.0)
    if out is None:                      # some versions require a path
        p = tmp / "seg.wav"
        model.tts_to_file(text, sid, str(p), speed=1.0)
        out, _ = sf.read(str(p))
    a = np.asarray(out, dtype=np.float32)
    return trim_silence(a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/synth/melo")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from melo.api import TTS

    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"

    print("loading MeloTTS JP + EN models (CPU) ...")
    jp = TTS(language="JP", device="cpu")
    en = TTS(language="EN", device="cpu")
    jp_sid = list(jp.hps.data.spk2id.values())[0]
    en_sid = {k: v for k, v in en.hps.data.spk2id.items()}
    gap = np.zeros(int(SR * GAP_MS / 1000), dtype=np.float32)

    print(f"melo: rendering {len(rows)} utterances, EN accent rotated over {len(EN_ACCENTS)}")
    written, errors, total_dur = [], 0, 0.0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, row in enumerate(rows):
            accent = EN_ACCENTS[i % len(EN_ACCENTS)]
            try:
                parts = []
                for seg in row.get("segments", []):
                    if seg["lang"] == "ja":
                        a = synth(jp, seg["text"], jp_sid, tmp)
                    else:
                        a = synth(en, seg["text"], en_sid[accent], tmp)
                    a16 = resample_poly(a, SR, MELO_SR).astype(np.float32)
                    parts.append(a16)
                    parts.append(gap)
                audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
                if audio.size == 0:
                    raise RuntimeError("empty audio")
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  ERR {row.get('id')}: {type(e).__name__}: {e}")
                continue

            uid = f"melo_{row['id']}"
            wav_path = audio_dir / f"{uid}.wav"
            sf.write(str(wav_path), audio, SR)
            dur = len(audio) / SR
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
                "engine": "melo",
                "voice": f"JP+{accent}",
                "speaker": f"JP+{accent}",
                "gender": "NA",
                "duration_s": round(dur, 3),
                "sr": SR,
                "source": "synth/melo",
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
