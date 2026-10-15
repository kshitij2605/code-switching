#!/usr/bin/env python3
"""Engine #1 — render CS transcripts with edge-tts multilingual voices.

edge-tts exposes only 2 Japanese voices, but 5 *multilingual* voices that speak
both Japanese and English in a single voice. We render each whole code-switch
utterance in ONE multilingual speaker, so the JA->EN switch happens inside a
consistent voice (no two-speaker splice seam). Speakers are rotated across the
corpus for multi-speaker diversity, with light rate/pitch jitter for acoustic
variety.

Input : a transcript JSONL from gen_cs_transcripts.py
Output: <out>/audio/<id>.wav (16 kHz mono) + <out>/manifest.jsonl in the shared
        schema consumed by 02-preprocessing and the eval scorer.

Run (on mactrn01, in the venv):
  .venv/bin/python scripts/01-data-generation/asr-ft/engines/render_edge_tts.py \
      --in data/transcripts/cs_transcripts.jsonl \
      --out data/synth/edge_tts --limit 100
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import subprocess
import wave
from pathlib import Path

import edge_tts
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# edge-tts uses a WebSocket that does NOT auto-honor HTTP(S)_PROXY env vars, so on
# a proxied host (e.g. mactrn01) we must pass the proxy explicitly or every
# connection times out. Fall back to None (direct) when no proxy is set.
PROXY = (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
         or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY") or None)
SR = 16000            # target sample rate (matches eval/training)
CHANNELS = 1
SAMPWIDTH = 2         # 16-bit PCM

# 5 multilingual speakers (JA+EN in one voice), gender-balanced.
VOICES = [
    ("en-US-AvaMultilingualNeural", "F"),
    ("en-US-EmmaMultilingualNeural", "F"),
    ("en-US-AndrewMultilingualNeural", "M"),
    ("en-US-BrianMultilingualNeural", "M"),
    ("en-AU-WilliamMultilingualNeural", "M"),
]

# small deterministic prosody jitter per utterance (index -> rate/pitch)
RATES = ["+0%", "-8%", "+6%", "-4%", "+10%"]
PITCHES = ["+0Hz", "-10Hz", "+8Hz", "+15Hz", "-6Hz"]


def _mp3_to_pcm16k(mp3: bytes) -> bytes:
    """Decode MP3 bytes -> raw signed-16-bit 16 kHz mono PCM via ffmpeg."""
    p = subprocess.run(
        [FFMPEG, "-v", "error", "-f", "mp3", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(SR), "-ac", str(CHANNELS), "pipe:1"],
        input=mp3, capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {p.stderr.decode()[:200]}")
    return p.stdout


def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(SR)
        wf.writeframes(pcm)
    return buf.getvalue()


async def _tts(text: str, voice: str, rate: str, pitch: str) -> bytes:
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, proxy=PROXY)
    buf = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    return buf


async def render_one(row: dict, idx: int, audio_dir: Path) -> dict:
    voice, gender = VOICES[idx % len(VOICES)]
    rate = RATES[idx % len(RATES)]
    pitch = PITCHES[idx % len(PITCHES)]

    mp3 = await _tts(row["transcript"], voice, rate, pitch)
    if not mp3:
        raise RuntimeError("empty audio from edge-tts")
    pcm = _mp3_to_pcm16k(mp3)
    wav = _pcm_to_wav(pcm)

    uid = f"edge_{row['id']}"
    wav_path = audio_dir / f"{uid}.wav"
    wav_path.write_bytes(wav)
    dur = len(pcm) / (SR * SAMPWIDTH * CHANNELS)

    return {
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
        "engine": "edge-tts",
        "voice": voice,
        "speaker": voice,          # one multilingual voice == one speaker
        "gender": gender,
        "rate": rate,
        "pitch": pitch,
        "duration_s": round(dur, 3),
        "sr": SR,
        "source": "synth/edge_tts",
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/synth/edge_tts")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"

    print(f"edge-tts: rendering {len(rows)} utterances across {len(VOICES)} multilingual voices")
    written, errors, total_dur = [], 0, 0.0
    for i, row in enumerate(rows):
        try:
            m = await render_one(row, i, audio_dir)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  ERR {row.get('id')}: {type(e).__name__}: {e}")
            continue
        written.append(m)
        total_dur += m["duration_s"]
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} ...")

    with manifest.open("w", encoding="utf-8") as f:
        for m in written:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    from collections import Counter
    by_voice = Counter(m["voice"].split("-")[-1] for m in written)
    print(f"\ndone: {len(written)} ok, {errors} err")
    print(f"  total audio : {total_dur / 60:.1f} min ({total_dur / max(len(written),1):.1f} s/clip)")
    print(f"  per speaker : {dict(by_voice)}")
    print(f"  manifest    : {manifest}")


if __name__ == "__main__":
    asyncio.run(main())
