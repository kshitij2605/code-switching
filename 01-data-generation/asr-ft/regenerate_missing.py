"""
Regenerates specific missing CS samples with retry logic.
Run from /data/lfm2.5/run-a with .venv active.
"""

import asyncio, edge_tts, json, io, subprocess, wave, random, time
from pathlib import Path
import imageio_ffmpeg

_FFMPEG    = imageio_ffmpeg.get_ffmpeg_exe()
_RATE      = 24000
_CHANNELS  = 1
_SAMPWIDTH = 2
PROXY      = None  # set to "http://host:port/" if behind a proxy
SILENCE_MS = 180
RETRIES    = 5
RETRY_WAIT = 3   # seconds between retries

AUDIO_OUT = Path("data/code_switching/audio")
META_OUT  = Path("data/code_switching/metadata")

# ── Audio helpers ─────────────────────────────────────────────────────────────

def _mp3_to_pcm(mp3: bytes) -> bytes:
    r = subprocess.run(
        [_FFMPEG, "-v", "error", "-f", "mp3", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(_RATE), "-ac", str(_CHANNELS), "pipe:1"],
        input=mp3, capture_output=True
    )
    return r.stdout

def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS); wf.setsampwidth(_SAMPWIDTH)
        wf.setframerate(_RATE);      wf.writeframes(pcm)
    return buf.getvalue()

def _silence(ms: int) -> bytes:
    return b"\x00" * (_RATE * ms // 1000 * _SAMPWIDTH * _CHANNELS)

async def tts_bytes(text: str, voice: str) -> bytes:
    for attempt in range(1, RETRIES + 1):
        try:
            comm = edge_tts.Communicate(text, voice, proxy=PROXY)
            buf = b""
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf += chunk["data"]
            if buf:
                return buf
            raise RuntimeError("Empty audio response")
        except Exception as e:
            print(f"    attempt {attempt}/{RETRIES} failed: {e}")
            if attempt < RETRIES:
                await asyncio.sleep(RETRY_WAIT * attempt)
    raise RuntimeError(f"All {RETRIES} attempts failed for: {text[:40]}")

# ── Rebuild segments for specific sample indices ──────────────────────────────
# We re-use the same seed and logic from generate_cs_data.py so segments match.

random.seed(42)

# Import vocabulary and topic functions from generate_cs_data
import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_cs_data import TOPICS, pick_segments, build_transcriptions

VOICE_JA = "ja-JP-NanamiNeural"
VOICE_EN = "en-US-JennyNeural"

def get_segments_for_index(target_idx: int):
    """Replay the RNG to get segments for sample index target_idx (1-based)."""
    random.seed(42)
    segs = None
    for i in range(1, target_idx + 1):
        segs = pick_segments()
    return segs

async def regenerate_sample(idx: int) -> dict:
    print(f"\n[Regenerating cs_{idx:03d}]")
    segs = get_segments_for_index(idx)
    full, all_ja, all_en, jp_parts, en_parts = build_transcriptions(segs)

    pcm_all = b""
    for seg in segs:
        voice = VOICE_JA if seg["lang"] == "ja" else VOICE_EN
        print(f"  TTS: [{seg['lang']}] {seg['text']}")
        mp3 = await tts_bytes(seg["text"], voice)
        pcm_all += _mp3_to_pcm(mp3) + _silence(SILENCE_MS)

    wav = _pcm_to_wav(pcm_all)
    filepath = AUDIO_OUT / f"cs_{idx:03d}.wav"
    filepath.write_bytes(wav)
    duration = round(len(pcm_all) / (_RATE * _SAMPWIDTH * _CHANNELS), 2)

    print(f"  ✓ Saved {filepath} ({len(wav)//1024} KB, {duration}s)")
    return {
        "id":                f"cs_{idx:03d}",
        "audio_file":        str(filepath),
        "full_transcription": full,
        "all_japanese":       all_ja,
        "all_english":        all_en,
        "japanese_parts":     jp_parts,
        "english_parts":      en_parts,
        "duration_seconds":   duration,
        "num_segments":       len(segs),
        "segments":           segs,
    }

async def main():
    missing_indices = [115, 116, 117, 118]
    print(f"Regenerating {len(missing_indices)} missing samples: {missing_indices}")

    # Load existing metadata
    meta_path = META_OUT / "transcriptions.json"
    existing = json.loads(meta_path.read_text(encoding="utf-8"))
    existing_by_id = {s["id"]: s for s in existing}

    new_samples = []
    for idx in missing_indices:
        try:
            sample = await regenerate_sample(idx)
            existing_by_id[sample["id"]] = sample
            new_samples.append(sample["id"])
        except Exception as e:
            print(f"  FAILED cs_{idx:03d}: {e}")

    # Sort and rewrite full metadata
    all_samples = sorted(existing_by_id.values(), key=lambda s: s["id"])
    meta_path.write_text(json.dumps(all_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. Regenerated: {new_samples}")
    print(f"Total CS samples in metadata: {len(all_samples)}")
    print(f"Audio files on disk: {len(list(AUDIO_OUT.glob('cs_*.wav')))}")

if __name__ == "__main__":
    asyncio.run(main())
