"""
Dry run: generate 3 samples each of CS / EN / JA and print results.
Writes audio to data/dry_run/ and prints the JSON metadata to stdout.
Run from /data/lfm2.5/run-a with .venv active.
"""

import asyncio, edge_tts, json, io, subprocess, wave, random
from pathlib import Path
import imageio_ffmpeg

# ── ffmpeg setup ──────────────────────────────────────────────────────────────
_FFMPEG    = imageio_ffmpeg.get_ffmpeg_exe()
_RATE      = 24000
_CHANNELS  = 1
_SAMPWIDTH = 2
PROXY      = None  # set to "http://host:port/" if behind a proxy

OUT = Path("data/dry_run")
OUT.mkdir(parents=True, exist_ok=True)

def _mp3_to_pcm(mp3: bytes) -> bytes:
    r = subprocess.run(
        [_FFMPEG, "-v", "error", "-f", "mp3", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(_RATE), "-ac", str(_CHANNELS), "pipe:1"],
        input=mp3, capture_output=True
    )
    if not r.stdout:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()}")
    return r.stdout

def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS); wf.setsampwidth(_SAMPWIDTH)
        wf.setframerate(_RATE);      wf.writeframes(pcm)
    return buf.getvalue()

def _silence(ms: int) -> bytes:
    return b"\x00" * (_RATE * ms // 1000 * _SAMPWIDTH * _CHANNELS)

def _duration(pcm: bytes) -> float:
    return round(len(pcm) / (_RATE * _SAMPWIDTH * _CHANNELS), 2)

async def tts(text: str, voice: str) -> bytes:
    comm = edge_tts.Communicate(text, voice, proxy=PROXY)
    buf = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    return buf


# ── Code-switching dry run ────────────────────────────────────────────────────
CS_SAMPLES = [
    {
        "id": "dry_cs_001",
        "segments": [
            {"lang": "ja", "text": "明日の",        "en_equiv": "Tomorrow's"},
            {"lang": "en", "text": "meeting",        "ja_equiv": "ミーティング"},
            {"lang": "ja", "text": "は何時から",     "en_equiv": "starts at what time"},
            {"lang": "en", "text": "please check",   "ja_equiv": "確認してください"},
            {"lang": "ja", "text": "ね",             "en_equiv": "okay?"},
        ],
    },
    {
        "id": "dry_cs_002",
        "segments": [
            {"lang": "ja", "text": "このbugは",     "en_equiv": "This bug"},
            {"lang": "en", "text": "really annoying", "ja_equiv": "本当に厄介"},
            {"lang": "ja", "text": "だよ",           "en_equiv": "you know"},
            {"lang": "en", "text": "let's fix it",   "ja_equiv": "直そう"},
            {"lang": "ja", "text": "今すぐ",         "en_equiv": "right now"},
        ],
    },
    {
        "id": "dry_cs_003",
        "segments": [
            {"lang": "ja", "text": "ちょっと",       "en_equiv": "just a little"},
            {"lang": "en", "text": "tired",           "ja_equiv": "疲れた"},
            {"lang": "ja", "text": "だけど",          "en_equiv": "but"},
            {"lang": "en", "text": "let's go",        "ja_equiv": "行こう"},
            {"lang": "ja", "text": "頑張ろう",        "en_equiv": "let's do our best"},
        ],
    },
]

async def run_cs():
    print("\n" + "="*60)
    print("CODE-SWITCHING DRY RUN (3 samples)")
    print("="*60)
    results = []
    for s in CS_SAMPLES:
        print(f"\n[{s['id']}]")
        pcm_all = b""
        for seg in s["segments"]:
            voice = "ja-JP-NanamiNeural" if seg["lang"] == "ja" else "en-US-JennyNeural"
            mp3 = await tts(seg["text"], voice)
            pcm_all += _mp3_to_pcm(mp3) + _silence(180)

        wav = _pcm_to_wav(pcm_all)
        path = OUT / f"{s['id']}.wav"
        path.write_bytes(wav)

        full   = "".join(seg["text"] for seg in s["segments"])
        all_ja = "".join(seg["ja_equiv"] if seg["lang"]=="en" else seg["text"] for seg in s["segments"])
        all_en = "".join(seg["en_equiv"] if seg["lang"]=="ja" else seg["text"] for seg in s["segments"])

        meta = {
            "id":                s["id"],
            "audio_file":        str(path),
            "file_size_bytes":   len(wav),
            "duration_seconds":  _duration(pcm_all),
            "full_transcription": full,
            "all_japanese":       all_ja,
            "all_english":        all_en,
        }
        results.append(meta)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    return results


# ── English-only dry run ──────────────────────────────────────────────────────
EN_SAMPLES = [
    {"id": "dry_en_001", "en": "The deadline for this project is next Friday.", "ja": "このプロジェクトの締め切りは来週の金曜日です。"},
    {"id": "dry_en_002", "en": "The bug in the login module has been fixed.",   "ja": "ログインモジュールのバグが修正されました。"},
    {"id": "dry_en_003", "en": "Let's grab coffee before the meeting starts.",  "ja": "会議が始まる前にコーヒーを飲みに行きましょう。"},
]

async def run_en():
    print("\n" + "="*60)
    print("ENGLISH-ONLY DRY RUN (3 samples)")
    print("="*60)
    results = []
    for s in EN_SAMPLES:
        print(f"\n[{s['id']}]")
        mp3 = await tts(s["en"], "en-US-JennyNeural")
        pcm = _mp3_to_pcm(mp3)
        wav = _pcm_to_wav(pcm)
        path = OUT / f"{s['id']}.wav"
        path.write_bytes(wav)

        meta = {
            "id":                    s["id"],
            "audio_file":            str(path),
            "file_size_bytes":       len(wav),
            "duration_seconds":      _duration(pcm),
            "english_transcription": s["en"],
            "japanese_translation":  s["ja"],
        }
        results.append(meta)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    return results


# ── Japanese-only dry run ─────────────────────────────────────────────────────
JA_SAMPLES = [
    {"id": "dry_ja_001", "ja": "明日の会議の資料を準備しておいてください。",  "en": "Please prepare the materials for tomorrow's meeting."},
    {"id": "dry_ja_002", "ja": "ログインモジュールのバグが修正されました。",  "en": "The bug in the login module has been fixed."},
    {"id": "dry_ja_003", "ja": "定期的な運動は健康的な体重の維持に役立ちます。", "en": "Regular exercise helps maintain a healthy weight."},
]

async def run_ja():
    print("\n" + "="*60)
    print("JAPANESE-ONLY DRY RUN (3 samples)")
    print("="*60)
    results = []
    for s in JA_SAMPLES:
        print(f"\n[{s['id']}]")
        mp3 = await tts(s["ja"], "ja-JP-NanamiNeural")
        pcm = _mp3_to_pcm(mp3)
        wav = _pcm_to_wav(pcm)
        path = OUT / f"{s['id']}.wav"
        path.write_bytes(wav)

        meta = {
            "id":                     s["id"],
            "audio_file":             str(path),
            "file_size_bytes":        len(wav),
            "duration_seconds":       _duration(pcm),
            "japanese_transcription": s["ja"],
            "english_translation":    s["en"],
        }
        results.append(meta)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"ffmpeg binary: {_FFMPEG}")
    print(f"Output dir:    {OUT.resolve()}")

    cs = await run_cs()
    en = await run_en()
    ja = await run_ja()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_files = list(OUT.glob("*.wav"))
    total_size = sum(f.stat().st_size for f in all_files)
    print(f"  WAV files written : {len(all_files)}")
    print(f"  Total size        : {total_size/1024:.1f} KB")
    for f in sorted(all_files):
        print(f"    {f.name}  {f.stat().st_size/1024:.1f} KB")

if __name__ == "__main__":
    asyncio.run(main())
