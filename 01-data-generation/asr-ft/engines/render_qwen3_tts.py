#!/usr/bin/env python3
"""Render code-switched transcripts with Qwen3-TTS (Apache-2.0; Japanese
supported). This is a commercial-clean engine for the Phase 3 data engine.

Input: the renderers' transcript JSONL ({id, transcript, segments, ...}).
Each utterance is synthesised in ONE voice with language=Japanese (the matrix
language), so a switch point has no splice seam and no voice change (either would
be a shortcut cue for the recogniser).

Modes:
  preset  Qwen/Qwen3-TTS-12Hz-*-CustomVoice built-in speakers (--speakers)
  clone   Qwen/Qwen3-TTS-12Hz-*-Base with reference clips from --voice-prompts,
          a JSONL of {audio_path, text, speaker?, license_source?}

Licences: the engine is Apache-2.0. Cloned audio also inherits the prompt
source's tags, plus "voiceprompt/cloned" (use="check"): a CC BY recording
licence does not equal the speaker's consent to have their voice cloned, so
product data should use the consented voice bank.

Env: /data/code-switching/envs/qwen3-tts (envs/qwen3-tts.freeze.txt). Output: 16 kHz mono
WAV + manifest.jsonl rows {id, audio_path, transcript, engine, voice, ...}.

Example:
  CUDA_VISIBLE_DEVICES=0 /data/code-switching/envs/qwen3-tts/bin/python render_qwen3_tts.py \
      --in transcripts.jsonl --out data/synth/qwen3tts --mode preset --speakers Vivian Ryan
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.licenses import tag_row  # noqa: E402

SR = 16000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["preset", "clone"], default="preset")
    ap.add_argument("--model-id", default=None,
                    help="default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (preset) or -1.7B-Base (clone)")
    ap.add_argument("--speakers", nargs="*", default=[], help="preset speaker names (rotated)")
    ap.add_argument("--voice-prompts", default=None, help="clone: JSONL of reference clips")
    ap.add_argument("--language", default="Japanese")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from qwen_tts import Qwen3TTSModel

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    mid = args.model_id or ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" if args.mode == "preset"
                            else "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    if args.mode == "preset" and not args.speakers:
        raise SystemExit("--mode preset needs --speakers")
    prompts = []
    if args.mode == "clone":
        prompts = [json.loads(l) for l in open(args.voice_prompts, encoding="utf-8") if l.strip()]
        if not prompts:
            raise SystemExit("--voice-prompts is empty")
    model = Qwen3TTSModel.from_pretrained(mid, device_map="cuda:0", dtype=torch.bfloat16,
                                          attn_implementation="sdpa")
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    written, errors = [], 0
    for i, row in enumerate(rows):
        try:
            if args.mode == "preset":
                spk = args.speakers[i % len(args.speakers)]
                wavs, sr = model.generate_custom_voice(text=row["transcript"], language=args.language,
                                                       speaker=spk)
                voice, derived = spk, []
            else:
                p = rng.choice(prompts)
                wavs, sr = model.generate_voice_clone(text=row["transcript"], language=args.language,
                                                      ref_audio=p["audio_path"], ref_text=p["text"])
                voice = p.get("speaker") or Path(p["audio_path"]).stem
                derived = [s for s in (p.get("license_source") or "").split("+") if s] + ["voiceprompt/cloned"]
            a = np.asarray(wavs[0], dtype=np.float32)
            if sr != SR:
                a = resample_poly(a, SR, sr).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  ERR {row.get('id')}: {type(e).__name__}: {e}")
            continue
        uid = f"qwen3tts_{row['id']}"
        wav = out / "audio" / f"{uid}.wav"
        sf.write(wav, a, SR)
        rec = {"id": uid, "audio_path": str(wav.resolve()), "transcript": row["transcript"],
               "segments": row.get("segments", []), "engine": "qwen3_tts", "voice": voice,
               "speaker": voice, "mode": args.mode, "model_id": mid, "language": args.language,
               "duration_s": round(len(a) / SR, 3)}
        written.append(tag_row(rec, "tts/qwen3-tts", derived))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in written:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(r["duration_s"] for r in written)
    print(f"qwen3-tts ({args.mode}): {len(written)} clips, {total / 60:.1f} min, {errors} errors "
          f"-> {out / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
