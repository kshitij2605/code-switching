#!/usr/bin/env python
"""Live JA->EN translation cascade (CLI): VAD -> ASR (LFM2.5-Audio) -> MT -> streaming TTS.

This CLI is fixed JA->EN; the **bidirectional** (auto-detect) path is the demo app (scripts/demo_app.py).

  python scripts/live_translate.py --mock            # offline wiring test (no models/deps)
  python scripts/live_translate.py --wav meeting.wav # translate a Japanese file
  python scripts/live_translate.py --mic             # microphone (needs sounddevice)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.live.config import LiveConfig  # noqa: E402
from csmeeting.live.pipeline import LiveTranslator, play  # noqa: E402


def build(cfg: LiveConfig, mock: bool) -> LiveTranslator:
    if mock:
        from csmeeting.live.engine import MockAudioEngine
        from csmeeting.live.translate import IdentityTranslator
        return LiveTranslator(cfg, MockAudioEngine(cfg), IdentityTranslator())
    from csmeeting.live.engine import AudioEngine
    from csmeeting.live.translate import LFMTranslator
    return LiveTranslator(cfg, AudioEngine(cfg), LFMTranslator(cfg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mic", action="store_true", help="microphone input")
    src.add_argument("--wav", help="translate a Japanese wav file")
    ap.add_argument("--mock", action="store_true", help="run the cascade wiring with no models/deps")
    ap.add_argument("--asr-model", help="JA-side audio model (your code-switch ASR fine-tune)")
    ap.add_argument("--tts-model", help="EN-side audio model (default base English)")
    ap.add_argument("--translator")
    ap.add_argument("--adapter", help="LoRA adapter for the MT stage")
    ap.add_argument("--voice", help='EN TTS prompt, e.g. "Perform TTS. Use the UK male voice."')
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", default="data/live_out", help="save EN wavs here")
    ap.add_argument("--no-play", action="store_true", help="don't play audio in mic mode")
    args = ap.parse_args()

    cfg = LiveConfig(device=args.device)
    for attr, val in [("ja_audio_model_id", args.asr_model), ("en_audio_model_id", args.tts_model),
                      ("translator_model_id", args.translator), ("translator_adapter", args.adapter),
                      ("tts_prompt_en", args.voice)]:
        if val:
            setattr(cfg, attr, val)

    lt = build(cfg, args.mock)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def on_utt(utt) -> None:
        print(f"\n[{utt.index}] JA: {utt.ja_text}")
        print(f"     EN: {utt.en_text}   ({utt.audio.size} samples @ {utt.sr} Hz)")
        if utt.audio.size:
            try:
                import soundfile as sf
                sf.write(str(out_dir / f"utt_{utt.index:03d}_en.wav"), utt.audio, utt.sr)
            except Exception as e:  # noqa: BLE001
                print(f"     (audio save skipped: {e})")
        if args.mic and not args.no_play and not args.mock:
            try:
                play(utt)
            except Exception as e:  # noqa: BLE001
                print(f"     (playback skipped: {e})")

    if args.mic:
        lt.run_mic(on_utt)
    elif args.wav:
        lt.run_file(args.wav, on_utt)
    else:
        print("No --wav/--mic given -> running ONE utterance through the cascade.")
        if not args.mock:
            print("Tip: add --mock to exercise the wiring without models.")
        import numpy as np
        on_utt(lt.process(np.zeros(cfg.vad_sample_rate, dtype="float32"), cfg.vad_sample_rate, 0))


if __name__ == "__main__":
    main()
