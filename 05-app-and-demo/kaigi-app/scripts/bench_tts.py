#!/usr/bin/env python
"""Benchmark TTS real-time factor (RTF) + time-to-first-audio (TTFA) for the spoken path.

Use this on the target hardware (the H100 host *and* the Ryzen laptop) to validate the two latency improvements:
  1. ONNX execution provider — CPU vs the Ryzen/Radeon iGPU (DirectML).  --backend onnx --onnx-ep {cpu,dml}
  2. torch generation mode — sequential vs interleaved (time-to-first-audio). --backend torch --tts-mode {...}

RTF = wall_time_to_generate / audio_seconds_produced  (RTF < 1 == real-time, gap-free playback).
TTFA = wall time until the first non-empty streamed chunk.

Examples
  # torch / GPU, compare generation modes:
  python scripts/bench_tts.py --backend torch --device cuda --lang en --tts-mode sequential
  python scripts/bench_tts.py --backend torch --device cuda --lang en --tts-mode interleaved

  # ONNX / CPU vs iGPU on the laptop (run each, compare RTF):
  python scripts/bench_tts.py --backend onnx --device cpu --onnx-ep cpu \
      --en-onnx <LFM2.5-Audio-1.5B-ONNX> --onnx-src <onnx-export/src> --lang en
  python scripts/bench_tts.py --backend onnx --device cpu --onnx-ep dml \
      --en-onnx <LFM2.5-Audio-1.5B-ONNX> --onnx-src <onnx-export/src> --lang en

  # wiring test, no models:
  python scripts/bench_tts.py --mock
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.live.config import LiveConfig  # noqa: E402

_DEFAULT_TEXT = {
    "en": "Let's review the agenda for today's meeting and confirm the action items before we close.",
    "ja": "本日の会議のアジェンダを確認して、終了する前にアクションアイテムを共有しましょう。",
}


def _build_engine(cfg: LiveConfig, mock: bool):
    if mock:
        from csmeeting.live.engine import MockAudioEngine
        return MockAudioEngine(cfg)
    if cfg.audio_backend == "onnx":
        from csmeeting.live.onnx_engine import OnnxAudioEngine
        return OnnxAudioEngine(cfg)
    from csmeeting.live.engine import AudioEngine
    return AudioEngine(cfg)


def _measure(engine, text: str, lang: str, sr_out: int) -> tuple[float, float, float]:
    """Return (ttfa_s, wall_s, audio_s) for one streamed synthesis."""
    t0 = time.perf_counter()
    ttfa = float("nan")
    n_samples = 0
    for chunk in engine.synthesize_stream(text, lang):
        if chunk is not None and len(chunk) > 0:
            if n_samples == 0:
                ttfa = time.perf_counter() - t0
            n_samples += len(chunk)
    wall = time.perf_counter() - t0
    audio_s = n_samples / sr_out if sr_out else 0.0
    return ttfa, wall, audio_s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="no models — wiring/measurement test")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--onnx-ep", default="cpu", choices=["cpu", "dml", "cuda", "auto"])
    ap.add_argument("--tts-mode", default="sequential", choices=["sequential", "interleaved"])
    ap.add_argument("--lang", default="en", choices=["en", "ja"])
    ap.add_argument("--text", help="override the text to synthesize")
    ap.add_argument("--runs", type=int, default=5, help="timed runs (after warmup)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed warmup runs (cold decode is much slower)")
    # model wiring (mirror demo_app.py)
    ap.add_argument("--asr-model"); ap.add_argument("--en-audio-model")
    ap.add_argument("--ja-onnx"); ap.add_argument("--en-onnx"); ap.add_argument("--onnx-src")
    ap.add_argument("--voice", help='EN TTS prompt, e.g. "Perform TTS. Use the UK male voice."')
    args = ap.parse_args()

    cfg = LiveConfig(device=args.device, audio_backend=args.backend,
                     onnx_ep=args.onnx_ep, tts_generation_mode=args.tts_mode)
    for attr, val in [("ja_audio_model_id", args.asr_model), ("en_audio_model_id", args.en_audio_model),
                      ("ja_onnx_dir", args.ja_onnx), ("en_onnx_dir", args.en_onnx),
                      ("onnx_export_src", args.onnx_src), ("tts_prompt_en", args.voice)]:
        if val:
            setattr(cfg, attr, val)

    text = args.text or _DEFAULT_TEXT[args.lang]
    print(f"backend={args.backend} device={args.device} onnx_ep={args.onnx_ep} tts_mode={args.tts_mode} "
          f"lang={args.lang}")
    print("loading engine..." if not args.mock else "mock engine (no models)...")
    engine = _build_engine(cfg, args.mock)

    for _ in range(max(0, args.warmup)):
        _measure(engine, text, args.lang, cfg.sample_rate_out)

    ttfas, rtfs, walls, audios = [], [], [], []
    for i in range(args.runs):
        ttfa, wall, audio_s = _measure(engine, text, args.lang, cfg.sample_rate_out)
        rtf = wall / audio_s if audio_s > 0 else float("inf")
        ttfas.append(ttfa); rtfs.append(rtf); walls.append(wall); audios.append(audio_s)
        print(f"  run {i + 1}: ttfa={ttfa:.3f}s  wall={wall:.3f}s  audio={audio_s:.3f}s  RTF={rtf:.3f}")

    def _med(xs):
        return statistics.median(xs) if xs else float("nan")

    print("-" * 60)
    print(f"median over {args.runs} runs:  TTFA={_med(ttfas):.3f}s   RTF={_med(rtfs):.3f}   "
          f"({'REAL-TIME ✅' if _med(rtfs) < 1 else 'NOT real-time ❌ (RTF>=1)'})")
    print(f"audio duration={_med(audios):.3f}s  wall={_med(walls):.3f}s")


if __name__ == "__main__":
    main()
