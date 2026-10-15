#!/usr/bin/env python
"""Kaigi demo UI — bidirectional JP↔EN translation (auto-detect) + spoken output + minutes.

  pip install -e ".[demo]"
  python scripts/demo_app.py                 # loads models (GPU)
  python scripts/demo_app.py --mock          # UI logic only, no models
  python scripts/demo_app.py --share         # public Gradio link
  python scripts/demo_app.py --asr-model <your-finetuned-cs-asr>   # JA-side ASR = the code-switch fine-tune
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.live.config import LiveConfig  # noqa: E402

DIR_MAP = {"Auto (detect)": "auto", "JA → EN": "ja2en", "EN → JA": "en2ja"}


def to_float_mono(audio) -> tuple[np.ndarray, int]:
    sr, data = audio
    data = np.asarray(data)
    if data.dtype.kind in "iu":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="UI logic only, no models/deps")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"],
                    help="audio backend: torch (CUDA) | onnx (CPU-only, LFM model)")
    ap.add_argument("--onnx-ep", default="cpu", choices=["cpu", "dml", "cuda", "auto"],
                    help="onnx backend execution provider: dml -> Ryzen/Radeon iGPU (push TTS to RTF<1)")
    ap.add_argument("--tts-mode", default="sequential", choices=["sequential", "interleaved"],
                    help="torch TTS generation: interleaved lowers time-to-first-audio (speech-to-speech mode)")
    ap.add_argument("--asr-model", help="JA-side audio model / code-switch fine-tune (torch backend)")
    ap.add_argument("--en-audio-model", help="EN-side audio model (torch backend)")
    ap.add_argument("--ja-onnx", help="JA LFM2.5-Audio ONNX dir (onnx backend)")
    ap.add_argument("--en-onnx", help="EN LFM2.5-Audio ONNX dir (onnx backend)")
    ap.add_argument("--onnx-src", help="path to onnx-export/src (so `import liquidonnx` works)")
    ap.add_argument("--translator"); ap.add_argument("--voice", help='EN TTS prompt, e.g. "Perform TTS. Use the UK male voice."')
    ap.add_argument("--tt-adapter", help="translating-TTS LoRA on the JA audio model (translate+speak in one pass; "
                                         "replaces the separate text translator)")
    ap.add_argument("--no-text-model", action="store_true",
                    help="don't load LFM2.5-1.2B-JP (use with --tt-adapter; disables minutes)")
    ap.add_argument("--asr-gguf-url", help="route ASR to the GGUF liquid-audio server, e.g. http://localhost:8090")
    ap.add_argument("--asr-lora", help="GPU PyTorch ASR via this code-switch LoRA adapter (fast, contention-immune)")
    ap.add_argument("--share", action="store_true", help="create a public Gradio share link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    cfg = LiveConfig(device=args.device, audio_backend=args.backend,
                     onnx_ep=args.onnx_ep, tts_generation_mode=args.tts_mode)
    for attr, val in [("ja_audio_model_id", args.asr_model), ("en_audio_model_id", args.en_audio_model),
                      ("ja_onnx_dir", args.ja_onnx), ("en_onnx_dir", args.en_onnx), ("onnx_export_src", args.onnx_src),
                      ("translator_model_id", args.translator), ("tts_prompt_en", args.voice),
                      ("translating_tts_adapter", args.tt_adapter), ("asr_gguf_url", args.asr_gguf_url),
                      ("asr_lora", args.asr_lora)]:
        if val:
            setattr(cfg, attr, val)
    if args.no_text_model:
        cfg.load_text_model = False

    from csmeeting.app.assistant import Assistant
    print("Mock mode (no models)..." if args.mock else "Loading models (first run downloads weights)...")
    assistant = Assistant(cfg=cfg, mock=args.mock)

    import gradio as gr

    def on_audio(audio, direction_label):
        if audio is None:
            yield assistant.transcript(), "", None, ""
            return
        direction = DIR_MAP.get(direction_label, "auto")
        wav, sr = to_float_mono(audio)
        for _src_text, translation, chunk, src, tgt in assistant.process_stream(wav, sr, direction):
            yield assistant.transcript(), translation, chunk, f"**{src.upper()} → {tgt.upper()}**"

    def on_reset():
        assistant.reset()
        return "", "", None, ""

    with gr.Blocks(title="Kaigi") as demo:
        gr.Markdown("# 会議 Kaigi — on-device JP↔EN meeting assistant\n"
                    "**100% on-device · 0 bytes to the cloud.** Speak Japanese *or* English; the other side hears the "
                    "translation. Confidential minutes generated locally.")
        direction = gr.Radio(list(DIR_MAP), value="Auto (detect)", label="Direction")
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="numpy", label="🎙 Speak (record, then stop)")
            out_audio = gr.Audio(label="🔊 Translation (spoken)", autoplay=True, streaming=True)
        detected = gr.Markdown()
        with gr.Row():
            transcript_box = gr.Textbox(label="Transcript (source language, code-switch aware)", lines=10)
            translation_box = gr.Textbox(label="Translation (latest)", lines=10)
        with gr.Row():
            minutes_btn = gr.Button("📝 Generate minutes", variant="primary")
            reset_btn = gr.Button("🗑 Reset")
        minutes_box = gr.Markdown()

        mic.stop_recording(on_audio, inputs=[mic, direction],
                           outputs=[transcript_box, translation_box, out_audio, detected])
        minutes_btn.click(assistant.minutes, outputs=minutes_box)
        reset_btn.click(on_reset, outputs=[transcript_box, translation_box, out_audio, minutes_box])

    demo.launch(share=args.share, server_port=args.port, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
