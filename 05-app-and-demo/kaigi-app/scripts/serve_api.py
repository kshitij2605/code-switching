#!/usr/bin/env python
"""FastAPI bridge for the Kaigi web front end (web/).

Exposes the on-device `Assistant` over HTTP + WebSocket so the React UI can drive
the real cascade (ASR → MT → TTS → minutes). The browser sends 16-bit PCM WAV
(decoded here with the stdlib `wave` module — no extra audio deps), and the
server streams back transcript / translation / spoken-audio events.

  pip install -e ".[demo]" fastapi "uvicorn[standard]"
  python scripts/serve_api.py --mock          # no models — wiring only
  python scripts/serve_api.py                 # loads models (GPU)
  python scripts/serve_api.py --asr-model <your-finetuned-cs-asr>

Then run the front end:  cd web && npm run dev   (proxies /api and /ws here)
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time
import wave
from pathlib import Path

import numpy as np
# Imported at module level so the string annotations created by
# `from __future__ import annotations` (e.g. `ws: WebSocket`) resolve against
# module globals — otherwise FastAPI mis-reads the WebSocket param as a query field.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.live.config import LiveConfig  # noqa: E402

DIR_MAP = {"auto": "auto", "ja2en": "ja2en", "en2ja": "en2ja"}


def decode_wav(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode a 16-bit PCM WAV (what the browser sends) to float32 mono."""
    with wave.open(io.BytesIO(raw), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def pcm16_b64(chunk: np.ndarray) -> str:
    """float32 audio → base64 little-endian int16 (for the browser to play)."""
    clipped = np.clip(chunk, -1.0, 1.0)
    return base64.b64encode((clipped * 32767.0).astype("<i2").tobytes()).decode("ascii")


def build_app(assistant, cfg: LiveConfig, mock: bool, web_dist: str | None = None):
    app = FastAPI(title="Kaigi API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    def status() -> dict:
        return {"mode": "live", "device": "mock" if mock else cfg.device, "ready": True}

    @app.get("/api/transcript")
    def transcript() -> dict:
        return {"text": assistant.transcript()}

    @app.post("/api/minutes")
    def minutes() -> dict:
        return {"markdown": assistant.minutes()}

    @app.post("/api/reset")
    def reset() -> dict:
        assistant.reset()
        return {"ok": True}

    @app.websocket("/ws/translate")
    async def translate(ws: WebSocket) -> None:
        await ws.accept()
        try:
            meta = await ws.receive_json()
            direction = DIR_MAP.get(meta.get("direction", "auto"), "auto")
            raw = await ws.receive_bytes()
            wav, sr = decode_wav(raw)

            t0 = time.perf_counter()
            sent_text = False
            for src_text, translation, chunk, src, tgt in assistant.process_stream(wav, sr, direction):
                if not sent_text:
                    await ws.send_json({"type": "detected", "src": src, "tgt": tgt})
                    await ws.send_json({"type": "transcript", "text": src_text})
                    await ws.send_json({"type": "translation", "text": translation})
                    sent_text = True
                if chunk is not None:
                    out_sr, audio = chunk
                    await ws.send_json({"type": "audio", "sr": out_sr, "data": pcm16_b64(audio)})
            await ws.send_json({"type": "done", "latencyMs": round((time.perf_counter() - t0) * 1000)})
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # surface errors to the client instead of dropping
            await ws.send_json({"type": "error", "message": str(exc)})
        finally:
            await ws.close()

    # Serve the built React app from the same origin (so its relative /api and /ws URLs just work).
    # Mounted last so the API/WS routes above take precedence over the static catch-all.
    if web_dist:
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
        print(f"Serving web UI from {web_dist}")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="UI logic only, no models/deps")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--asr-model", help="JA-side audio model / code-switch fine-tune")
    ap.add_argument("--en-audio-model", help="EN-side audio model")
    ap.add_argument("--ja-onnx")
    ap.add_argument("--en-onnx")
    ap.add_argument("--onnx-src")
    ap.add_argument("--translator")
    ap.add_argument("--voice", help='EN TTS prompt, e.g. "Perform TTS. Use the UK male voice."')
    ap.add_argument("--tt-adapter", help="translating-TTS LoRA on the JA audio model (translate+speak in one pass; "
                                         "replaces the separate text translator)")
    ap.add_argument("--no-text-model", action="store_true",
                    help="don't load LFM2.5-1.2B-JP (use with --tt-adapter; disables minutes)")
    ap.add_argument("--asr-gguf-url", help="route ASR to the GGUF liquid-audio server, e.g. http://localhost:8090 "
                                           "(start it with scripts/gguf_asr_server.sh)")
    ap.add_argument("--asr-lora", help="GPU PyTorch ASR via this code-switch LoRA adapter (base ja model + adapter); "
                                       "fast + contention-immune. Preferred on a GPU box.")
    ap.add_argument("--minutes-lora", help="run minutes on the audio backbone via this LoRA (2nd adapter); "
                                           "pair with --no-text-model to drop LFM2.5-1.2B-JP entirely.")
    ap.add_argument("--web-dist", help="serve the built React app (web/dist) from this server, same origin")
    ap.add_argument("--ssl-certfile", help="TLS cert (enables HTTPS/WSS — needed for mic access over the network)")
    ap.add_argument("--ssl-keyfile", help="TLS private key")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    cfg = LiveConfig(device=args.device, audio_backend=args.backend)
    for attr, val in [
        ("ja_audio_model_id", args.asr_model),
        ("en_audio_model_id", args.en_audio_model),
        ("ja_onnx_dir", args.ja_onnx),
        ("en_onnx_dir", args.en_onnx),
        ("onnx_export_src", args.onnx_src),
        ("translator_model_id", args.translator),
        ("tts_prompt_en", args.voice),
        ("translating_tts_adapter", args.tt_adapter),
        ("asr_gguf_url", args.asr_gguf_url),
        ("asr_lora", args.asr_lora),
        ("minutes_lora", args.minutes_lora),
    ]:
        if val:
            setattr(cfg, attr, val)
    if args.no_text_model:
        cfg.load_text_model = False

    from csmeeting.app.assistant import Assistant

    print("Mock mode (no models)..." if args.mock else "Loading models (first run downloads weights)...")
    assistant = Assistant(cfg=cfg, mock=args.mock)

    import uvicorn

    app = build_app(assistant, cfg, args.mock, web_dist=args.web_dist)
    scheme = "https" if args.ssl_certfile else "http"
    print(f"Kaigi API on {scheme}://localhost:{args.port}")
    # wsproto is the most robust WebSocket backend across uvicorn/websockets versions;
    # fall back to auto-detection if it is not installed.
    try:
        import wsproto  # noqa: F401

        ws_impl = "wsproto"
    except ImportError:
        ws_impl = "auto"
    uvicorn.run(app, host="0.0.0.0", port=args.port, ws=ws_impl,
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)


if __name__ == "__main__":
    main()
