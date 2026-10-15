# 05 · App & Demo

## kaigi-app/ — 会議 Kaigi meeting assistant (tts-ft)
A privacy-first, fully on-device JP↔EN meeting assistant: **one `LFM2.5-Audio-1.5B-JP` base +
3 swappable LoRA adapters** (ASR / translate-and-speak / minutes), selected per task via
`set_adapter` (~3.5 GB total). Includes:
- `src/csmeeting/` — runtime: live engine, VAD, language-ID, translate, minutes, data-gen modules.
- `scripts/` — `serve_api.py` (FastAPI/WSS bridge), `demo_app.py` (Gradio), `live_translate.py`,
  `minutes_demo.py`, and the data/eval CLIs.
- `web/` — React + Vite + TS web UI (responsive translation console).
- `documentation/`, `docs/` — architecture, running, fine-tuning.

Run: `python scripts/serve_api.py --asr-lora <...> --tt-adapter <...> --minutes-lora <...>`
(see `kaigi-app/README.md` and `documentation/running.md`).

## asr-ft-serving/ — Gradio ASR demos (asr-ft)
- `serve_gradio.py`, `serve_gradio_lora.py` — base vs LoRA side-by-side transcription UI.
- `app_torch.py` — PyTorch inference app.
