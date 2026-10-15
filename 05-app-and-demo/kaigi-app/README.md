# 会議 Kaigi — on-device JP↔EN code-switching meeting assistant

A privacy-first, **fully on-device** meeting assistant built on Liquid AI **LFM2.5** models. Three features from one
local model stack:

1. **Live code-switched transcription** — Japanese with embedded English, transcribed with the loanword-vs-switch
   distinction preserved (katakana = naturalized loanword; Latin = true English switch), where generic ASR fails.
2. **Live bidirectional translation (JA↔EN) with spoken output** — speak Japanese *or* English; the language is
   auto-detected and the other side hears the translation **≈0.9 s** after the speaker pauses (streaming TTS).
3. **Confidential auto-minutes** — summary / decisions / action items (with owners), generated locally.

> **Why a small LFM is the unlock:** confidential bilingual meetings (M&A, PII, personnel) legally/contractually
> can't go to the cloud (Japan's APPI). A model that runs **100% offline** is the only compliant option — and one
> *tuned for the loanword/switch boundary* beats generic cloud ASR on exactly the speech Japanese firms use.

**▶️ How to run (GPU + CPU): [`documentation/running.md`](documentation/running.md) ·
📐 Architecture: [`documentation/system-architecture.md`](documentation/system-architecture.md) ·
🧮 Resources: [`documentation/system-resources.md`](documentation/system-resources.md) ·
🎛 Fine-tuning: [translate+TTS](documentation/finetuning-translate-tts.md) · [minutes](documentation/finetuning-minutes.md)**

---

## Architecture (at a glance)

**One `LFM2.5-Audio-1.5B-JP` base + three swappable LoRA adapters** — the whole assistant runs on a single audio model:

```
🎙 ─► VAD ─► ASR ───────────► translate + speak ──────────► 🔊 (streaming, other language, 24 kHz)
     pause  [adapter: asr]     [adapter: default]
              │ source text        │ translated text (emitted first, then speech)
              └──────── running transcript ────► Minutes  [adapter: minutes]
                                                  要約/決定/アクション + EN summary
        └────── all on-device · 0 bytes to cloud · one base, adapters switched per task (~3 GB) ──────┘
```

The **translate-and-speak** adapter outputs the translated **text first, then the speech** in one pass — so there's
no separate text translator. ASR, translate+speak, and minutes are three LoRA adapters on the *same* base, selected
per task via `set_adapter` (no quality/speed penalty vs separate models). This replaced the old 4-model cascade
(`LFM2.5-1.2B-JP` text translator + a second English audio model are both gone). Fine-tuning details:
[`documentation/finetuning-translate-tts.md`](documentation/finetuning-translate-tts.md) ·
[`documentation/finetuning-minutes.md`](documentation/finetuning-minutes.md). Why a cascade (verb-final latency):
[`docs/translation.md`](docs/translation.md).

## Status (measured on H100)

The app now runs as **one `LFM2.5-Audio-1.5B-JP` base + 3 LoRA adapters** (all fine-tuned, ~48 MB each):

| Pillar | How | Status |
|---|---|---|
| ASR (code-switch, JA+EN) | base + **`asr` LoRA** (GPU PyTorch) | ✅ ~1–2 s, contention-immune |
| **Translate + speak** (JA↔EN) | base + **`default` LoRA** (`v6_libritts`, best) — translate-and-speak in one pass | ✅ replaces the separate text translator; v6 (+ real LibriTTS-R English) beat v3 by +4.9 JA→EN translate / +2.7 EN audio |
| **Minutes** | base + **`minutes` LoRA** (`minutes_lora_v5`) — text→text on the backbone | ✅ replaces `LFM2.5-1.2B-JP`; correct attribution + 未指定 + English summary |
| **Auto language-ID** (direction) | LFM ASR transcript + script (`lid.lang_of`, no Whisper) | ✅ ja/en per utterance |
| **Adapter switching** | one base, `set_adapter` per task | ✅ no quality/speed penalty; **~3.5 GB** total |
| **Web UI** (React) + FastAPI/WSS bridge | `web/` + `scripts/serve_api.py` | ✅ live over HTTPS (mic works) |
| GGUF CPU ASR (on-device path) | llama.cpp liquid-audio runner (docker) | ✅ works; CPU-only, for the laptop target |
| On-device deploy (Ryzen/ONNX) | — | ⬜ stretch |

Per-utterance latency (GPU): ASR ~1–2 s · translate+speak first-audio ~0.5 s (streamed) · minutes a few s.
**Bidirectional & auto:** one base does JA+EN ASR, both translation directions, both spoken outputs, and minutes;
direction auto-detected per utterance (script-based) or set in the UI. The fine-tunes: see
[`documentation/finetuning-translate-tts.md`](documentation/finetuning-translate-tts.md) and
[`documentation/finetuning-minutes.md`](documentation/finetuning-minutes.md).

---

## Quickstart

### Run the assistant (the demo)

**Modern web UI (React)** — recommended; full details in [`web/README.md`](web/README.md):
```bash
cd web && npm install && npm run dev            # Demo mode (simulated), http://localhost:5174
# Live mode — wire the real on-device models via the FastAPI bridge (auto-detected by the UI):
pip install -e ".[web,live]"                    # or ".[web,cpu]" for CPU/ONNX
python scripts/serve_api.py --mock              # bridge wiring only, no models

# Consolidated single-model stack (one audio base + 3 LoRA adapters; text model dropped):
python scripts/serve_api.py \
  --asr-lora     <path/to/cs_asr_lora> \  # code-switch ASR LoRA on GPU
  --tt-adapter   <…/optionD/checkpoints/v6_libritts/final> \   # translate + speak in one pass (best: v6)
  --minutes-lora <…/optionC/checkpoints/minutes_lora_v5/final> \    # minutes on the backbone
  --no-text-model                                            # drop LFM2.5-1.2B-JP entirely
# add --web-dist web/dist  and  --ssl-certfile/--ssl-keyfile  to serve the UI over HTTPS (mic needs a secure origin)
# alt ASR for the on-device/laptop target: --asr-gguf-url http://localhost:8090 (CPU GGUF; see documentation/)
```


**Gradio UI** (single-file, no Node):
```bash
pip install -e ".[demo]"                       # needs Python 3.12 (liquid-audio); CUDA GPU
python scripts/demo_app.py                      # Gradio UI: speak JP → transcript + English + minutes
python scripts/demo_app.py --mock               # UI logic only, no models
python scripts/demo_app.py --asr-model <your-finetuned-cs-asr>   # wire the code-switch fine-tune

# CPU-only (no GPU) — ONNX backend, LFM models only (see documentation/system-resources.md §B):
pip install -e ".[cpu]"     # onnxruntime + the LFM text model on CPU; also needs liquidonnx (onnx-export)
python scripts/demo_app.py --backend onnx --device cpu \
    --en-onnx <LFM2.5-Audio-1.5B-ONNX> --ja-onnx <LFM2.5-Audio-1.5B-JP-ONNX> --onnx-src <onnx-export/src>

# Ryzen/Radeon iGPU — push CPU TTS toward real-time (RTF<1) via DirectML (needs `onnxruntime-directml`):
python scripts/demo_app.py --backend onnx --device cpu --onnx-ep dml \
    --en-onnx <…-ONNX> --ja-onnx <…-JP-ONNX> --onnx-src <onnx-export/src>
python scripts/bench_tts.py --backend onnx --onnx-ep dml --en-onnx <…-ONNX> --onnx-src <…>  # measure RTF/TTFA
```
Latency options (what's real-time, the CPU-TTS fix, interleaved generation: `--tts-mode interleaved`).
⚠️ Pin **torch cu128** (not cu130) — see [`documentation/system-resources.md`](documentation/system-resources.md).

### Component tools
```bash
python scripts/live_translate.py --wav meeting.wav   # cascade on a file (or --mic / --mock)
python scripts/eval_translation.py --n 200           # MT quality (chrF/BLEU on FLEURS) — test-first gate
python scripts/minutes_demo.py                        # minutes on sample transcripts (or --transcript file)
```

### Generate the ASR fine-tune data (feeds the ASR fine-tune)
```bash
python scripts/00_smoke.py                            # offline data-gen wiring test
python scripts/01_gen_transcripts.py --n-template 300 # code-switch transcripts (seeds + contrast pairs)
python scripts/02_synthesize.py                       # → wav + manifest (bilingual TTS)
# then drop cs_asr_iterator into the kit — see kit_integration/
```

---

## Repository map

| Path | What |
|---|---|
| `src/csmeeting/live/` | Cascade runtime — `config`, `engine` (ASR + streaming TTS), `translate` (MT), `vad`, `pipeline` |
| `src/csmeeting/mt/` | MT prompt, eval harness, fine-tune data prep |
| `src/csmeeting/minutes/` | Minutes prompt + generator |
| `src/csmeeting/app/` | `assistant.py` — composes everything for the UI |
| `src/csmeeting/convention.py` | Script→language span segmentation (the loanword/switch rule, as code) |
| `src/csmeeting/{gen_transcripts,tts_backends,synthesize,manifest,cs_asr_iterator}.py` | ASR fine-tune data-gen |
| `scripts/` | CLIs — `demo_app`, `serve_api` (web bridge), `live_translate`, `eval_translation`, `minutes_demo`, `prep_mt_data`, `0*_…` data-gen |
| `web/` | **React web UI** — responsive translation console (Vite + TS + Tailwind + GSAP); talks to `serve_api` |
| `documentation/` | **System docs** — architecture, resources, running, **fine-tuning** (`finetuning-translate-tts.md`, `finetuning-minutes.md`) |
| `docs/` | Component design notes — `transcription_convention`, `translation`, `minutes` |
| `kit_integration/` | Drop-in for the Liquid4All LFM audio training kit |
| `config/pipeline.yaml`, `pyproject.toml` | Config + packaging (extras: `live`, `mt`, `demo`, `pack`, `llm`, `eval`) |

## Data mix (ASR fine-tune)

To add Japanese + the switch skill *without forgetting English*:
- ~20–35% **synthetic JP↔EN code-switch** (this repo) · ~50–70% **monolingual JA** · ~10–15% **English replay**.
- Labeling convention (**align the team on this first**): [`docs/transcription_convention.md`](docs/transcription_convention.md).

## Open work

- 🔧 Wire the teammate's **code-switch ASR checkpoint** (`--asr-model`, one flag).
- ⬜ **On-device deploy** (Ryzen AI / ONNX-Q4) — the target is a laptop, not the H100.
- 🟡 **Commercial-clean synth TTS** for data-gen (edge-tts → Kokoro/CosyVoice2).
- 🟡 Launch the **live demo**; ⬜ monolingual/parallel corpus download + JECS eval set.
