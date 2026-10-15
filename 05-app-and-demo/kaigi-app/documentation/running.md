# Running Kaigi

Two backends:
- **`torch` (GPU)** — PyTorch/`liquid_audio`, needs a CUDA GPU. Fastest; spoken TTS is real-time (streaming).
- **`onnx` (CPU)** — onnxruntime, **CPU-only, LFM models only**. ASR/translation/minutes real-time; **spoken TTS is
  turn-based** (RTF ~2.6, not live) — see [`system-resources.md`](system-resources.md) §B.

Everything is LFM-only and runs offline at inference (models download once).

> **⚡ Consolidated stack (recommended).** The app runs on **one `LFM2.5-Audio-1.5B-JP` base + 3 LoRA adapters**
> (asr / translate-and-speak / minutes), dropping the separate text model. Launch the React backend with:
> ```bash
> python scripts/serve_api.py \
>   --asr-lora     <path/to/cs_asr_lora> \
>   --tt-adapter   <…/optionD/checkpoints/v6_libritts/final> \
>   --minutes-lora <…/optionC/checkpoints/minutes_lora_v5/final> \
>   --no-text-model --web-dist web/dist \
>   --ssl-certfile cert.pem --ssl-keyfile key.pem --port 8800     # HTTPS so the browser mic works over the network
> ```
> Set `HF_HOME` to your model cache and `HF_HUB_OFFLINE=1` if weights are pre-downloaded (avoids a proxy-blocked
> re-download). On a CPU-only / on-device host, swap `--asr-lora` for `--asr-gguf-url http://localhost:8090`
> (the llama.cpp GGUF ASR runner — see [`finetuning-translate-tts.md`](finetuning-translate-tts.md) /
> the GGUF notes). Fine-tune details: [`finetuning-translate-tts.md`](finetuning-translate-tts.md) ·
> [`finetuning-minutes.md`](finetuning-minutes.md).


---

## 0. Prerequisites

- **Python 3.12** (required by `liquid-audio` and by the ONNX runtime path).
- `git`, and [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`.
- A Hugging Face account/token is optional (public models), but set `HF_TOKEN` to avoid rate limits.
- GPU path only: an NVIDIA GPU with a **CUDA 12.x** driver.

Clone this repo and `cd` into it (`the project`).

---

## 1. GPU path (`torch` backend)

```bash
uv venv --python 3.12 && source .venv/bin/activate     # or: python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[demo]"
# IMPORTANT: pin CUDA-12.x torch (a fresh install may pull cu130, which fails on a 12.x driver -> cuda False):
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

python scripts/demo_app.py                              # Gradio UI on http://localhost:7860
python scripts/demo_app.py --asr-model <your-cs-asr>   # swap in the code-switch ASR fine-tune (JA side)
python scripts/demo_app.py --mock                      # UI logic only, no models
```
Models (~10 GB) download on first run. Full-app VRAM ≈ 8.4 GB; use a ≥12 GB GPU.

---

## 2. CPU path (`onnx` backend) — no GPU

The audio runs as ONNX on CPU. You need (a) the ONNX runtime package `liquidonnx`, (b) the **two audio ONNX models**.

### 2.1 Get `liquidonnx` (the onnx-export package)

```bash
git clone https://github.com/Liquid4All/onnx-export.git
cd onnx-export && uv sync && cd ..        # installs onnxruntime + liquidonnx + scipy into onnx-export/.venv
```
`liquidonnx` is **not on PyPI**, so either run inside that venv, or point our app at it with `--onnx-src onnx-export/src`.

### 2.2 Get the audio ONNX models

**English (download — repo exists):**
```bash
huggingface-cli download LiquidAI/LFM2.5-Audio-1.5B-ONNX \
    --include "config.json" "tokenizer*" "onnx/*q4*" "onnx/*.bin" "onnx/*.json" \
    --local-dir ./models/LFM2.5-Audio-1.5B-ONNX
```

**Japanese (no repo → export once, on a GPU machine, then copy to the CPU box):**
```bash
# in the onnx-export checkout, on a CUDA GPU (export tracing needs cu128 torch):
cd onnx-export
uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
CUDA_VISIBLE_DEVICES=0 uv run lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B-JP --precision q4 --output-dir ./exports
# -> ./exports/exports/LFM2.5-Audio-1.5B-JP-ONNX   (copy this folder to your CPU machine)
cd ..
```
> Export is a **one-time GPU step** that produces **CPU-runnable** ONNX. You can't export on a pure-CPU laptop — do it
> on any CUDA box (or reuse the one already exported on `the GPU host` at
> `/data/lfm2.5/run-c/onnx-exports/exports/LFM2.5-Audio-1.5B-JP-ONNX`).

### 2.3 Install this repo's CPU deps + run

```bash
pip install -e ".[cpu]"        # onnxruntime, scipy, transformers, soundfile, numpy (LFM text model runs on CPU)

python scripts/demo_app.py --backend onnx --device cpu \
    --en-onnx ./models/LFM2.5-Audio-1.5B-ONNX \
    --ja-onnx ./models/LFM2.5-Audio-1.5B-JP-ONNX \
    --onnx-src ./onnx-export/src
```
Open http://localhost:7860, pick a **Direction** (Auto / JA→EN / EN→JA), record, stop. Transcription + translation
text appear immediately; the spoken translation plays after it finishes generating (turn-based on CPU).

> Tip: run inside `onnx-export/.venv` (which already has onnxruntime+liquidonnx) and `pip install -e .` this repo into
> it — then you can drop `--onnx-src`.

---

## 3. Direction & language

- **Auto** (default): the language is detected from the LFM ASR transcript (any Japanese script ⇒ Japanese →
  translate to English; otherwise → Japanese). Works best with the code-switch ASR model on the JA side.
- **JA→EN / EN→JA**: force a direction (most reliable for a demo).

---

## 4. Component tools (CLIs)

```bash
# Cascade on a Japanese file or mic (GPU backend):
python scripts/live_translate.py --wav meeting.wav        # or --mic / --mock

# MT quality (decide if the translator needs fine-tuning):
pip install -e ".[mt]"
python scripts/eval_translation.py --n 200                # chrF/BLEU on FLEURS ja->en

# Minutes from a transcript:
python scripts/minutes_demo.py                            # built-in samples, or --transcript file.txt

# Generate the synthetic code-switch ASR training data:
python scripts/00_smoke.py                                # offline wiring test
python scripts/01_gen_transcripts.py --n-template 300
python scripts/02_synthesize.py
```

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False on a 12.x driver | You have a cu130 torch — reinstall `torch==2.8.0+cu128` from the cu128 index. |
| `ModuleNotFoundError: liquidonnx` (CPU path) | Pass `--onnx-src <onnx-export/src>`, or run inside `onnx-export/.venv`. |
| `OnnxAudioEngine: set ja_onnx_dir and/or en_onnx_dir` | Pass `--en-onnx` and/or `--ja-onnx` pointing at the model folders (the dir that contains `onnx/` + `config.json`). |
| Spoken English/Japanese is slow / lags | Expected on CPU (TTS RTF ~2.6, turn-based). Use the GPU backend, or the Ryzen iGPU via onnxruntime DirectML. |
| `lfm2-audio-export` fails tracing on GPU | Pin cu128 torch in the onnx-export venv first (§2.2). |
| Audio model needs Python ≥3.12 | `liquid-audio` requires 3.12; use a 3.12 venv. |

See also: [`system-architecture.md`](system-architecture.md) (components) and [`system-resources.md`](system-resources.md)
(footprints + measured CPU/GPU latency).
