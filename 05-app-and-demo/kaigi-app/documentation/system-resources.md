# Kaigi — System Resources

Resources to **run the full application** (inference of all three pillars), plus the separate requirements for the
**on‑device deploy target** and for **fine‑tuning / data work**. Numbers marked *(measured)* were taken on the
project host (`the GPU host`, NVIDIA H100 NVL, GPU 0); others are from vendor/research figures and flagged as estimates.

---

## A. Running the full application (inference)

The running app loads **2 audio models + 1 text model** (MT and Minutes share one text model):

### Memory (VRAM) — *measured*, bf16

| Model | Params | VRAM (bf16) | On-disk (bf16) | On-disk (Q4) |
|---|---|---|---|---|
| ASR — `LFM2.5-Audio-1.5B` (+`-JP`/fine-tune) | 1.45B | **3.03 GB** | ~3.65 GB | ~1.4 GB¹ |
| TTS — `LFM2.5-Audio-1.5B` (English) | 1.45B | **3.03 GB** | ~3.65 GB | ~1.4 GB¹ |
| MT + Minutes — `LFM2.5-1.2B-JP` (shared) | 1.17B | **2.34 GB** | ~2.4 GB | ~0.73 GB (Q4_K_M) |
| **Full app (weights)** | — | **8.40 GB (measured)** | ~9.7 GB | ~3.5 GB |

¹ Audio GGUF is the LM backbone only (~0.57–1.25 GB); the FastConformer encoder + detokenizer add ~0.7 GB, so a fully
quantized audio model is ~1.4 GB.

- **Working VRAM** (weights + activations + KV cache during generation): budget **~10 GB** for bf16. A **12 GB GPU**
  is the practical minimum for the bf16 PyTorch path; 16–24 GB comfortable.
- **One‑audio‑model variant** (fine‑tune ASR *from the English base* so one checkpoint does ASR **and** English TTS):
  **~5.4 GB** weights → fits an 8 GB GPU. Recommended for the on‑device build.
- **Quantized (Q4) target:** ~2–3.5 GB → fits laptop iGPU/CPU RAM.

### Compute & latency — *measured*, H100 GPU 0

| Item | Value |
|---|---|
| Model load (3 models, cold) | ~6.3 s (one‑time) |
| ASR | 0.3 s / utterance |
| MT | 0.1 s / utterance |
| Minutes | ~1.9 s (768 tokens) |
| TTS — streaming time‑to‑first‑audio | **0.46 s** |
| TTS — real‑time factor | **0.42** (generation faster than playback) |
| End‑to‑end (speaker pause → English audio) | **≈ 0.9 s** |

On a high‑end GPU the app is comfortably real‑time. The bottleneck is TTS generation; **streaming keeps RTF < 1**,
which is the requirement for gap‑free playback (see §B for the laptop).

### Software stack

| Layer | Requirement |
|---|---|
| Audio runtime | **`liquid-audio` 1.3** → **Python ≥ 3.12** |
| Tensors | **`torch` ≥ 2.8 built for CUDA 12.x (cu128)** — **NOT cu130** (the H100 driver reports CUDA 12.7; cu130 → `cuda False`). Pin `torch==2.8.0+cu128 torchaudio==2.8.0`. |
| Text models | `transformers` ≥ 4.55, `accelerate` |
| Audio I/O | `soundfile`, `librosa` (pulled by liquid-audio) |
| Demo UI | `gradio` ≥ 4.44 |
| GPU | **A CUDA GPU is required for the current path** — the LFM detokenizer is hard‑pinned to CUDA (`.cuda()`); driver must support CUDA 12.x. |

The **MT / Minutes stage alone** (text models via `transformers`) runs on **Python 3.9+** and any CUDA GPU — only the
**audio** stage forces Python 3.12 + the CUDA detokenizer. Install extras: `pip install -e ".[live]"` (cascade) /
`".[demo]"` (UI) / `".[mt]"` (translation tooling).

### Disk

| Item | Size |
|---|---|
| Model weights cache (3 models, bf16) | ~10 GB |
| Quantized models (Q4, on‑device) | ~3.5 GB |
| Python env (torch + cuda wheels + deps) | ~6–8 GB |
| MT eval/SFT text data | ~10 MB |

### Network
**Offline at inference** — no data leaves the machine (the privacy story). Network is needed only **once** to
download model weights from Hugging Face.

### Minimum / recommended (inference)

| | Minimum | Recommended |
|---|---|---|
| GPU | 1× CUDA GPU, **12 GB**, CUDA 12.x driver | 16–24 GB |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB free | 40 GB free |
| Python | 3.12 (audio) | 3.12 |

---

## B. On‑device deploy target (AMD Ryzen AI laptop)

This is the **target deployment** and the reason for the privacy/offline design. **The CPU/ONNX backend IS built** (LFM-only, no GPU,
no Whisper) — `audio_backend="onnx"` + `device="cpu"` → `OnnxAudioEngine` wraps onnxruntime via `liquidonnx`.

**Measured on CPU (Xeon 6442Y, in-process, q4 ONNX — a *generous* proxy; a Ryzen laptop has fewer cores → slower):**

| Stage | CPU | Real-time? |
|---|---|---|
| ASR (JA & EN) | ~1.6 s per 3 s clip | ✅ yes |
| **TTS** | **RTF ≈ 2.6 (EN) / 3.1 (JA)** | ❌ **turn-based** (5 s reply ≈ 13 s) |
| MT / minutes (text) | a few s / utterance | ✅ fine |

So on pure CPU: **live transcription + translation-text + minutes work; spoken TTS is turn-based** (generate then
play, not live streaming). The Ryzen **Radeon iGPU via onnxruntime DirectML** EP is the way to push TTS toward
real-time — now wired via `--onnx-ep dml` (`LiveConfig.onnx_ep`; needs the `onnxruntime-directml` wheel). **Still
untested** — measure on the actual laptop with `python scripts/bench_tts.py --backend onnx --onnx-ep {cpu,dml}` and
fill in the real RTF here. The interleaved-generation option (`--tts-mode interleaved`) lowers time-to-first-audio further. Models: EN ONNX = `LiquidAI/LFM2.5-Audio-1.5B-ONNX`; **JA ONNX
must be exported** (`lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B-JP --precision q4`). Other facts:

- **The current PyTorch `liquid_audio` path will NOT run on a Ryzen AI laptop as‑is** — it requires CUDA (the
  detokenizer is CUDA‑pinned). The on‑device path is a **different runtime**:
  - **Audio:** ONNX‑Q4 via `lfm2-audio-infer`, or the **GGUF audio CLI** (`llama-liquid-audio-cli`, 4 GGUF files).
    *(No official Windows‑x64 audio binary yet → Linux/WSL2 or the ONNX path.)*
  - **Text (MT/minutes):** `llama.cpp` GGUF (Q4) — runs on CPU/iGPU/NPU via Lemonade/GAIA.
- **Footprint (quantized, estimate):** text ~0.7–0.9 GB; each audio model ~1.4 GB; full ~**2–3 GB RAM** (≈2 GB with
  the one‑audio‑model variant).
- **Performance (Ryzen AI 9 HX 370, research figures):** the 1.2B text model hits **~116 tok/s decode, 2,975 tok/s
  prefill at 856 MB** (Q4_0, llama.cpp). Audio is the risk: per‑frame generation is slower than on an H100, so the
  **RTF must stay < 1** — the streaming decoder helps, but quantization is essential. Validate TTS RTF on the actual
  laptop early.
- **NPU caveat:** the FastConformer encoder + audio detokenizer have **no NPU kernels** → those run on CPU/iGPU even in
  "NPU" mode; only the LM backbone offloads. Don't promise NPU‑accelerated speech‑to‑speech.
- **Minimum (estimate):** Ryzen AI 300‑series (Strix/Strix Halo), **16 GB RAM**, ~10 GB disk; quantized models; Linux
  or WSL2 for the audio CLI, or the ONNX‑CPU path on Windows.

---

## C. Fine‑tuning & data work (what we used on `the GPU host`)

Separate from running the app. Host: **4× NVIDIA H100 NVL (94 GB each), GPU 0; 96 CPU; 1 TB RAM; 10.5 TB shared root**.

| Task | Resource need | Notes |
|---|---|---|
| **Code‑switch ASR fine‑tune** (teammate) | **Full bf16** fine‑tune → ~A100‑80 GB / L40S‑48 GB / **1× H100**; kit recipe bs=64, ctx=256 | `liquid-audio` trainer is full‑FT (no official LoRA). Checkpoints ~3 GB (bf16) each. |
| **MT / Minutes fine‑tune** | LoRA on 1.2B → **any ≥16 GB GPU** | **Determined unnecessary** — base models validated. Adapters ~50 MB if ever done. |
| **Synthetic CS data‑gen** | CPU + a TTS model (Kokoro/CosyVoice2, GPU optional) | Produces the ASR training audio. |
| **Env** | `audio-venv` (uv, **Python 3.12**, torch **cu128**) for audio; system **Python 3.9** for text (transformers/datasets) | Two envs because liquid‑audio needs 3.12. |
| **Disk** | model cache ~12 GB; MT text data ~10 MB; mono/CS audio corpora = GBs; audio checkpoints ~3 GB each | Root was 100 % full (47 GB free) then freed to ~580 GB. No separate scratch volume. |
| **Artifacts dir** | `/data/lfm2.5/run-c` (`HF_HOME=…/hf_cache`, `env.sh` sets GPU 0 + cache) | Self‑contained host scripts: `eval_mt.py`, `minutes_v2.py`, `cascade_smoke.py`, `tts_stream2.py`. |

> **The cu130 gotcha (important):** a fresh `pip/uv install liquid-audio` pulls `torch 2.12+cu130` (CUDA 13), which
> fails on the CUDA‑12.7 driver (`cuda False`). Always pin **`torch==2.8.0+cu128`** from
> `https://download.pytorch.org/whl/cu128`.

---

## D. Quick capacity summary

| Scenario | VRAM/RAM | Disk | Python | GPU |
|---|---|---|---|---|
| **Run app (bf16, dev GPU)** | ~10 GB VRAM | ~20 GB | 3.12 | CUDA 12.x, ≥12 GB |
| **Run app (one‑audio‑model, bf16)** | ~6 GB VRAM | ~16 GB | 3.12 | CUDA 12.x, ≥8 GB |
| **Run app (Q4, on‑device target)** | ~2–3 GB RAM | ~10 GB | — | Ryzen AI (ONNX/GGUF) |
| **ASR fine‑tune** | ~40–80 GB VRAM | +tens GB | 3.12 | A100/L40S/H100 |
| **MT/minutes (text only)** | ~3 GB VRAM | ~5 GB | 3.9+ | any CUDA |
