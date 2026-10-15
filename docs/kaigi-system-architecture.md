# Kaigi — System Architecture

**Kaigi (会議)** is an on-device Japanese↔English **code-switching meeting assistant** built on Liquid AI LFM2.5
models. Everything runs locally so confidential meeting audio never leaves the machine (the privacy/sovereignty
"unlock" — Japan's APPI treats meeting/medical audio as data that legally shouldn't go to the cloud).

Three features, all from the same on-device model stack:
1. **Live code-switched transcription** — Japanese with embedded English, transcribed with the loanword‑vs‑switch
   distinction preserved (katakana = naturalized loanword; Latin = true English switch).
2. **Live bidirectional translation (JA↔EN) with spoken output** — speak Japanese *or* English; the language is
   auto-detected (or set manually) and the listener hears the other side in ~real time.
3. **Confidential auto‑minutes** — summary / decisions / action items (with owners), generated locally.

> Status legend in this doc: ✅ validated on real weights (on the H100 host) · 🟡 built, not yet run on real weights
> · 🔧 owned by a teammate · ⬜ not started.

> **⚡ Current architecture (consolidated).** The app now runs as **one `LFM2.5-Audio-1.5B-JP` base + three LoRA
> adapters** — `asr` (code-switch), `default` (translate-and-speak), `minutes` — switched per task via `set_adapter`.
> The separate `LFM2.5-1.2B-JP` text translator/minutes model **and** the second (English) audio model have been
> **eliminated**. §2 and §4 below describe the original cascade for context; the fine-tunes that enabled the
> consolidation are documented in [`finetuning-translate-tts.md`](finetuning-translate-tts.md) and
> [`finetuning-minutes.md`](finetuning-minutes.md), and summarized in §2.1.

---

## 1. High-level architecture

```
                          ┌─────────────────────────  on-device (no cloud)  ──────────────────────────┐
   🎙 JA speech ─► VAD ──► ASR ──────────► MT ──────────► TTS (streaming) ─► 🔊 English (≈0.9 s lag)
   (mic / wav)    pause-   LFM2.5-Audio    LFM2.5-1.2B-JP  LFM2.5-Audio                                  
                  seg.     "Perform ASR     (JA→EN)         English voice                                
                           in japanese."                   (frame-streamed)                             
                              │                  │                                                       
                              │ JA transcript    │ EN translation                                        
                              ▼                  ▼                                                        
                        ┌───────────────────────────────┐     on demand                                 
                        │  running transcript (accumulated)│ ───────────► Minutes (LFM2.5-1.2B-JP)        
                        └───────────────────────────────┘              要約/決定/アクション + EN summary  
                          └────────────────────────────  Gradio UI  ───────────────────────────────────┘
```

**Why a cascade (ASR→MT→TTS), not direct speech-to-speech:** Japanese is verb‑final (SOV), so the latency floor is
*linguistic* (you must wait ~a clause for the verb to translate), not the number of model hops — a streamed cascade
matches S2S on perceived latency for this pair. The cascade also **reuses the ASR fine‑tune** and **emits the text**
that the transcription and minutes features need anyway. (Full rationale: [`../docs/translation.md`](../docs/translation.md).)

---

## 2. Models

| Pillar | Model | Params | Role | Status |
|---|---|---|---|---|
| **ASR** | `LFM2.5-Audio-1.5B` (fine-tuned from `-JP` for code-switch) | 1.45B | JA / code-switch audio → JA text | 🔧 fine-tune by teammate; base `-JP` works today ✅ |
| **MT** | `LFM2.5-1.2B-JP` (base) | 1.17B | JA text → EN text | ✅ base good, no fine-tune |
| **Minutes** | `LFM2.5-1.2B-JP` (base, **shared with MT**) | — | transcript → bilingual minutes | ✅ base + prompt good |
| **TTS** | `LFM2.5-Audio-1.5B` (base, English voice) | 1.45B | EN text → 24 kHz English audio | ✅ streaming |

The MT and Minutes stages are the **same loaded text model** driven by two different prompts, so the running app
holds **2 audio models + 1 text model** (see [`system-resources.md`](system-resources.md)).

### 2.1 Consolidated stack (current) — one base + three LoRA adapters

The models above were collapsed onto a **single `LFM2.5-Audio-1.5B-JP` base** carrying three LoRA adapters, each
~48 MB (r=16, α=32, 0.85 % trainable). Only one is active per call (`PeftModel.set_adapter`); `disable_adapter()`
gives the clean base for any plain ASR/TTS. **No quality or speed penalty vs separate instances** (switching just
selects the active low-rank weights); **~3.5 GB total** instead of ~8 GB.

| Adapter | Task | Prompt(s) | Output | Fine-tune doc |
|---|---|---|---|---|
| `asr` | code-switch ASR (JA+EN) | `"Perform ASR."` | source text | teammate's `lfm_lora` |
| `default` | translate **and** speak | `"Translate to English/Japanese and speak."` | translated text **then** 24 kHz audio | [`finetuning-translate-tts.md`](finetuning-translate-tts.md) (`transtts_lora_v3`) |
| `minutes` | transcript → minutes | minutes `SYSTEM_PROMPT` | bilingual minutes (text) | [`finetuning-minutes.md`](finetuning-minutes.md) (`minutes_lora_v5`) |

Routing per utterance (`Assistant` + `AudioEngine`): `transcribe` → `set_adapter("asr")`; `translate_speak_stream`
→ `set_adapter("default")` (emits translated text, then streamed speech); `minutes` → `set_adapter("minutes")`
(text→text, whole-sequence decode), each restoring the primary. ASR can alternatively run on the **GGUF CPU** runner
(`asr_gguf_url`) for the on-device/laptop target. Diagram:

```
🎙 ─► VAD ─► [set_adapter asr] ASR ─► [set_adapter default] translate+speak ─► 🔊 (streamed)
                  │ source text                │ translated text → audio
                  └──── transcript ──► [set_adapter minutes] Minutes (要約/決定/アクション + EN summary)
                          one LFM2.5-Audio-1.5B-JP base · adapters switched per task · ~3.5 GB
```

**Bidirectional routing (no extra models).** Each audio model does ASR **and** TTS for its language, and the text
model translates both ways, so the same loaded stack serves both directions by routing prompts:
| Direction | ASR | MT | TTS |
|---|---|---|---|
| JA→EN | JP audio (`"Perform ASR in japanese."`) | LFM2.5-1.2B-JP (JA→EN prompt) | English audio (`"Perform TTS. Use the US voice."`) |
| EN→JA | English audio (`"Perform ASR."`) | LFM2.5-1.2B-JP (EN→JA prompt) | JP audio (`"Perform TTS in japanese."`) |

**Direction control.** `LiveConfig.direction` ∈ `auto` / `ja2en` / `en2ja`. In **auto** (LFM-only, no Whisper): we
transcribe with the LFM ASR and classify by **script** (`lid.lang_of`: any kana/kanji ⇒ ja — true even for
code-switched Japanese, which always has kana; pure English has none), then translate to the other language. Manual
modes skip detection. (The teammate's *code-switch* ASR makes this a single both-languages ASR.)

**Backend (GPU vs CPU).** `LiveConfig.audio_backend` ∈ `torch` (PyTorch/`liquid_audio`, **CUDA**) or `onnx`
(`OnnxAudioEngine` → onnxruntime, **CPU-only, LFM model**). The ONNX path is LFM-only and validated on CPU (ASR
real-time; TTS turn-based at RTF ~2.6) — see [`system-resources.md`](system-resources.md) §B.

---

## 3. The LFM2.5-Audio model (internals)

An **end-to-end audio‑language model** (not a pipeline of separate ASR/TTS):

- **Backbone:** LFM2.5 hybrid (gated short convolutions + a few GQA attention blocks), 1.2B, 32,768‑token context.
- **Audio in:** ~115M **FastConformer** encoder (from `nvidia/canary-180m-flash`) → continuous embeddings.
- **Audio out:** an **RQ‑Transformer** emits **8 Mimi codebooks** per frame → a lightweight **LFM‑based detokenizer**
  → **24 kHz** waveform.
- **Frame rate:** Mimi runs at **12.5 Hz = 80 ms per frame = 1920 samples** @ 24 kHz. (Key for streaming.)
- **Two generation modes:** `generate_sequential` (ASR / TTS — what we use) and `generate_interleaved` (real‑time
  speech‑to‑speech). **The task is selected by the system prompt** (`"Perform ASR in japanese."`,
  `"Perform TTS. Use the US female voice."`).
- **Detokenizer is causal** (sliding‑window attention, window ≈30 frames) — this is what makes streaming decode valid.
- Loaded via the **`liquid_audio`** package (PyTorch); not a stock `transformers` class. Requires a **CUDA GPU** in the
  current path (the detokenizer is hard‑pinned to CUDA).

---

## 4. Components (detailed)

### 4.1 VAD / segmentation — `src/csmeeting/live/vad.py`
Energy‑based VAD (no deps) with an optional Silero backend. Segments audio into **utterances on the pause**
("translate on the pause"). For JA→EN this is both simpler and higher quality — waiting for the pause yields the
sentence‑final verb, giving a grammatical translation. ✅ (file mode validated; mic streaming uses the same `EnergyVAD`).

### 4.2 ASR — `src/csmeeting/live/engine.py::AudioEngine.transcribe`
`liquid_audio` `ChatState` + `generate_sequential`, system prompt `"Perform ASR in japanese."`. Decoded text tokens
are concatenated and **special tokens (`<|im_end|>`) stripped**. Uses base `LFM2.5-Audio-1.5B-JP` today; point
`LiveConfig.asr_model_id` (or `--asr-model`) at the teammate's **code‑switch fine‑tune** when ready. ✅ (accurate on a
clean clip; code‑switch robustness comes from the fine‑tune).

### 4.3 MT — `src/csmeeting/live/translate.py` + `src/csmeeting/mt/prompt.py`
`LFM2.5-1.2B-JP` + a translation prompt with in‑domain few‑shots. **Test‑first verdict: ship the base model, no
fine‑tune** — chrF 48.6 / BLEU 12.8 on FLEURS ja→en, with fluent, accurate samples (low BLEU is a metric artifact of
FLEURS's lowercase single references). Code‑switched input is *easier* (the English passes through). SFT data is
prepared as a backstop (`scripts/prep_mt_data.py`) but unused. ✅

### 4.4 TTS — `engine.py::synthesize` (batch) and `engine.py::synthesize_stream` (streaming)
Base `LFM2.5-Audio-1.5B` English voice. **Streaming** is the production path: decode + emit each 80 ms frame in a
**bounded causal window** (`frames[finalized-32 : n]` every 8 frames, holding back 2 frames for the ISTFT edge),
**filtering the reserved terminal frame** (code ≥2048, else `decode()` raises), with a one‑time **detokenizer warmup**.
Measured: **time‑to‑first‑audio ≈ 0.46 s** (vs 2.83 s batch), **RTF ≈ 0.42** (generation outpaces playback → no
starvation). ✅

### 4.5 Minutes — `src/csmeeting/minutes/`
`LFM2.5-1.2B-JP` + a system prompt with an **explicit owner‑attribution rule + one‑shot example** (this fixed the
v1 failure where all action items were assigned to one speaker). Produces 要約/Summary, 決定事項/Decisions,
アクションアイテム (with 担当/owner + 期限/due), and an English summary. **Use JP, not `-Thinking`** (Thinking
over‑reasons and fails to emit the minutes). ✅ ([`../docs/minutes.md`](../docs/minutes.md)).

### 4.6 Assistant — `src/csmeeting/app/assistant.py`
Composes the pieces into one object: the `AudioEngine` (ASR + TTS) + **one shared text LM** (MT + minutes) + the
running transcript. `process_stream(wav, sr)` yields `(ja, en, audio_chunk)` — text first, then streamed English
audio. `minutes()` renders from the accumulated transcript. 🟡 (compiles; real‑weights run pending a UI launch).

### 4.7 Demo app — `scripts/demo_app.py`
Gradio UI: mic → live JA transcript + latest EN translation + **streamed** spoken English (`gr.Audio(streaming=True)`)
+ a "Generate minutes" button, with a "0 bytes to cloud" framing. `--mock` runs the UI logic with no models;
`--share` exposes a public link; `--asr-model` swaps in the fine‑tuned ASR. 🟡

### 4.8 Data‑generation pipeline (for the ASR fine‑tune) — `src/csmeeting/{convention,gen_transcripts,tts_backends,synthesize,manifest,cs_asr_iterator}.py`
Produces the **synthetic JP↔EN code‑switch corpus** that trains the ASR fine‑tune (the teammate's model). Core idea:
**script encodes language** (`convention.py` auto‑segments gold text into ja/en spans by Unicode script), so each span
is rendered with a script‑matched TTS voice. `cs_asr_iterator.py` yields the `list[ChatMessage]` the Liquid audio
trainer consumes. ✅ (generation validated; **TTS backend pivot pending** — edge‑tts → Kokoro/CosyVoice2 for
commercial‑clean output).

---

## 5. Data flow & latency (per utterance, measured on H100 GPU 0)

| Step | Component | Time |
|---|---|---|
| Segment utterance | VAD | (on pause) |
| Audio → JA text | ASR | **0.3 s** |
| JA → EN text | MT | **0.1 s** |
| EN text → first audio | streaming TTS | **0.46 s** (then continuous, RTF 0.42) |
| **Listener hears English** | — | **≈ 0.9 s after the speaker pauses** |
| Transcript → minutes (on demand) | Minutes | ~1.9 s (768 tokens) |
| One‑time model load | — | ~6.3 s |

---

## 6. Repository map

| Path | Purpose |
|---|---|
| `src/csmeeting/live/` | Cascade runtime: `config`, `engine` (ASR+TTS, streaming), `translate` (MT), `vad`, `pipeline` |
| `src/csmeeting/mt/` | MT prompt, eval harness, fine‑tune data prep |
| `src/csmeeting/minutes/` | Minutes prompt + generator |
| `src/csmeeting/app/` | `assistant.py` — composes everything for the UI |
| `src/csmeeting/{convention,gen_transcripts,tts_backends,synthesize,manifest,cs_asr_iterator}.py` | ASR fine‑tune data‑gen pipeline |
| `scripts/` | CLIs: `demo_app`, `live_translate`, `eval_translation`, `prep_mt_data`, `minutes_demo`, `00_smoke`, `01_gen_transcripts`, `02_synthesize` |
| `docs/` | Component design notes: `transcription_convention`, `translation`, `minutes` |
| `documentation/` | System docs (this file + `system-resources.md`) |
| `kit_integration/` | Drop‑in for the Liquid4All LFM audio training kit |
| `config/pipeline.yaml`, `pyproject.toml` | Config + packaging (extras: `live`, `mt`, `demo`, `pack`, `llm`, `eval`, `ffmpeg`) |

---

## 7. Key design decisions

1. **Cascade over direct S2S** — JA verb‑final ⇒ latency is linguistic; cascade reuses ASR + yields needed text.
2. **Test‑first on MT and Minutes** — validated the base model is good enough, *avoided two fine‑tunes*.
3. **Script = language** — the transcription convention encodes loanword‑vs‑switch in the script itself.
4. **Streaming TTS via bounded causal‑window decode** — the real‑time unlock (0.46 s TTFA).
5. **Shared text model for MT + minutes** — one 1.17B model, two prompts → laptop‑friendly memory.
6. **Two audio models (ASR fine‑tune + base‑English TTS) vs one** — English TTS needs the English checkpoint;
   alternatively fine‑tune ASR *from the English base* to keep one model that does both (decided after the ASR
   fine‑tune lands — saves ~3 GB).

---

## 8. Integration points & remaining work

- 🔧 **Teammate's code‑switch ASR checkpoint** → set `--asr-model <hf-repo-or-path>`. One flag, no code change.
- ⬜ **On‑device deploy** (AMD Ryzen AI): the current PyTorch path needs CUDA; the laptop target is **ONNX‑Q4
  (`lfm2-audio-infer`) / GGUF** + llama.cpp for text. See [`system-resources.md`](system-resources.md) §B.
- ⬜ **Live demo** standing up the Gradio app on a GPU machine (or the laptop).
- 🟡 **Commercial‑clean synth TTS** for the data‑gen pipeline (Kokoro/CosyVoice2).
- The Liquid4All LFM audio training kit is the training harness for the audio fine‑tune (`kit_integration/`).

---

## 9. Validation status

| Component | Validated on real weights? | Evidence |
|---|---|---|
| MT (JA↔EN) | ✅ | chrF 48.6 / BLEU 12.8 on FLEURS (JA→EN); EN→JA validated on GPU |
| Minutes | ✅ | correct owner attribution on sample transcripts, GPU |
| Cascade ASR→MT→TTS | ✅ | round‑trip on GPU, 0.9 s/utterance |
| Streaming TTS (JA + EN) | ✅ | TTFA 0.46 s (EN voice) / 0.31 s (JA voice), RTF 0.42 |
| EN→JA direction (full round-trip) | ✅ | EN ASR + EN→JA MT + JA TTS on GPU |
| Auto language-ID (LFM-only) | ✅ | LFM ASR transcript script; ja/en classified correctly |
| CPU/ONNX backend (no GPU) | ✅ | ASR ~1.6 s real-time; TTS RTF ~2.6 (EN)/3.1 (JA), turn-based |
| Full‑app VRAM | ✅ | 8.40 GB (measured) |
| Assistant / Gradio UI | 🟡 | compiles; not yet launched on real weights |
| Code‑switch ASR (fine‑tuned) | 🔧 | teammate |
| On‑device (Ryzen/ONNX) | ⬜ | — |
