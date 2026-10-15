# Fine-tuning: Translating-TTS (one audio model translates **and** speaks)

How the **translate + speak** LoRA was built — the adapter that lets a single `LFM2.5-Audio-1.5B-JP` take text in
one language and emit **translated speech** in the other, replacing the separate `LFM2.5-1.2B-JP` text translator.

- **Method:** LoRA (PEFT), not full fine-tune. **Trainable params: 12.4 M = 0.85 %** of 1.47 B.
- **Deployed (best) adapter:** **`optionD/checkpoints/v6_libritts/final`** (~48 MB) — v3 + real 24 kHz studio
  English (LibriTTS-R). Supersedes `optionB/checkpoints/transtts_lora_v3/final` (the prior best). See §8.
- **Scripts (on `the GPU host:/data/lfm2.5/run-c/`, mirrored in `scripts/optionB_*.py` / `scripts/optionD_*.py`):**
  `optionB_build_data.py`, `optionB_kokoro_gen.py`, `optionB_kokoro_gen_v3.py`, `optionB_preprocess.py`,
  `optionB/train_lora.py`, `optionB_eval*.py`, `optionD_libritts.py`, `optionD_eval.py`.

---

## 1. Why

The original cascade was **ASR (audio) → MT (`LFM2.5-1.2B-JP`, text) → TTS (audio)** — three models, the text model
loaded only to translate. Goal: fold the translation into the audio model's own LM backbone so it outputs the
translated **text + speech** in one `generate_sequential` pass, dropping the separate translator (~2.4 GB).

### Feasibility study first (measured, not assumed)
Before training we measured the **zero-shot** backbones on FLEURS (chrF):

| Path | JA→EN | EN→JA |
|---|---|---|
| `LFM2.5-1.2B-JP` (dedicated text model) | **47.0** | **34.5** |
| EN-Audio backbone (text→text) | 44.4 | 25.8 |
| JP-Audio backbone (text→text) | 33.3 | 25.2 |

And cross-lingual TTS (round-trip ASR chrF): JP-Audio can speak English only *partially* (degrades), EN-Audio
**cannot** speak Japanese; English ASR on JP-Audio is unreliable. **Conclusion:** a single off-the-shelf model can't
do it — but a LoRA that teaches *translate-then-speak* on real target-language audio fixes translation **and** the
voice at once. That's this fine-tune.

---

## 2. Task formulation

One **sequential** turn that emits translated text, then speech:

```
system : "Translate to English and speak."     (JA→EN)   |   "Translate to Japanese and speak."  (EN→JA)
user   : <source text>
assistant: <translated text> <|audio_start|> <target-language audio tokens>
```

At inference `generate_sequential` stays in TEXT modality (emits the translation), then emits `<|audio_start|>`
(token 128) and switches to AUDIO — so the app gets **both** the subtitle text (for transcript/minutes) and the
streamed speech. The training sample is built with `LFM2AudioChatMapper`, assistant content =
`[TextSegment(translation), AudioSegment(target_wav)]` (both supervised).

---

## 3. Data

### 3.1 Source (translations already existed)
`/data/lfm2.5/run-a/data` has per-clip metadata with pre-computed translations, so pairs are free:

| Direction | input text | target text | target audio | source set |
|---|---|---|---|---|
| JA→EN | `japanese_translation` | `english_transcription` | the EN wav | `english_only` |
| EN→JA | `english_translation` | `japanese_transcription` | the JA wav | `japanese_only` |

That gave the **v1** set: 300 + 300 = 600 pairs (target audio = the original **edge-tts** clips).

### 3.2 Augmentation with Kokoro-82M (voice diversity)
English audio was the weak side, so v2/v3 synthesized **fresh target audio with Kokoro-82M** (24 kHz, matches Mimi),
pulling JA↔EN parallel text from **all three** metadata sets (`english_only`, `japanese_only`, `code_switching`):

- **v2:** +1,600 Kokoro clips (1,100 EN one-voice + 500 JA) → 2,200 total.
- **v3 (deployed):** **2 distinct EN voices per JA→EN sentence** from a 16-voice English pool (+4 JA voices) →
  2,200 EN + 500 JA + the 600 edge-tts = **3,300 pairs** (3,069 train / 231 eval).

Keeping the edge-tts clips alongside Kokoro is deliberate: **two synthetic sources** reduce overfitting to one
synthesizer (per the cross-lingual-TTS literature; clean high-sample-rate audio matters for a TTS *target*).

### 3.3 Real audio (tried, reverted)
**v4** added 660 real FLEURS-validation clips (16 kHz, back-translated input). It **regressed audio quality badly**
(English round-trip 60→16) — the band-limited/noisy real audio taught dull/noisy Mimi targets. Lesson: for a TTS
*output* target, clean 24 kHz synthetic beats "real but noisy." **v4 discarded; v3 is final.**

### 3.4 Preprocess
`optionB_preprocess.py` → `LFM2AudioChatMapper` → HF dataset (`max_context 1024`). One pass on GPU; no examples
dropped at 1024.

---

## 4. Training

`train_lora.py` (single-GPU, `liquid_audio` collator). Identical LoRA recipe to the other adapters:

| Param | Value |
|---|---|
| peft type | LoRA |
| rank `r` / `alpha` / dropout | 16 / 32 / 0.05 |
| target modules (8) | `q_proj,k_proj,v_proj,out_proj` (attention), `in_proj` (SSM/conv mixer), `w1,w2,w3` (FFN) |
| trainable params | 12,410,880 (**0.85 %** of 1.47 B) |
| optimizer / lr | AdamW / 1e-4, cosine, warmup 50–120 |
| batch / ctx / precision | 8 / 1024 / bf16 |
| steps | v1 400 · v2 800 · **v3 1500** · v4 1800 |
| audio sampling at gen | `audio_temperature 0.8`, `audio_top_k 64` |

v3 val loss fell monotonically 1.86 → **1.396** (still descending — no overfit). ~2–8 min/run on one H100.

---

## 5. Evaluation

Two metrics, on **neutral FLEURS** (never trained on): **translation chrF** vs reference, and **audio
intelligibility** = round-trip ASR (EN-base judges English audio, clean JP judges Japanese) chrF vs the produced text.

| Variant | JA→EN translate | **English audio (RT)** | EN→JA translate | **Japanese audio (RT)** |
|---|---|---|---|---|
| base (no adapter) | ~2 | 0 | ~32 | 0 |
| v1 (600, edge) | 38.0 | 37.1 | 27.8 | 60.7 |
| v2 (+Kokoro 1-voice) | 37.6 | 55.2 | 26.5 | 68.7 |
| **v3 (+Kokoro 2-voice, deployed)** | **39.2** | **59.8** | **27.8** | **70.5** |
| v4 (+real FLEURS) | 40.7 | 16.2 ✗ | 29.2 | 49.1 ✗ |

**Performance improvements:** base→tuned is the headline — the base model can't translate-and-speak at all
(audio RT 0). English audio intelligibility climbed **37 → 55 → 59** as voice diversity grew; Japanese reached ~70.
Translation chrF is stable (the adapter targets audio, not text MT). On an *in-domain* held-out set v3 reads even
higher (JA→EN translate ≈ 74, EN audio RT ≈ 61). English audio is the weaker side at diminishing returns; further
gains would need clean 24 kHz studio English speech, not more synthetic voices — **which §8 (v6) then did.**

---

## 6. Integration

`LiveConfig.translating_tts_adapter` → the engine wraps the JA audio model as a `PeftModel` (adapter name
`default`). `AudioEngine.translate_speak_stream(text, tgt)` selects the prompt by target language, runs
`generate_sequential`, **yields the translation text first, then 24 kHz audio chunks** (same bounded-window streaming
as TTS). It's one of three adapters that share the base (see `finetuning-minutes.md` §6 and `system-architecture.md`).
Prompts: `tt_prompt_en` / `tt_prompt_ja`. Run: `serve_api.py --tt-adapter <…/optionD/checkpoints/v6_libritts/final>`.

---

## 7. Lessons

1. **Measure feasibility first** — the zero-shot study said which checkpoint could ever produce which language.
2. **Distill from existing labels** — the metadata's translations made data nearly free.
3. **Voice diversity via a second synthesizer (Kokoro) lifts the weak language**; clean 24 kHz > real-but-noisy for
   a TTS target (v4's regression).
4. **Sequential text-then-audio** keeps the subtitle/minutes text while speaking — no separate MT call.

---

## 8. v6 — real studio English (LibriTTS-R) → **current best**

Driven by a literature review:
the weak English voice is a data/sample-rate-match problem, and the fix is **clean 24 kHz studio English**, not more
synthetic voices (and v4 confirmed noisy 16 kHz FLEURS hurts). So v6 adds **LibriTTS-R** (real, 24 kHz = exact Mimi
SR, CC BY 4.0): each clip's English audio is a `ja→en` target, with the English transcript translated EN→JA
(`LFM2.5-1.2B-JP`) for the input side; audio peak-normalized. Mixed with the v3 set → **5,509 train** pairs
(`libritts_r 2363` real EN + `kokoro 2573` + `edge 574`). Script `optionD_libritts.py`; trained 1500 steps (same
LoRA recipe); adapter `optionD/checkpoints/v6_libritts/final`.

**Result (head-to-head vs v3 on neutral FLEURS, `optionD_eval.py`):**
| | v3 | **v6** | Δ |
|---|---|---|---|
| JA→EN translate | 38.3 | **43.2** | **+4.9** |
| English audio-RT | 57.2 | **59.9** | **+2.7** |
| EN→JA translate | 29.5 | 28.8 | −0.7 (noise) |
| JA audio-RT | 58.9 | 57.0 | −1.9 (noise) |

Real studio English lifted **JA→EN translation +4.9** (more natural English target text) and **English audio +2.7**,
Japanese ≈ tied. **v6 is the deployed best.** Next levers (not yet trained): real Japanese (Common Voice JA, `optionD_cvja.py`,
v7 data built but not trained), FLEURS-R, inference-time CFG, and code-switch synthesis — see research doc §"prioritized action plan".

