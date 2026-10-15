# Live JA→EN translation — cascade design + MT notes

> **⚡ Update — translation is now a fine-tuned adapter on the audio model, not the separate text model.**
> The "test-first, ship the base `LFM2.5-1.2B-JP`" verdict below was the *original* MT stage. It has since been
> replaced: a **translate-and-speak LoRA** on `LFM2.5-Audio-1.5B-JP` now does JA↔EN translation **and** the spoken
> output in one pass (the text translator is dropped). Full details + data + results:
> [`../documentation/finetuning-translate-tts.md`](../documentation/finetuning-translate-tts.md). The notes below
> remain useful for the cascade rationale and the original MT baseline.

## Architecture (decided: cascade, per-utterance, streamed)

```
JA speech ─► [VAD: segment an utterance on pause]
          ─► ASR   : LFM2.5-Audio (JA / code-switch → JA text)   ──► live transcript
          ─► MT    : LFM2.5-1.2B-JP (JA text → EN text)          ──► EN subtitle
          ─► TTS   : LFM2.5-Audio English voice (EN text → audio) ─► listener hears English
                                                                    └─► both texts feed minutes
```

Why cascade over direct speech-to-speech (for JA→EN): the real-time bottleneck is **linguistic** (Japanese is
verb-final → you must wait ~a clause to translate), not the number of model hops, so S2S's latency edge is small;
meanwhile you **already need the text** (transcription + minutes), the ASR stage **is** the code-switch fine-tune
you're building, and translation quality is best controlled as its own stage. See the conversation rationale.

## Do we need to fine-tune the translator? → **Test first.**

We are **not** claiming the base model can't translate — it almost certainly produces competent JA→EN zero-shot
(`LFM2.5-1.2B-JP` is JP-optimized + multilingual). Translation just isn't a *benchmarked* task on the card, so we
measure instead of assuming.

- **Default:** base `LFM2.5-1.2B-JP` + a translation prompt, **no fine-tune** (`src/csmeeting/mt/prompt.py`).
- **Measure:** `python scripts/eval_translation.py` → chrF / BLEU on FLEURS ja→en + your in-domain set.
- **Fine-tune ONLY if** it underperforms on: business/keigo register, terminology (product/company names), output
  consistency/format, or you need it smaller/faster. Supported path = Liquid cookbook `cpt_translation_with_unsloth.ipynb`;
  `python scripts/prep_mt_data.py` builds **commercial-safe** SFT data (JESC + Tatoeba + FLEURS).
- **Note:** code-switched input (JA + embedded English) is *easier* to translate, not harder — not a reason to fine-tune.

**Decision rule:** if base chrF on FLEURS ja→en is in a healthy range *and* in-domain spot-checks read well → ship the
base translator. Otherwise fine-tune the MT stage only.

## Components & checkpoints

| Stage | Model | Prompt / mode | Notes |
|---|---|---|---|
| ASR | LFM2.5-Audio (fine-tuned from `-JP`; base `-JP` works until then) | `"Perform ASR in japanese."`, `generate_sequential` | code-switch aware after fine-tune |
| MT | `LFM2.5-1.2B-JP` (base; optional LoRA) | text chat, translation prompt | test-first |
| TTS | LFM2.5-Audio **base English** (`LFM2.5-Audio-1.5B`) | `"Perform TTS. Use the US female voice."`, `generate_sequential` | base has US/UK voices; `-JP` has only a JP voice |

### One vs two audio models (deployment decision)
English TTS needs the **English** audio checkpoint; JA/CS ASR is best from the **`-JP`-based** fine-tune. Two options:
- **Two models** (default here): ASR = fine-tuned-from-`-JP`, TTS = base English. Best quality, ~2× memory (quantize to
  GGUF ~1 GB each).
- **One model:** fine-tune ASR **from the English base** + keep its TTS → one checkpoint does JA-CS-ASR *and* EN-TTS.
  Simpler deploy; verify JA ASR doesn't regress vs the `-JP`-based fine-tune and that TTS isn't forgotten. Decide after
  the ASR fine-tune lands.

## Real-time policy

**Segment by VAD / utterance (translate on the pause)** — for JA→EN this is *both* simpler and higher quality
(waiting for the pause gives you the verb → a grammatical translation). Perceived lag ≈ 1–2 s after the speaker stops,
matching Pocketalk/VoiceTra-class UX. Word-level streaming before the verb produces garbage for this pair.

**Measured latency (H100, GPU 0):** ASR ≈ 0.3 s, MT ≈ 0.1 s, and **streaming TTS first-audio ≈ 0.46 s** (vs 2.8 s if
you wait for the whole clip — 6× faster; TTS generates at **RTF ≈ 0.4** so playback never starves). Listener hears
English ≈ **0.9 s** after the speaker pauses. Streaming TTS = `AudioEngine.synthesize_stream` (decode + emit each
80 ms Mimi frame in a bounded causal window, holding back 2 frames for the ISTFT edge, filtering the reserved
terminal frame); the demo app plays it via `gr.Audio(streaming=True)`. On a Ryzen laptop per-frame generation is
slower (higher RTF) → quantize (GGUF/ONNX) to keep RTF < 1; the streaming architecture is what makes it real-time.

## Bidirectional (JA↔EN) + auto language-ID

The same model stack runs **both directions** — no extra models. Each audio model does ASR *and* TTS for its
language; the text model translates both ways. Routing by `LiveConfig.direction`:

| Direction | ASR | MT | TTS |
|---|---|---|---|
| JA→EN | JP audio | LFM2.5-1.2B-JP (JA→EN) | English audio |
| EN→JA | English audio | LFM2.5-1.2B-JP (EN→JA) | JP audio |

`direction` ∈ `auto` / `ja2en` / `en2ja`. **auto** is **LFM-only** (no Whisper): transcribe with the LFM ASR and
classify by script (`lid.lang_of` — any kana/kanji ⇒ ja; pure English has none), then translate to the other
language. Validated on real weights: EN ASR (the English model's native task), EN→JA MT (natural Japanese), JA TTS
(streaming TTFA 0.31 s on GPU). `Assistant.process_stream(wav, sr, direction)` is the entry point; the demo app has
a direction toggle.

**Backend:** `LiveConfig.audio_backend="onnx"` + `device="cpu"` runs the whole cascade **CPU-only with LFM models**
(onnxruntime via `liquidonnx`) — ASR real-time, TTS turn-based; see [`../documentation/system-resources.md`](../documentation/system-resources.md) §B.

## Run

```bash
pip install -e ".[live]"                 # liquid-audio + transformers + sounddevice
python scripts/live_translate.py --mock              # offline wiring test (no models)
python scripts/live_translate.py --wav meeting.wav   # translate a file
python scripts/live_translate.py --mic               # microphone (needs sounddevice)

# MT decision tooling
pip install -e ".[mt]"
python scripts/eval_translation.py --n 200           # base translator chrF/BLEU on FLEURS ja->en
python scripts/prep_mt_data.py --out data/mt/train.jsonl   # only if you decide to fine-tune
```
