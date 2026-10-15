# Wiring this data into the LFM audio training kit

The kit (the Liquid4All LFM audio training kit)
audio track fine-tunes `LFM2.5-Audio-1.5B` by reading an iterator that yields `list[ChatMessage]`. We swap the kit's
default TTS iterator for our **code-switch ASR** iterator.

## Steps

1. Generate data in THIS repo:
   ```bash
   python scripts/01_gen_transcripts.py --n-template 300
   python scripts/02_synthesize.py            # real audio (edge-tts, needs network)
   ```
   You now have `data/synth/manifest.jsonl` + `data/synth/audio/*.wav`.

2. In the kit, make our package importable and point its iterator at the manifest. Two options:
   - copy `manifest.jsonl` + `audio/` into the kit (e.g. `kit/data/cs/`), or
   - `pip install -e <path-to-this-repo>` inside the kit's venv so `import csmeeting` works.

3. Edit the kit's `scripts/audio/train.py` — replace the body of `TrainingSamples.__iter__` with the snippet in
   [`train_iter_snippet.py`](train_iter_snippet.py). It's ~5 lines.

4. **Two things to change from the kit's TTS defaults:**
   - **System prompt / task:** the kit's iterator is TTS (`system="Perform TTS..."`, user=text, assistant=audio).
     Ours is **ASR**: `system="Perform ASR in japanese."`, **user=audio, assistant=text**. The snippet already does this.
   - **`max_context_length`:** the reference preprocessing uses `256`, which silently *drops* longer samples. Meeting
     utterances are short, so 256 is usually fine — but check the skip-warning count and raise it if needed
     (costs VRAM).

5. Start from the **`-JP`** base for Japanese audio competence:
   ```bash
   MODEL_ID=LiquidAI/LFM2.5-Audio-1.5B-JP \
   PUSH_TO_HUB=<you>/lfm25-audio-cs-jp \
     ./scripts/audio/launch_hf_job.sh
   ```
   (Full bf16 fine-tune, ~50 min on 1×A100 80GB at the reference recipe. Run `make smoke-audio` on a CUDA box first.)

6. **Replay mix (do NOT skip):** also feed monolingual Japanese (Common Voice JA) + a small English slice so the
   model gains the switch skill without forgetting English. Concatenate manifests, or extend the iterator to read
   several manifests.

## Why ASR (not TTS or S2S) for the fine-tune

The demo needs (1) code-switch **transcription**, (2) JA→EN **translation+TTS**, (3) minutes. Only (1) needs the
*new* capability the base lacks (JP↔EN switch handling). Translation+TTS can pipeline through the text models +
the base TTS voice, which already work. So fine-tune budget goes to ASR; the rest is inference wiring. (If time
allows, a second pass can add speech-translation samples — `system="Perform speech translation to english."` with
JA audio in / EN text out — using the same manifest plus an English `translation` field.)
