# 01 · Data Generation

Synthetic Japanese↔English speech and transcript generation, plus public-dataset downloads.

## asr-ft/ — synthetic ASR corpus (TTS)
- `generate_cs_data.py`, `generate_en_data.py`, `generate_ja_data.py` — generate code-switched,
  English-only and Japanese-only audio + transcription metadata.
- `translate_cs_english.py` — produce the English-side reference for CS utterances.
- `regenerate_missing.py` — backfill failed TTS generations.
- `download_fleurs.py`, `download_fleurs_500.py`, `download_cs_fleurs_byan.py` — pull FLEURS /
  CS-FLEURS (`byan/cs-fleurs`, `jpn-eng` slice) for training/eval material.

Output JSONL contract: `{"audio_file", "system": "Perform ASR.", "target", "type"}` with a
9-way task taxonomy (`cs_native`, `cs_ja`, `cs_en`, `en_*`, `ja_*`).

## tts-ft/ — code-switch transcript + bilingual TTS
- `00_smoke.py` — offline data-gen wiring test.
- `01_gen_transcripts.py` — code-switch transcripts (seeds + contrast pairs).
- `02_synthesize.py` — transcripts -> wav + manifest (bilingual TTS).
- `generate_cosyvoice_cs.py` — CosyVoice2 synthesis for commercial-clean CS audio.
- `download_mono.py` — monolingual JA/EN corpora for the anti-forgetting replay mix.

> tts-ft entrypoints import the `csmeeting` package — see `05-app-and-demo/kaigi-app/src`.
