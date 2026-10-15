# 02 · Preprocessing

Convert raw audio+text into the LFM2-Audio training format and curate the data mix.

## asr-ft/
- `preprocess_asr_data.py` — JSONL -> HuggingFace Arrow dataset (columns: `text`, `audio_in`,
  `audio_in_lens`, `audio_out`, `modality_flag`, `supervision_mask`).
- `combine_balanced.py` — assemble balanced native-ASR mixes
  (CS + up-weighted EN + JA) to limit monolingual forgetting.
- `prepare_training_data.py`, `validate_dataset.py`, `dry_run.py` — data prep, validation, smoke test.
- `augment_data.py` — MUSAN noise/music augmentation (3 variants per clip: noise@20dB, noise@13dB,
  music@20dB). Augmented audio itself is not committed — regenerate with this script.

## tts-ft/
- `preprocess_cs_asr.py`, `optionB_preprocess.py`, `optionC_preprocess.py` — preprocessing for the
  ASR / translate-TTS / minutes adapters.
- `mix_manifests.py`, `filter_lowdensity.py`, `convert_external_manifest.py`, `jecs_to_manifest.py` —
  manifest merging, low-density-switch filtering, cross-format conversion (JECS, teammate manifests).
- `optionB_build_data.py`, `optionB_kokoro_gen*.py`, `optionB_real_audio.py`, `prep_mt_data.py`,
  `augment_audio.py` — translate-TTS data building (Kokoro / real audio) and MT data prep.
