"""Configuration + direction routing for the bidirectional live translation cascade."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiveConfig:
    # --- audio models: each does ASR + TTS for its language ---
    ja_audio_model_id: str = "LiquidAI/LFM2.5-Audio-1.5B-JP"   # JA ASR + JA TTS (swap for the code-switch fine-tune)
    en_audio_model_id: str = "LiquidAI/LFM2.5-Audio-1.5B"      # EN ASR + EN TTS (US/UK voices)
    translator_model_id: str = "LiquidAI/LFM2.5-1.2B-JP"        # JA<->EN, both directions
    translator_adapter: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"

    # --- translating-TTS: a LoRA adapter on the JA audio model that translates AND speaks in one pass ---
    # When set, the cross-lingual path (JA text -> EN audio, EN text -> JA audio) runs on the JA audio model
    # itself (adapter ON), replacing the separate text translator. The adapter is disabled for ASR / plain TTS
    # (it degrades ASR), so one model serves every role. See memory: kaigi-translating-tts (transtts_lora_v3).
    translating_tts_adapter: str | None = None
    load_text_model: bool = True            # load LFM2.5-1.2B-JP (needed for minutes; translation no longer uses it)

    # --- GGUF ASR: route ASR to the llama.cpp liquid-audio server (code-switch fine-tune as GGUF, CPU) ---
    # When set, AudioEngine.transcribe() POSTs audio to this OpenAI-compatible server instead of running the
    # PyTorch audio model for ASR. One prompt ("Perform ASR.") handles JA+EN. Launch the server with
    # scripts/gguf_asr_server.sh (docker). See memory: kaigi-gguf-asr-runner.
    asr_gguf_url: str | None = None         # e.g. "http://localhost:8090"
    asr_prompt_gguf: str = "Perform ASR."   # ASR prompt (GGUF runner + GPU-LoRA path both use this)

    # GPU PyTorch ASR via the code-switch LoRA adapter (base ja_audio_model + this adapter). Fast and immune to
    # host CPU contention — the right ASR path on a GPU box (the GGUF/CPU path is for on-device). Takes precedence
    # is NOT given over GGUF: if asr_gguf_url is also set, GGUF wins. One "Perform ASR." prompt handles JA+EN.
    asr_lora: str | None = None             # e.g. ".../checkpoints/lfm_lora/final"

    # Minutes LoRA on the audio backbone (loaded as a 2nd adapter on the JA audio model). When set, minutes
    # run text->text on the backbone instead of the separate LFM2.5-1.2B-JP — set load_text_model=False to drop it.
    minutes_lora: str | None = None         # e.g. ".../optionC/checkpoints/minutes_lora_v3/final"

    # --- audio backend: "torch" (PyTorch/liquid_audio, needs CUDA) | "onnx" (onnxruntime, CPU-only, LFM model) ---
    audio_backend: str = "torch"
    ja_onnx_dir: str | None = None          # exported JA LFM2.5-Audio ONNX dir (used when audio_backend="onnx")
    en_onnx_dir: str | None = None          # exported EN LFM2.5-Audio ONNX dir
    onnx_precision: str = "q4"              # fp16 | q4 | q8
    onnx_export_src: str | None = None      # path to onnx-export/src, so `import liquidonnx` works without pip install
    # onnxruntime execution provider for the ONNX backend:
    #   "cpu"  — CPUExecutionProvider (default, TTS is turn-based: RTF ~2.6)
    #   "dml"  — DmlExecutionProvider → Ryzen/Radeon iGPU; the path to push TTS toward RTF<1 (needs onnxruntime-directml)
    #   "cuda" — CUDAExecutionProvider (if an NVIDIA GPU is present)
    #   "auto" — prefer dml, then cuda, then cpu (whatever onnxruntime reports as available)
    onnx_ep: str = "cpu"

    # torch audio generation mode for TTS / the spoken path (AudioEngine only):
    #   "sequential"  — model.generate_sequential (default; Liquid's documented ASR/TTS mode)
    #   "interleaved" — model.generate_interleaved (lower time-to-first-audio; designed for speech-to-speech, see §C)
    tts_generation_mode: str = "sequential"

    # --- task prompts (the LFM2.5-Audio task is set by the system message), per language ---
    asr_prompt_ja: str = "Perform ASR in japanese."
    asr_prompt_en: str = "Perform ASR."
    tts_prompt_ja: str = "Perform TTS in japanese."
    tts_prompt_en: str = "Perform TTS. Use the US female voice."

    # translating-TTS task prompts (used with translating_tts_adapter): text in -> translated text + speech out
    tt_prompt_en: str = "Translate to English and speak."     # input JA -> EN audio
    tt_prompt_ja: str = "Translate to Japanese and speak."    # input EN -> JA audio

    # direction: "auto" (detect per utterance) | "ja2en" | "en2ja"
    direction: str = "auto"

    # Segment the incoming recording on pauses (VAD) and transcribe each utterance, then join — avoids the
    # long-blob ASR failure (the model drops/merges content when given a whole multi-sentence recording at once).
    vad_segment_input: bool = True

    # --- generation budgets ---
    max_new_tokens_asr: int = 256
    max_new_tokens_mt: int = 256
    max_new_tokens_tts: int = 1024
    tts_audio_temperature: float = 0.8
    tts_audio_top_k: int = 64

    sample_rate_out: int = 24000   # LFM2.5-Audio decodes at 24 kHz

    # --- VAD / segmentation (translate on the pause) ---
    vad_backend: str = "energy"    # "energy" (no deps) | "silero"
    vad_sample_rate: int = 16000
    vad_frame_ms: int = 30
    vad_min_speech_ms: int = 300
    vad_min_silence_ms: int = 700
    vad_energy_threshold: float = 0.012


def resolve_direction(direction: str, src_lang: str | None = None) -> tuple[str, str]:
    """Return (source_lang, target_lang) for a direction. 'auto' uses the detected src_lang."""
    if direction == "ja2en":
        return "ja", "en"
    if direction == "en2ja":
        return "en", "ja"
    src = src_lang or "ja"               # auto: detected source decides
    return src, ("en" if src == "ja" else "ja")
