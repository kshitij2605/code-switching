"""The Kaigi assistant: bidirectional ASR + MT + TTS + minutes, LFM-only, with optional auto direction.

Audio backend is selectable: "torch" (PyTorch/liquid_audio, needs CUDA) or "onnx" (onnxruntime, **CPU-only**, the
LFM2.5-Audio model). Auto direction uses the LFM ASR's transcript script (no external LID model). The MT and minutes
stages share one text model (LFM2.5-1.2B-JP), which runs on `cfg.device` (cuda or cpu).
"""

from __future__ import annotations

import numpy as np

from csmeeting.live.config import LiveConfig, resolve_direction
from csmeeting.live.lid import lang_of


class Assistant:
    def __init__(self, cfg: LiveConfig | None = None, mock: bool = False):
        self.cfg = cfg or LiveConfig()
        self.mock = mock
        self.transcript_lines: list[str] = []

        if mock:
            from csmeeting.live.engine import MockAudioEngine
            self.engine = MockAudioEngine(self.cfg)
            self._torch = None
            return

        # audio backend
        if self.cfg.audio_backend == "onnx":
            from csmeeting.live.onnx_engine import OnnxAudioEngine
            self.engine = OnnxAudioEngine(self.cfg)
        else:
            from csmeeting.live.engine import AudioEngine
            self.engine = AudioEngine(self.cfg)

        # Translation runs on the audio model when a translating-TTS adapter is configured (one model does
        # translate+speak). The separate text model is then only needed for minutes; skip it if not wanted.
        self.use_audio_translator = getattr(self.engine, "has_tt_adapter", False)

        import torch
        self._torch = torch
        self.tok = self.lm = None
        if self.cfg.load_text_model:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.cfg.translator_model_id)
            on_cuda = self.cfg.device == "cuda"
            dtype = getattr(torch, self.cfg.dtype, torch.bfloat16) if on_cuda else torch.float32
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.cfg.translator_model_id, device_map="cuda" if on_cuda else None, dtype=dtype,
            )
            if not on_cuda:
                self.lm = self.lm.to(self.cfg.device)
            self.lm.eval()

    def _gen(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        torch = self._torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.lm.device)
        plen = ids["input_ids"].shape[1]
        with torch.no_grad():
            out = self.lm.generate(**ids, do_sample=True, temperature=temperature, top_p=0.9,
                                   repetition_penalty=1.05, max_new_tokens=max_new_tokens)
        txt = self.tok.decode(out[0][plen:], skip_special_tokens=True).strip()
        if "</think>" in txt:
            txt = txt.split("</think>", 1)[1].strip()
        return txt

    def translate(self, text: str, src: str, tgt: str) -> str:
        if self.mock:
            return f"[mock-{tgt}] {text}"
        if self.lm is None:
            raise RuntimeError("text model not loaded (load_text_model=False); use the translating-TTS path")
        from csmeeting.mt.prompt import build_messages
        return self._gen(build_messages(text, src, tgt), self.cfg.max_new_tokens_mt, 0.2)

    def _asr_text(self, wav: np.ndarray, sr: int, lang: str) -> str:
        """ASR with VAD segmentation: split the recording on pauses and transcribe each utterance, then join.
        Giving the model a whole multi-sentence recording at once makes it drop/merge content; per-utterance ASR
        is clean. Falls back to a single call for short clips / when segmentation is off or unavailable."""
        if self.mock or not getattr(self.cfg, "vad_segment_input", True):
            return self.engine.transcribe(wav, sr, lang)
        try:
            from csmeeting.live.vad import segment_array
            segs = segment_array(wav, sr, self.cfg)
        except Exception:  # noqa: BLE001
            segs = None
        if not segs:                                          # short clip / no pause → one call
            return self.engine.transcribe(wav, sr, lang)
        parts = [self.engine.transcribe(s, self.cfg.vad_sample_rate, lang) for s in segs]
        return " ".join(p.strip() for p in parts if p.strip())

    def _resolve(self, wav: np.ndarray, sr: int, direction: str) -> tuple[str, str, str]:
        """Return (src, tgt, src_text). For 'auto', detect language from the LFM ASR transcript script."""
        if direction == "auto":
            text = self._asr_text(wav, sr, "ja")     # JA-side (code-switch) model handles JA + embedded EN
            src = lang_of(text)
            # English utterance -> re-ASR with the EN model, unless ASR is language-agnostic (GGUF / GPU-LoRA path:
            # one "Perform ASR." prompt handles both → re-transcribing would just repeat the call).
            one_call_asr = getattr(self.engine, "uses_gguf_asr", False) or getattr(self.engine, "uses_gpu_asr", False)
            if src == "en" and not one_call_asr:
                text = self._asr_text(wav, sr, "en")
            return src, ("en" if src == "ja" else "ja"), text
        src, tgt = resolve_direction(direction)
        return src, tgt, self._asr_text(wav, sr, src)

    def process_stream(self, wav: np.ndarray, sr: int, direction: str | None = None):
        """Yields (src_text, translation, (sr, chunk) | None, src, tgt). On CPU, audio is emitted after full synthesis."""
        direction = direction or self.cfg.direction
        src, tgt, src_text = self._resolve(wav, sr, direction)
        if src_text:
            self.transcript_lines.append(src_text)

        if self.use_audio_translator and not self.mock:
            # one model: translate + speak in a single pass (text first, then streamed audio)
            translation = ""
            for kind, payload in self.engine.translate_speak_stream(src_text, tgt):
                if kind == "text":
                    translation = payload
                    yield src_text, translation, None, src, tgt
                elif payload is not None and len(payload) > 0:
                    yield src_text, translation, (self.cfg.sample_rate_out, payload), src, tgt
            return

        translation = self.translate(src_text, src, tgt)
        yield src_text, translation, None, src, tgt
        for chunk in self.engine.synthesize_stream(translation, tgt):
            if chunk is not None and len(chunk) > 0:
                yield src_text, translation, (self.cfg.sample_rate_out, chunk), src, tgt

    def process(self, wav: np.ndarray, sr: int, direction: str | None = None):
        direction = direction or self.cfg.direction
        src, tgt, src_text = self._resolve(wav, sr, direction)
        if src_text:
            self.transcript_lines.append(src_text)

        if self.use_audio_translator and not self.mock:
            translation, chunks = "", []
            for kind, payload in self.engine.translate_speak_stream(src_text, tgt):
                if kind == "text":
                    translation = payload
                elif payload is not None and len(payload) > 0:
                    chunks.append(payload)
            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            return src_text, translation, audio, self.cfg.sample_rate_out, src, tgt

        translation = self.translate(src_text, src, tgt)
        audio, out_sr = self.engine.synthesize(translation, tgt)
        return src_text, translation, audio, out_sr, src, tgt

    def minutes(self) -> str:
        text = self.transcript()
        if not text:
            return "_(no transcript yet)_"
        if self.mock:
            return "## 要約 / Summary\n- (mock minutes)\n## アクションアイテム / Action Items\n- [担当: —] —"
        if getattr(self.engine, "has_minutes_lora", False):     # minutes on the audio backbone (no text model)
            from csmeeting.minutes.prompt import SYSTEM_PROMPT
            return self.engine.generate_minutes(text, SYSTEM_PROMPT)
        if self.lm is None:
            return "_(minutes need the text model — start with load_text_model=True or a minutes LoRA)_"
        from csmeeting.minutes.prompt import build_messages
        return self._gen(build_messages(text), 768, 0.2)

    def transcript(self) -> str:
        return "\n".join(self.transcript_lines)

    def reset(self) -> None:
        self.transcript_lines = []
