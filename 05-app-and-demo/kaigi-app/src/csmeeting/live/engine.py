"""Audio engine: holds the JA-side and EN-side LFM2.5-Audio models; each does ASR + TTS for its language.

Routing by language: source 'ja' → JA model ASR; source 'en' → EN model ASR; target 'en' → EN model TTS;
target 'ja' → JA model TTS. All methods take a `lang` arg. Streaming TTS yields 24 kHz chunks (~0.3 s to first
audio). Verified `liquid_audio` API (ChatState + generate_sequential).
"""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from csmeeting.live.config import LiveConfig

_SPECIAL = re.compile(r"<\|[^|]*\|>")     # strip <|im_end|> etc. leaked by the ASR decode
_SPF = 24000 * 80 // 1000                  # 1920 samples / 80 ms Mimi frame


class AudioEngine:
    """Loads both audio models (JA-side + EN-side). Requires `pip install -e '.[live]'` + a CUDA GPU."""

    def __init__(self, cfg: "LiveConfig"):
        import torch
        from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

        self.cfg = cfg
        self._torch = torch
        self._ChatState = ChatState

        def _load(model_id: str):
            proc = LFM2AudioProcessor.from_pretrained(model_id, device=cfg.device).eval()
            model = LFM2AudioModel.from_pretrained(model_id).eval()
            try:
                model = model.to(cfg.device)
            except Exception:
                pass
            return proc, model

        self.ja_processor, self.ja_model = _load(cfg.ja_audio_model_id)
        # The EN audio model is only needed for EN ASR and EN TTS. With GGUF ASR (handles EN) AND the
        # translating-TTS adapter (does EN audio via the JA model), it's redundant — skip it to save VRAM.
        skip_en = bool((cfg.asr_gguf_url or cfg.asr_lora) and cfg.translating_tts_adapter)
        en_alias = skip_en or (cfg.en_audio_model_id == cfg.ja_audio_model_id)
        if en_alias:
            self.en_processor, self.en_model = self.ja_processor, self.ja_model
            if skip_en:
                print("[engine] EN audio model not loaded (GGUF ASR + translating-TTS adapter cover all roles)")
        else:
            self.en_processor, self.en_model = _load(cfg.en_audio_model_id)

        # Translating-TTS adapter: attach to the JA audio model so it can translate+speak in one pass.
        # Kept as a PeftModel (not merged) so it can be toggled — the adapter degrades ASR, so it is disabled
        # for transcribe()/synthesize() and enabled only for translate_speak_stream(). One model, every role.
        self.has_tt_adapter = False
        if cfg.translating_tts_adapter:
            from peft import PeftModel
            self.ja_model = PeftModel.from_pretrained(self.ja_model, cfg.translating_tts_adapter).eval()
            if en_alias:   # EN aliased to JA (skip_en or same id): keep refs in sync after wrapping
                self.en_model = self.ja_model
            self.has_tt_adapter = True

        # Optional GGUF ASR backend (llama.cpp liquid-audio server). When set, transcribe() uses it instead
        # of the PyTorch audio model — one prompt handles JA+EN, so the EN audio model isn't needed for ASR.
        self._gguf_asr = None
        self.uses_gguf_asr = False
        if cfg.asr_gguf_url:
            from csmeeting.live.gguf_asr import GgufAsrClient
            self._gguf_asr = GgufAsrClient(cfg.asr_gguf_url, prompt=cfg.asr_prompt_gguf,
                                           max_tokens=cfg.max_new_tokens_asr)
            self.uses_gguf_asr = True

        # GPU PyTorch ASR via the code-switch LoRA. Fast and immune to host CPU contention (the GGUF/CPU path
        # thrashes on a saturated shared box). Loaded as the "asr" adapter on the SHARED JA base when one exists
        # (single base for asr/tts/minutes, switched per task); else a dedicated base. "Perform ASR." → JA+EN.
        self._asr_model = None        # dedicated ASR base, only when no shared PeftModel exists
        self._asr_adapter = None      # name of the asr adapter on the shared base, if any
        self.uses_gpu_asr = False
        if cfg.asr_lora and not cfg.asr_gguf_url:
            from peft import PeftModel
            if isinstance(self.ja_model, PeftModel):                 # share the base
                self.ja_model.load_adapter(cfg.asr_lora, adapter_name="asr")
                self._asr_adapter = "asr"
                print(f"[engine] GPU ASR as 'asr' adapter on shared base: {cfg.asr_lora}")
            else:                                                    # no other adapter → dedicated base
                asr_base = LFM2AudioModel.from_pretrained(cfg.ja_audio_model_id).eval()
                try:
                    asr_base = asr_base.to(cfg.device)
                except Exception:  # noqa: BLE001
                    pass
                self._asr_model = PeftModel.from_pretrained(asr_base, cfg.asr_lora).eval()
                print(f"[engine] GPU ASR via dedicated base: {cfg.asr_lora}")
            self.uses_gpu_asr = True

        # Minutes LoRA as a 2nd adapter on the JA audio model (multi-adapter) → minutes on the backbone,
        # so the separate text model can be dropped. The translating-TTS adapter (if any) stays the primary.
        self.has_minutes_lora = False
        self._primary_adapter = "default"
        if cfg.minutes_lora:
            from peft import PeftModel
            if isinstance(self.ja_model, PeftModel):
                self.ja_model.load_adapter(cfg.minutes_lora, adapter_name="minutes")
                self._primary_adapter = "default"          # the translating-TTS adapter
            else:
                self.ja_model = PeftModel.from_pretrained(self.ja_model, cfg.minutes_lora, adapter_name="minutes").eval()
                if en_alias:
                    self.en_model = self.ja_model
                self._primary_adapter = "minutes"          # no TTS adapter; minutes is the only one
            self.has_minutes_lora = True
            print(f"[engine] minutes LoRA loaded as 2nd adapter: {cfg.minutes_lora}")

        # Whether multiple LoRA adapters share self.ja_model (so set_adapter must select per task).
        self._switch_adapters = bool(self._asr_adapter) or self.has_minutes_lora

        # warm up each detokenizer (cold first decode is ~0.6 s; warm ~5 ms)
        for proc in {id(self.ja_processor): self.ja_processor, id(self.en_processor): self.en_processor}.values():
            try:
                with torch.no_grad():
                    proc.decode(torch.zeros(1, 8, 8, dtype=torch.long, device=cfg.device))
            except Exception:  # noqa: BLE001
                pass

    def _route(self, lang: str):
        """Return (processor, model, asr_prompt, tts_prompt) for a language."""
        if lang == "ja":
            return self.ja_processor, self.ja_model, self.cfg.asr_prompt_ja, self.cfg.tts_prompt_ja
        return self.en_processor, self.en_model, self.cfg.asr_prompt_en, self.cfg.tts_prompt_en

    def _plain_ctx(self, model):
        """Disable the translating-TTS adapter for plain ASR/TTS (it degrades them); no-op otherwise."""
        if self.has_tt_adapter and model is self.ja_model:
            return self.ja_model.disable_adapter()
        return contextlib.nullcontext()

    def _tts_tokens(self, model, chat):
        """Yield generation tokens for a TTS turn, dispatching on cfg.tts_generation_mode.

        "interleaved" -> model.generate_interleaved (lower time-to-first-audio; designed for speech-to-speech, so it
        may interleave spurious text tokens on a fixed-text TTS turn — callers keep only audio frames). "sequential"
        (default) -> model.generate_sequential, Liquid's documented ASR/TTS mode. Both share the same signature, so
        the audio-frame filter downstream is identical.
        """
        gen = model.generate_interleaved if self.cfg.tts_generation_mode == "interleaved" else model.generate_sequential
        return gen(
            **chat, max_new_tokens=self.cfg.max_new_tokens_tts,
            audio_temperature=self.cfg.tts_audio_temperature, audio_top_k=self.cfg.tts_audio_top_k,
        )

    def _transcribe_gpu(self, wav: np.ndarray, sr: int) -> str:
        """ASR via the GPU PyTorch LoRA (language-agnostic 'Perform ASR.'). Uses the dedicated ASR base, or the
        shared base with the 'asr' adapter active (restoring the primary adapter after)."""
        torch = self._torch
        proc = self.ja_processor
        if self._asr_model is not None:
            model, switch = self._asr_model, False
        else:
            model, switch = self.ja_model, True
            model.set_adapter(self._asr_adapter)
        wt = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        if wt.ndim == 1:
            wt = wt.unsqueeze(0)
        chat = self._ChatState(proc)
        chat.new_turn("system"); chat.add_text(self.cfg.asr_prompt_gguf); chat.end_turn()
        chat.new_turn("user"); chat.add_audio(wt, sr); chat.end_turn()
        chat.new_turn("assistant")
        ids: list[int] = []
        try:
            with torch.no_grad():
                for t in model.generate_sequential(**chat, max_new_tokens=self.cfg.max_new_tokens_asr):
                    if t.numel() == 1:
                        v = int(t.view(-1)[0])
                        if v in (7, 128):     # <|im_end|> / <|audio_start|>
                            break
                        ids.append(v)
        finally:
            if switch:
                model.set_adapter(self._primary_adapter)
        # decode the WHOLE token sequence at once — per-token decode splits rare kanji into '�'
        txt = proc.text.decode(torch.tensor(ids, dtype=torch.long)) if ids else ""
        return _SPECIAL.sub("", txt).strip()

    def transcribe(self, wav: np.ndarray, sr: int, lang: str) -> str:
        """Audio (in language `lang`) -> text. Routes to GGUF server or GPU-LoRA model when configured."""
        if self._gguf_asr is not None:
            return self._gguf_asr.transcribe(wav, sr, lang)
        if self.uses_gpu_asr:
            return self._transcribe_gpu(wav, sr)
        torch = self._torch
        proc, model, asr_prompt, _ = self._route(lang)
        wt = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        if wt.ndim == 1:
            wt = wt.unsqueeze(0)
        chat = self._ChatState(proc)
        chat.new_turn("system"); chat.add_text(asr_prompt); chat.end_turn()
        chat.new_turn("user"); chat.add_audio(wt, sr); chat.end_turn()
        chat.new_turn("assistant")
        ids: list[int] = []
        with torch.no_grad(), self._plain_ctx(model):
            for t in model.generate_sequential(**chat, max_new_tokens=self.cfg.max_new_tokens_asr):
                if t.numel() == 1:
                    v = int(t.view(-1)[0])
                    if v in (7, 128):
                        break
                    ids.append(v)
        txt = proc.text.decode(torch.tensor(ids, dtype=torch.long)) if ids else ""   # whole-seq decode
        return _SPECIAL.sub("", txt).strip()

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        """Text -> full 24 kHz waveform in language `lang` (batch; use synthesize_stream for low latency)."""
        torch = self._torch
        proc, model, _, tts_prompt = self._route(lang)
        chat = self._ChatState(proc)
        chat.new_turn("system"); chat.add_text(tts_prompt); chat.end_turn()
        chat.new_turn("user"); chat.add_text(text); chat.end_turn()
        chat.new_turn("assistant")
        out = []
        with torch.no_grad(), self._plain_ctx(model):
            for t in self._tts_tokens(model, chat):
                if t.numel() > 1 and bool((t < 2048).all()):
                    out.append(t)
        if not out:
            return np.zeros(0, dtype=np.float32), self.cfg.sample_rate_out
        wav = proc.decode(torch.stack(out, 1).unsqueeze(0))[0].detach().cpu().float().numpy()
        if wav.ndim > 1:
            wav = wav[0]
        return wav.astype(np.float32), self.cfg.sample_rate_out

    def synthesize_stream(self, text: str, lang: str, chunk_frames: int = 8, ctx: int = 32, lookahead: int = 2):
        """Streaming TTS in language `lang`: yield 24 kHz float32 chunks as generated (~0.3 s to first audio).

        Decode a bounded window frames[finalized-ctx : n] every `chunk_frames`, emit only newly finalized frames
        (hold back `lookahead` for the ISTFT edge), filtering the reserved terminal frame (code >=2048).
        """
        torch = self._torch
        proc, model, _, tts_prompt = self._route(lang)
        chat = self._ChatState(proc)
        chat.new_turn("system"); chat.add_text(tts_prompt); chat.end_turn()
        chat.new_turn("user"); chat.add_text(text); chat.end_turn()
        chat.new_turn("assistant")

        def _decode(fr):
            return proc.decode(torch.stack(fr, 1).unsqueeze(0))[0].detach().cpu().float().numpy()

        frames: list = []
        finalized = 0
        with torch.no_grad(), self._plain_ctx(model):
            for t in self._tts_tokens(model, chat):
                if t.numel() > 1 and bool((t < 2048).all()):
                    frames.append(t)
                    n = len(frames)
                    if n % chunk_frames == 0 and n - finalized > lookahead:
                        ws = max(0, finalized - ctx)
                        wav = _decode(frames[ws:n])
                        lo, hi = (finalized - ws) * _SPF, (n - lookahead - ws) * _SPF
                        if hi > lo:
                            yield wav[lo:hi].astype(np.float32)
                            finalized = n - lookahead
            if len(frames) > finalized:
                ws = max(0, finalized - ctx)
                tail = _decode(frames[ws:])[(finalized - ws) * _SPF:]
                if len(tail) > 0:
                    yield tail.astype(np.float32)

    def translate_speak_stream(self, text: str, tgt: str, chunk_frames: int = 8, ctx: int = 32, lookahead: int = 2):
        """One-pass translate+TTS on the JA audio model + translating-TTS adapter (adapter ON).

        The model emits the translated TEXT first, then `<|audio_start|>` and the speech. Yields
        ("text", str) once the translation is complete (for the subtitle / minutes), then ("audio", chunk)
        24 kHz float32 chunks streamed with the same bounded-window decode as synthesize_stream.
        Requires `translating_tts_adapter`; replaces the separate text translator for the cross-lingual path.
        """
        if not self.has_tt_adapter:
            raise RuntimeError("translate_speak_stream requires cfg.translating_tts_adapter")
        torch = self._torch
        proc, model = self.ja_processor, self.ja_model
        if self._switch_adapters:                       # multi-adapter: make the TTS adapter active
            model.set_adapter(self._primary_adapter)
        tts_prompt = self.cfg.tt_prompt_en if tgt == "en" else self.cfg.tt_prompt_ja
        chat = self._ChatState(proc)
        chat.new_turn("system"); chat.add_text(tts_prompt); chat.end_turn()
        chat.new_turn("user"); chat.add_text(text); chat.end_turn()
        chat.new_turn("assistant")

        def _decode(fr):
            return proc.decode(torch.stack(fr, 1).unsqueeze(0))[0].detach().cpu().float().numpy()

        frames: list = []
        finalized = 0
        tids: list[int] = []          # translated-text token ids (decoded whole at the end — avoids '�' on split kanji)
        text_done = False
        in_audio = False

        def _txt():
            return _SPECIAL.sub("", proc.text.decode(torch.tensor(tids, dtype=torch.long)) if tids else "").strip()
        with torch.no_grad():
            for t in model.generate_sequential(
                **chat, max_new_tokens=self.cfg.max_new_tokens_tts,
                audio_temperature=self.cfg.tts_audio_temperature, audio_top_k=self.cfg.tts_audio_top_k,
            ):
                if t.numel() == 1:
                    tid = int(t.view(-1)[0])
                    if tid == 128:                       # <|audio_start|> -> switch to speech
                        in_audio = True
                        if not text_done:
                            yield "text", _txt()
                            text_done = True
                        continue
                    if tid == 7:                         # <|im_end|>
                        break
                    if not in_audio:
                        tids.append(tid)
                elif t.numel() > 1 and bool((t < 2048).all()):
                    frames.append(t)
                    n = len(frames)
                    if n % chunk_frames == 0 and n - finalized > lookahead:
                        ws = max(0, finalized - ctx)
                        wav = _decode(frames[ws:n])
                        lo, hi = (finalized - ws) * _SPF, (n - lookahead - ws) * _SPF
                        if hi > lo:
                            yield "audio", wav[lo:hi].astype(np.float32)
                            finalized = n - lookahead
            if not text_done:                            # no audio was produced — still emit the translation
                yield "text", _txt()
            if len(frames) > finalized:
                ws = max(0, finalized - ctx)
                tail = _decode(frames[ws:])[(finalized - ws) * _SPF:]
                if len(tail) > 0:
                    yield "audio", tail.astype(np.float32)


    def generate_minutes(self, transcript: str, system_prompt: str, max_new_tokens: int = 768) -> str:
        """Minutes via the audio backbone + minutes LoRA (text->text). Whole-sequence decode (per-token splits
        rare kanji). Activates the 'minutes' adapter, restores the primary (translating-TTS) adapter after."""
        if not self.has_minutes_lora:
            raise RuntimeError("generate_minutes requires cfg.minutes_lora")
        torch = self._torch
        proc, model = self.ja_processor, self.ja_model
        model.set_adapter("minutes")
        try:
            chat = self._ChatState(proc)
            chat.new_turn("system"); chat.add_text(system_prompt); chat.end_turn()
            chat.new_turn("user"); chat.add_text(transcript); chat.end_turn()
            chat.new_turn("assistant")
            ids: list[int] = []
            with torch.no_grad():
                for t in model.generate_sequential(**chat, max_new_tokens=max_new_tokens,
                                                   text_temperature=0.2, text_top_k=50):
                    if t.numel() == 1:
                        v = int(t.view(-1)[0])
                        if v in (7, 128):       # <|im_end|> / <|audio_start|>
                            break
                        ids.append(v)
            txt = proc.text.decode(torch.tensor(ids, dtype=torch.long)) if ids else ""
        finally:
            model.set_adapter(self._primary_adapter)
        return _SPECIAL.sub("", txt).strip()


class MockAudioEngine:
    """No-deps stand-in for offline wiring tests (`--mock`)."""

    def __init__(self, cfg: "LiveConfig", fake_ja: str = "今日のmeetingのagendaをshareします。",
                 fake_en: str = "Let's review the agenda for today's meeting."):
        self.cfg = cfg
        self.fake_ja = fake_ja
        self.fake_en = fake_en

    def transcribe(self, wav: np.ndarray, sr: int, lang: str) -> str:
        return self.fake_ja if lang == "ja" else self.fake_en

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        sr = self.cfg.sample_rate_out
        return np.zeros(int(sr * max(0.3, 0.04 * len(text))), dtype=np.float32), sr

    def synthesize_stream(self, text: str, lang: str, **kwargs):
        audio, _sr = self.synthesize(text, lang)
        step = self.cfg.sample_rate_out // 2
        for i in range(0, len(audio), step):
            yield audio[i:i + step]
