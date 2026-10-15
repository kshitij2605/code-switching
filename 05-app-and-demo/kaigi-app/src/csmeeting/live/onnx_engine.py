"""CPU audio engine — wraps liquidonnx `LFM2AudioInference` (onnxruntime, CPU-only, the LFM2.5-Audio model).

LFM-only, no PyTorch/CUDA. Mirrors `AudioEngine`'s interface (transcribe / synthesize / synthesize_stream by lang),
loading one ONNX model per language. Validated on host: ASR ~real-time on CPU; **TTS is RTF ~2.6 (turn-based, not
real-time on CPU)**, so `synthesize_stream` produces the full clip then emits it in chunks (no live streaming).

Requires the `onnx-export` package (`liquidonnx`). Either pip-install it, or set `LiveConfig.onnx_export_src` to the
`onnx-export/src` path. The JA model has no published ONNX repo → export it once: `lfm2-audio-export
LiquidAI/LFM2.5-Audio-1.5B-JP --precision q4`.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import tempfile
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from csmeeting.live.config import LiveConfig

_KEYMAP = {
    "decoder": "decoder_file", "audio_embedding": "audio_embedding_file", "audio_encoder": "audio_encoder_file",
    "audio_detokenizer": "audio_detokenizer_file", "vocoder_depthformer": "vocoder_depthformer_file",
    "depthformer": "vocoder_depthformer_file",
}

# preference order per onnx_ep setting; filtered against onnxruntime.get_available_providers()
_EP_ORDER = {
    "cpu": ["CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "auto": ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}
# names that signal "the caller explicitly wanted a hardware EP" → warn loudly if it's missing
_EP_REQUIRED = {"dml": "DmlExecutionProvider", "cuda": "CUDAExecutionProvider"}


def _liquidonnx(cfg: "LiveConfig"):
    import sys
    if cfg.onnx_export_src and cfg.onnx_export_src not in sys.path:
        sys.path.insert(0, cfg.onnx_export_src)
    from liquidonnx.lfm2_audio import infer as inf
    return inf


def _resolve_providers(ep: str) -> list[str]:
    """Map `onnx_ep` to an available onnxruntime provider list (always ends with CPU as a fallback)."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    ep = (ep or "cpu").lower()
    providers = [p for p in _EP_ORDER.get(ep, ["CPUExecutionProvider"]) if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    want = _EP_REQUIRED.get(ep)
    if want and want not in available:
        warnings.warn(
            f"onnx_ep={ep!r} requested but {want} is not in onnxruntime's available providers "
            f"({sorted(available)}); falling back to {providers}. For DirectML install the "
            f"`onnxruntime-directml` wheel (Windows) instead of stock `onnxruntime`.",
            stacklevel=2,
        )
    return providers


@contextlib.contextmanager
def _force_providers(providers: list[str]):
    """Inject `providers` (and DML-safe SessionOptions) into every onnxruntime session created in this scope.

    `liquidonnx` builds its own sessions and isn't guaranteed to accept a `providers=` kwarg, so we patch
    `InferenceSession.__init__` for the duration of model construction. DirectML requires memory-pattern off and
    sequential execution, set only when we also supply the providers and the caller didn't pass its own options.
    """
    import onnxruntime as ort

    orig = ort.InferenceSession.__init__
    needs_dml = "DmlExecutionProvider" in providers

    def patched(self, *args, **kwargs):  # noqa: ANN001
        if not kwargs.get("providers") and len(args) < 3:
            kwargs["providers"] = list(providers)
        if needs_dml and not kwargs.get("sess_options") and len(args) < 2:
            so = ort.SessionOptions()
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            kwargs["sess_options"] = so
        try:
            orig(self, *args, **kwargs)
        except TypeError:  # signature mismatch — retry without our injected kwargs
            kwargs.pop("providers", None)
            kwargs.pop("sess_options", None)
            orig(self, *args, **kwargs)

    ort.InferenceSession.__init__ = patched
    try:
        yield
    finally:
        ort.InferenceSession.__init__ = orig


class OnnxAudioEngine:
    def __init__(self, cfg: "LiveConfig"):
        import pathlib
        self.cfg = cfg
        self._lo = _liquidonnx(cfg)
        self._files = self._lo.resolve_precision_files(cfg.onnx_precision)
        kw = {_KEYMAP[k]: v for k, v in self._files.items() if k in _KEYMAP and v}

        self.providers = _resolve_providers(cfg.onnx_ep)
        print(f"[OnnxAudioEngine] onnx_ep={cfg.onnx_ep!r} -> providers={self.providers}")

        # If the LFM2AudioInference constructor accepts a providers-style kwarg, pass it (clean path);
        # the _force_providers patch covers wrappers that build sessions internally without exposing one.
        try:
            ctor_params = set(inspect.signature(self._lo.LFM2AudioInference).parameters)
        except (TypeError, ValueError):
            ctor_params = set()
        for name in ("providers", "execution_providers", "ep", "execution_provider"):
            if name in ctor_params:
                kw[name] = self.providers if name in ("providers", "execution_providers") else cfg.onnx_ep
                break

        self._inf: dict[str, object] = {}
        with _force_providers(self.providers):
            if cfg.ja_onnx_dir:
                self._inf["ja"] = self._lo.LFM2AudioInference(pathlib.Path(cfg.ja_onnx_dir), **kw)
            if cfg.en_onnx_dir:
                self._inf["en"] = self._lo.LFM2AudioInference(pathlib.Path(cfg.en_onnx_dir), **kw)
        if not self._inf:
            raise RuntimeError("OnnxAudioEngine: set ja_onnx_dir and/or en_onnx_dir in LiveConfig.")

    def _route(self, lang: str):
        inf = self._inf.get(lang) or next(iter(self._inf.values()))
        asr = self.cfg.asr_prompt_ja if lang == "ja" else self.cfg.asr_prompt_en
        tts = self.cfg.tts_prompt_ja if lang == "ja" else self.cfg.tts_prompt_en
        return inf, asr, tts

    def transcribe(self, wav: np.ndarray, sr: int, lang: str) -> str:
        import scipy.io.wavfile
        inf, asr_prompt, _ = self._route(lang)
        pcm = (np.clip(np.ascontiguousarray(wav, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            scipy.io.wavfile.write(path, sr, pcm)
            return inf.transcribe(path, system_prompt=asr_prompt).strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        inf, _, tts_prompt = self._route(lang)
        codes = inf.synthesize(text, system_prompt=tts_prompt,
                               audio_temperature=self.cfg.tts_audio_temperature, audio_top_k=self.cfg.tts_audio_top_k)
        if not codes:
            return np.zeros(0, dtype=np.float32), self.cfg.sample_rate_out
        # fast in-process decode (reuses the loaded detokenizer session); fall back to the module helper
        try:
            arr = np.asarray(codes).reshape(len(codes), 8).astype(np.int64)
            wav = np.asarray(inf.decode_audio(arr.T[None, :, :]), dtype=np.float32).reshape(-1)
        except Exception:  # noqa: BLE001
            import scipy.io.wavfile
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                self._lo.audio_codes_to_wav(codes, path, model_dir=inf.model_dir,
                                            audio_detokenizer_file=self._files["audio_detokenizer"])
                _sr, data = scipy.io.wavfile.read(path)
                wav = (data.astype(np.float32) / 32767) if data.dtype == np.int16 else data.astype(np.float32)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        return wav, self.cfg.sample_rate_out

    def synthesize_stream(self, text: str, lang: str, **kwargs):
        # CPU can't generate faster than real-time -> synthesize fully, then emit in ~0.5 s chunks for the player.
        wav, sr = self.synthesize(text, lang)
        step = max(1, sr // 2)
        for i in range(0, len(wav), step):
            yield wav[i:i + step]
