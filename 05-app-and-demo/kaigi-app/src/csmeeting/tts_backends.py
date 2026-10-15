"""Pluggable TTS. Default = edge-tts (free, no key, JP + EN neural voices).

Key idea: render each language span with a *script-matched* voice (Latin -> native English voice; ja/katakana ->
Japanese voice), using a single same-gender speaker pair per utterance so a spliced clip sounds like one bilingual
person. This keeps audio aligned with the gold label and teaches the loanword/switch boundary.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass

import numpy as np

# Default bilingual speaker profiles (override via config/pipeline.yaml). edge-tts voice ids.
DEFAULT_SPEAKERS = [
    {"name": "f1", "ja": "ja-JP-NanamiNeural", "en": "en-US-AriaNeural"},
    {"name": "m1", "ja": "ja-JP-KeitaNeural", "en": "en-US-GuyNeural"},
    {"name": "f2", "ja": "ja-JP-NanamiNeural", "en": "en-GB-SoniaNeural"},
    {"name": "m2", "ja": "ja-JP-KeitaNeural", "en": "en-AU-WilliamNeural"},
]

# Kokoro-82M (Apache-2.0) voice ids — commercial-clean, local, native 24 kHz. The first letter encodes
# language+gender: j*=Japanese, a*=American EN, b*=British EN; *f*=female, *m*=male (→ same-gender bilingual pairs).
DEFAULT_KOKORO_SPEAKERS = [
    {"name": "f1", "ja": "jf_alpha", "en": "af_heart"},
    {"name": "m1", "ja": "jm_kumo", "en": "am_michael"},
    {"name": "f2", "ja": "jf_gongitsune", "en": "bf_emma"},
    {"name": "m2", "ja": "jm_kumo", "en": "bm_george"},
]


@dataclass
class TTSResult:
    audio: np.ndarray   # float32 mono in [-1, 1]
    sr: int


def decode_audio_bytes(data: bytes) -> TTSResult:
    """Decode encoded audio bytes (mp3 from edge-tts) to float32 mono. soundfile first, pydub fallback."""
    try:
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:  # libsndfile without mp3 support -> try pydub/ffmpeg
        try:
            from pydub import AudioSegment
        except ImportError as e:
            raise RuntimeError(
                "Could not decode mp3 with soundfile and pydub is not installed. "
                "Install a modern `soundfile` (bundled libsndfile>=1.1 reads mp3) or `pip install -e '.[ffmpeg]'`."
            ) from e
        seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        if seg.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        arr = samples / float(1 << (8 * seg.sample_width - 1))
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return TTSResult(arr.astype(np.float32), int(sr))


def resample_linear(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or len(x) <= 1:
        return x.astype(np.float32)
    n = int(round(len(x) * target_sr / sr))
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    fp = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(fp, xp, x).astype(np.float32)


class EdgeTTSBackend:
    """Synthesize with Microsoft Edge neural voices via the `edge-tts` package (network required)."""

    def __init__(self, target_sr: int = 24000, gap_ms: int = 60):
        self.target_sr = target_sr
        self.gap_ms = gap_ms

    async def _stream_mp3(self, text: str, voice: str) -> bytes:
        import edge_tts  # lazy import (network dep)
        communicate = edge_tts.Communicate(text, voice)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        return bytes(buf)

    def synth_text(self, text: str, voice: str) -> np.ndarray:
        """One voice over the whole string. Returns float32 mono at target_sr."""
        mp3 = asyncio.run(self._stream_mp3(text, voice))
        res = decode_audio_bytes(mp3)
        return resample_linear(res.audio, res.sr, self.target_sr)

    def synth_spans(self, spans: list[dict], speaker: dict) -> np.ndarray:
        """Splice: each span voiced by speaker[span.lang], concatenated with a short silent gap."""
        gap = np.zeros(int(self.target_sr * self.gap_ms / 1000), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for i, span in enumerate(spans):
            voice = speaker.get(span["lang"], speaker["ja"])
            chunk = self.synth_text(span["text"], voice)
            if i > 0:
                pieces.append(gap)
            pieces.append(chunk)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


class KokoroBackend:
    """Synthesize with Kokoro-82M (Apache-2.0, local, native 24 kHz) — commercial-clean.

    Same interface as EdgeTTSBackend (`synth_text` / `synth_spans`) so it's a drop-in. One KPipeline per language
    code (lazily created); the voice's first letter selects the pipeline (j=Japanese, a=US-EN, b=UK-EN). Japanese
    G2P needs `misaki[ja]` (fugashi + unidic); install: `pip install kokoro misaki[ja]`.
    """

    _LANG_OF_PREFIX = {"j": "j", "a": "a", "b": "b"}   # voice id first char -> KPipeline lang_code

    def __init__(self, target_sr: int = 24000, gap_ms: int = 60, device: str | None = None):
        self.target_sr = target_sr
        self.gap_ms = gap_ms
        self.device = device
        self.native_sr = 24000          # Kokoro decodes at 24 kHz
        self._pipelines: dict[str, object] = {}

    def _pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            from kokoro import KPipeline  # lazy import (heavy dep)
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, device=self.device)
        return self._pipelines[lang_code]

    def _lang_code(self, voice: str) -> str:
        return self._LANG_OF_PREFIX.get(voice[:1], "a")

    def synth_text(self, text: str, voice: str) -> np.ndarray:
        """One Kokoro voice over the whole string. Returns float32 mono at target_sr."""
        pipe = self._pipeline(self._lang_code(voice))
        chunks: list[np.ndarray] = []
        for item in pipe(text, voice=voice):
            audio = getattr(item, "audio", None)
            if audio is None and isinstance(item, (tuple, list)):   # older API: (graphemes, phonemes, audio)
                audio = item[-1]
            if audio is None:
                continue
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(arr.astype(np.float32).reshape(-1))
        wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return resample_linear(wav, self.native_sr, self.target_sr)

    def synth_spans(self, spans: list[dict], speaker: dict) -> np.ndarray:
        """Splice: each span voiced by speaker[span.lang], concatenated with a short silent gap."""
        gap = np.zeros(int(self.target_sr * self.gap_ms / 1000), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for i, span in enumerate(spans):
            voice = speaker.get(span["lang"], speaker["ja"])
            chunk = self.synth_text(span["text"], voice)
            if i > 0:
                pieces.append(gap)
            pieces.append(chunk)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


# backend name -> (class, default speaker profiles)
_BACKENDS = {
    "edge": (EdgeTTSBackend, DEFAULT_SPEAKERS),
    "kokoro": (KokoroBackend, DEFAULT_KOKORO_SPEAKERS),
}


def make_backend(name: str = "edge", *, target_sr: int = 24000, gap_ms: int = 60):
    """Return (backend, default_speakers) for the named TTS backend ('edge' | 'kokoro')."""
    try:
        cls, speakers = _BACKENDS[name]
    except KeyError as e:
        raise ValueError(f"unknown TTS backend {name!r}; choose from {sorted(_BACKENDS)}") from e
    return cls(target_sr=target_sr, gap_ms=gap_ms), speakers


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(path, audio, sr, subtype="PCM_16")


def silence(sr: int, seconds: float = 0.6) -> np.ndarray:
    """Placeholder audio for offline pipeline smoke-tests (no network)."""
    return np.zeros(int(sr * seconds), dtype=np.float32)
