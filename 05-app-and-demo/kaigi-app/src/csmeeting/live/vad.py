"""Utterance segmentation — "translate on the pause".

For JA->EN, waiting for the speaker's pause yields the (verb-final) full clause, which gives a grammatical
translation. Default is a dependency-free energy VAD; an optional Silero backend (better, array-mode) is provided.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from csmeeting.live.config import LiveConfig


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target or len(x) <= 1:
        return x.astype(np.float32)
    n = int(round(len(x) * target / sr))
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    fp = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(fp, xp, x).astype(np.float32)


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12))


def frame_generator(wav: np.ndarray, frame_n: int) -> Iterator[np.ndarray]:
    for i in range(0, len(wav) - frame_n + 1, frame_n):
        yield wav[i : i + frame_n]


class EnergyVAD:
    """Streaming RMS-threshold VAD. Emits one np.ndarray (mono float32 @ vad_sample_rate) per utterance."""

    def __init__(self, cfg: "LiveConfig"):
        self.cfg = cfg
        self.frame_n = int(cfg.vad_sample_rate * cfg.vad_frame_ms / 1000)
        self.min_speech_frames = max(1, cfg.vad_min_speech_ms // cfg.vad_frame_ms)
        self.min_silence_frames = max(1, cfg.vad_min_silence_ms // cfg.vad_frame_ms)

    def segment(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        buf: list[np.ndarray] = []
        in_speech = False
        speech = silence = 0
        for f in frames:
            voiced = _rms(f) >= self.cfg.vad_energy_threshold
            if voiced:
                if not in_speech:
                    in_speech, buf, speech, silence = True, [], 0, 0
                buf.append(f); speech += 1; silence = 0
            elif in_speech:
                buf.append(f); silence += 1
                if silence >= self.min_silence_frames:
                    if speech >= self.min_speech_frames:
                        yield np.concatenate(buf)
                    in_speech, buf = False, []
        if in_speech and speech >= self.min_speech_frames:
            yield np.concatenate(buf)


def segment_array(wav: np.ndarray, sr: int, cfg: "LiveConfig") -> list[np.ndarray]:
    """Split a whole waveform into utterances (file mode). Returns segments at cfg.vad_sample_rate."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = _resample(wav.astype(np.float32), sr, cfg.vad_sample_rate)
    if cfg.vad_backend == "silero":
        return _segment_silero(wav, cfg)
    vad = EnergyVAD(cfg)
    return list(vad.segment(frame_generator(wav, vad.frame_n)))


def _segment_silero(wav: np.ndarray, cfg: "LiveConfig") -> list[np.ndarray]:
    """Optional, higher-quality array segmentation via Silero VAD (torch.hub). Falls back to energy on failure."""
    try:
        import torch
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        get_speech_timestamps = utils[0]
        t = torch.from_numpy(wav)
        ts = get_speech_timestamps(t, model, sampling_rate=cfg.vad_sample_rate,
                                   min_silence_duration_ms=cfg.vad_min_silence_ms,
                                   min_speech_duration_ms=cfg.vad_min_speech_ms)
        return [wav[seg["start"]: seg["end"]] for seg in ts]
    except Exception as e:  # noqa: BLE001
        print(f"[vad] silero unavailable ({e!r}); falling back to energy VAD")
        vad = EnergyVAD(cfg)
        return list(vad.segment(frame_generator(wav, vad.frame_n)))
