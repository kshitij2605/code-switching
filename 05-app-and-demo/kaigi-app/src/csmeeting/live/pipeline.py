"""Orchestrates the cascade: VAD -> ASR -> MT -> TTS, per utterance.

Each utterance yields the JA transcript (live transcript / minutes), the EN translation (subtitle / minutes), and
the EN audio (what the listener hears). Run over a file or the microphone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from csmeeting.live.config import LiveConfig


@dataclass
class Utterance:
    index: int
    ja_text: str                 # transcript (code-switch aware)
    en_text: str                 # translation
    audio: np.ndarray = field(repr=False)   # EN waveform
    sr: int = 24000


class LiveTranslator:
    def __init__(self, cfg: "LiveConfig", engine=None, translator=None):
        self.cfg = cfg
        self.engine = engine
        self.translator = translator

    # --- one utterance through the whole cascade ---
    def process(self, wav: np.ndarray, sr: int, index: int = 0) -> Utterance:
        # CLI cascade is fixed JA->EN; the bidirectional path is the Assistant (scripts/demo_app.py)
        ja = self.engine.transcribe(wav, sr, "ja")
        en = self.translator.translate(ja, "ja", "en")
        audio, out_sr = self.engine.synthesize(en, "en")
        return Utterance(index=index, ja_text=ja, en_text=en, audio=audio, sr=out_sr)

    # --- file mode ---
    def run_file(self, path: str, on_utterance: Callable[[Utterance], None]) -> list[Utterance]:
        import soundfile as sf
        from csmeeting.live.vad import segment_array

        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        segments = segment_array(wav, sr, self.cfg)
        if not segments:  # no pause detected (short clip) -> treat whole thing as one utterance
            segments = [wav if wav.ndim == 1 else wav.mean(axis=1)]
        out = []
        for i, seg in enumerate(segments):
            utt = self.process(seg, self.cfg.vad_sample_rate, i)
            on_utterance(utt)
            out.append(utt)
        return out

    # --- microphone mode ---
    def run_mic(self, on_utterance: Callable[[Utterance], None]) -> None:
        import queue

        import sounddevice as sd

        from csmeeting.live.vad import EnergyVAD

        cfg = self.cfg
        frame_n = int(cfg.vad_sample_rate * cfg.vad_frame_ms / 1000)
        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            q.put(indata[:, 0].copy())

        def _frames():
            while True:
                yield q.get()

        vad = EnergyVAD(cfg)
        print("[mic] listening — Ctrl-C to stop")
        with sd.InputStream(samplerate=cfg.vad_sample_rate, channels=1, blocksize=frame_n,
                            dtype="float32", callback=_callback):
            i = 0
            try:
                for seg in vad.segment(_frames()):
                    on_utterance(self.process(seg, cfg.vad_sample_rate, i))
                    i += 1
            except KeyboardInterrupt:
                print("\n[mic] stopped")


def play(utt: Utterance) -> None:
    """Play an utterance's EN audio through the default output device (needs sounddevice)."""
    if utt.audio.size == 0:
        return
    import sounddevice as sd
    sd.play(utt.audio, utt.sr)
    sd.wait()
