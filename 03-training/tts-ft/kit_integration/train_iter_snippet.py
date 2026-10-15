"""DROP-IN for the kit's scripts/audio/train.py.

The kit ships a `TrainingSamples` iterator that yields list[ChatMessage] for TTS. Replace it with the version below
to fine-tune for code-switch ASR from our manifest. This file is illustrative — copy the class into the kit's
train.py (or import csmeeting there). It is NOT run from this repo.
"""

from __future__ import annotations

from collections.abc import Iterator

# These come from the `liquid-audio` package available in the kit's audio env.
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment

# ---- BEFORE (kit default, TTS): -------------------------------------------------------------------------------
# class TrainingSamples:
#     def __iter__(self):
#         ds = load_dataset("reach-vb/jenny_tts_dataset", split="train").cast_column("audio", Audio(decode=False))
#         for row in ds:
#             yield [
#                 ChatMessage(role="system",    content=[TextSegment(text="Perform TTS. Use the Irish female voice.")]),
#                 ChatMessage(role="user",      content=[TextSegment(text=row["transcription"])]),
#                 ChatMessage(role="assistant", content=[AudioSegment(audio=row["audio"]["bytes"])]),
#             ]

# ---- AFTER (ours, code-switch ASR): ---------------------------------------------------------------------------
from pathlib import Path
import json

MANIFEST = "data/cs/manifest.jsonl"          # copy our manifest.jsonl + audio/ here, or use an absolute path
ASR_SYSTEM_PROMPT = "Perform ASR in japanese."   # LFM2.5-Audio-1.5B-JP card


class TrainingSamples:
    """Yields list[ChatMessage] for ASR: system prompt / user=audio bytes / assistant=gold transcript."""

    def __init__(self, manifest: str = MANIFEST, system_prompt: str = ASR_SYSTEM_PROMPT):
        self.manifest = Path(manifest)
        self.system_prompt = system_prompt

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        root = self.manifest.parent
        with open(self.manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                audio_path = Path(row["audio_path"])
                audio_path = audio_path if audio_path.is_absolute() else root / audio_path
                yield [
                    ChatMessage(role="system", content=[TextSegment(text=self.system_prompt)]),
                    ChatMessage(role="user", content=[AudioSegment(audio=audio_path.read_bytes())]),
                    ChatMessage(role="assistant", content=[TextSegment(text=row["transcript"])]),
                ]


# Equivalent one-liner if you `pip install -e` this repo into the kit's env:
#   from csmeeting.cs_asr_iterator import CodeSwitchASRIterator
#   TrainingSamples = lambda: CodeSwitchASRIterator("data/cs/manifest.jsonl")
