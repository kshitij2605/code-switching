"""manifest.jsonl  ->  list[ChatMessage]  (the drop-in for the Liquid4All LFM audio training kit).

The kit's scripts/audio/train.py expects an iterator that yields `list[ChatMessage]` per training sample. For ASR
fine-tuning each sample is:  system="Perform ASR in japanese."  /  user=<audio bytes>  /  assistant=<transcript>.

Usage in the kit (see kit_integration/train_iter_snippet.py):
    from csmeeting.cs_asr_iterator import CodeSwitchASRIterator
    TrainingSamples = lambda: CodeSwitchASRIterator("data/synth/manifest.jsonl")

This module imports `liquid_audio` lazily, so you can `--preview` a manifest without the model installed.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from csmeeting.manifest import read_jsonl

DEFAULT_SYSTEM_PROMPT = "Perform ASR in japanese."   # matches the LFM2.5-Audio-1.5B-JP model card
JA_SYSTEM_PROMPT = "Perform ASR in japanese."        # JA + code-switch (matrix language is Japanese)
EN_SYSTEM_PROMPT = "Perform ASR."                    # English replay


def _resolve_audio(audio_path: str, manifest_path: str | Path) -> Path:
    p = Path(audio_path)
    return p if p.is_absolute() else Path(manifest_path).parent / p


def _prompt_for_row(row: dict, ja_prompt: str, en_prompt: str) -> str:
    """English-only rows get the EN ASR prompt; JA + code-switch rows get the JA prompt (matrix lang = JA)."""
    style = str(row.get("style", ""))
    if style.startswith("mono_en"):
        return en_prompt
    spans = row.get("spans")
    if not style and spans and all(sp.get("lang") == "en" for sp in spans):
        return en_prompt
    return ja_prompt


def iter_samples(manifest_path: str | Path, system_prompt: str | None = None,
                 ja_prompt: str = JA_SYSTEM_PROMPT, en_prompt: str = EN_SYSTEM_PROMPT) -> Iterator[list]:
    """Yield list[ChatMessage] per row. Requires `liquid_audio` (run inside the kit / on the GPU job).

    If `system_prompt` is given it's used for every row; otherwise the prompt is chosen per row by language
    (EN replay -> "Perform ASR."; JA + code-switch -> "Perform ASR in japanese.").
    """
    from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment

    for row in read_jsonl(manifest_path):
        prompt = system_prompt if system_prompt is not None else _prompt_for_row(row, ja_prompt, en_prompt)
        audio_bytes = _resolve_audio(row["audio_path"], manifest_path).read_bytes()
        yield [
            ChatMessage(role="system", content=[TextSegment(text=prompt)]),
            ChatMessage(role="user", content=[AudioSegment(audio=audio_bytes)]),
            ChatMessage(role="assistant", content=[TextSegment(text=row["transcript"])]),
        ]


class CodeSwitchASRIterator:
    """Reusable iterable (the kit instantiates this and iterates it, possibly across epochs).

    system_prompt=None -> per-row prompt by language (recommended for a mixed CS + mono JA + mono EN manifest).
    """

    def __init__(self, manifest_path: str | Path, system_prompt: str | None = None,
                 ja_prompt: str = JA_SYSTEM_PROMPT, en_prompt: str = EN_SYSTEM_PROMPT):
        self.manifest_path = manifest_path
        self.system_prompt = system_prompt
        self.ja_prompt = ja_prompt
        self.en_prompt = en_prompt

    def __iter__(self) -> Iterator[list]:
        return iter_samples(self.manifest_path, self.system_prompt, self.ja_prompt, self.en_prompt)


def preview(manifest_path: str | Path, n: int) -> None:
    """Print the first n samples WITHOUT importing liquid_audio (sanity-check the manifest + audio files)."""
    rows: list[dict[str, Any]] = list(read_jsonl(manifest_path))
    print(f"manifest: {manifest_path}  ({len(rows)} rows)")
    for row in rows[:n]:
        audio = _resolve_audio(row["audio_path"], manifest_path)
        size = audio.stat().st_size if audio.exists() else -1
        flag = "" if size >= 0 else "  <-- MISSING AUDIO"
        print(f"\n  id={row['id']}  style={row.get('style')}  voice={row.get('voice')}  "
              f"dur={row.get('duration_s')}s  audio={size}B{flag}")
        print(f"  transcript: {row['transcript']}")
        print(f"  spans: {[(s['lang'], s['text']) for s in row.get('spans', [])]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preview a manifest as ASR ChatMessage samples.")
    ap.add_argument("manifest")
    ap.add_argument("--preview", type=int, default=3, help="print first N rows (no liquid_audio needed)")
    args = ap.parse_args()
    preview(args.manifest, args.preview)
