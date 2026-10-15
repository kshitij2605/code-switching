"""JSONL read/write + the schema for transcript rows and manifest rows."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# transcripts.jsonl row (output of gen_transcripts)
TRANSCRIPT_FIELDS = ("id", "transcript", "style", "domain", "source")
# manifest.jsonl row (output of synthesize) — adds audio + spans + voice
MANIFEST_FIELDS = ("id", "audio_path", "transcript", "spans", "style", "voice", "domain", "source", "duration_s", "sr")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
