"""Build commercial-safe JA->EN SFT data for the OPTIONAL MT fine-tune (only if eval says it's needed).

Sources (all commercial-clean): FLEURS (CC-BY), Tatoeba (CC-BY-2.0), JESC (CC-BY-SA).
Output = `messages` format for the kit's text track.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from csmeeting.manifest import write_jsonl
from csmeeting.mt.prompt import SYSTEM_PROMPT


def _ratio_ok(ja: str, en: str) -> bool:
    if not ja or not en:
        return False
    la, le = len(ja), len(en.split())
    if la < 2 or le < 1:
        return False
    return 0.2 <= (le / max(la, 1)) <= 5.0  # crude JA-chars vs EN-words sanity filter


def _pairs_from_translation(ds, limit: int) -> Iterator[tuple[str, str]]:
    n = 0
    for row in ds:
        t = row.get("translation") or {}
        ja, en = t.get("ja"), t.get("en")
        if ja and en and _ratio_ok(ja, en):
            yield ja, en
            n += 1
        if limit and n >= limit:
            break


def jesc_pairs(limit: int) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    yield from _pairs_from_translation(load_dataset("nntsuzu/JESC", split="train", streaming=True), limit)


def tatoeba_pairs(limit: int) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    try:
        ds = load_dataset("Helsinki-NLP/tatoeba", lang1="en", lang2="ja", split="train", streaming=True)
    except Exception as e:  # noqa: BLE001
        print(f"[tatoeba] load failed ({e}); skipping")
        return
    yield from _pairs_from_translation(ds, limit)


def fleurs_pairs(limit: int, split: str = "train") -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    ja = load_dataset("google/fleurs", "ja_jp", split=split)
    en = load_dataset("google/fleurs", "en_us", split=split)
    en_by_id: dict = {}
    for r in en:
        en_by_id.setdefault(r["id"], r["transcription"])
    n = 0
    for r in ja:
        e = en_by_id.get(r["id"])
        if e and _ratio_ok(r["transcription"], e):
            yield r["transcription"], e
            n += 1
        if limit and n >= limit:
            break


def to_sft(ja: str, en: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ja},
        {"role": "assistant", "content": en},
    ]}


def build(out_path: str | Path, *, jesc: int = 20000, tatoeba: int = 20000, fleurs: int = 2000) -> int:
    seen: set = set()
    rows: list[dict] = []

    def add(it: Iterator[tuple[str, str]]) -> None:
        for ja, en in it:
            if (ja, en) in seen:
                continue
            seen.add((ja, en))
            rows.append(to_sft(ja, en))

    if fleurs:
        add(fleurs_pairs(fleurs))
    if tatoeba:
        add(tatoeba_pairs(tatoeba))
    if jesc:
        add(jesc_pairs(jesc))
    return write_jsonl(out_path, rows)
