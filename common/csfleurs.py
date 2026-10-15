"""CS-FLEURS (byan/cs-fleurs) helpers shared by the training downloader and the
dev/eval loaders.

The XTTS subsets' text is MeCab-spaced with **English** markers, e.g.
    "旅行 **agents** **agents** の 多く は 、 ..."
``clean_xtts_text`` removes the markers and the tokeniser spacing but keeps the
words: the audio was synthesised from this text, so a repeated word
("agents agents") is really spoken. ``is_garbled`` flags those rows.
"""
from __future__ import annotations

import re

REPO = "byan/cs-fleurs"
LANG = "jpn-eng"
# jpn-eng splits and their role (common/leak_guard.py); mms/test has no Japanese pairs.
SPLIT_ROLES = {"read/test": "test", "xtts/test1": "dev", "xtts/test2": "dev", "xtts/train": "train"}

_JA = r"　-〿぀-ゟ゠-ヿ㐀-䶿一-鿿＀-￯"
_SPACE_NEXT_TO_JA = re.compile(rf"\s+(?=[{_JA}])|(?<=[{_JA}])\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+(?=[,.;:!?%)\]])")
_SPACE_AFTER_OPEN = re.compile(r"(?<=[(\[])\s+")
_DUP_SPAN = re.compile(r"\*\*([^*]+)\*\*\s+\*\*\1\*\*", re.IGNORECASE)


def clean_xtts_text(t: str) -> str:
    """Remove ** markers and MeCab spacing; keep spaces only between Latin words."""
    t = " ".join(t.replace("**", " ").split())
    t = _SPACE_NEXT_TO_JA.sub("", t)
    t = _SPACE_BEFORE_PUNCT.sub("", t)
    t = _SPACE_AFTER_OPEN.sub("", t)
    return t.strip()


def is_garbled(raw_text: str) -> bool:
    """True if an English span is immediately repeated (align-then-swap artefact)."""
    return bool(_DUP_SPAN.search(raw_text))


def sentence_id(row_id: str) -> str:
    """'1097_jpn-spk0_sample0' -> '1097' (one FLEURS sentence, several voices)."""
    return str(row_id).split("_", 1)[0]
