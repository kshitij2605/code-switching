"""Script-based language ID (ja vs en) — LFM-only, no external models.

For direction="auto" we transcribe the utterance with the LFM ASR and read the majority script of the transcript
(katakana/kanji/kana → ja, Latin → en). This avoids any non-LFM model (e.g. Whisper).
"""

from __future__ import annotations


def lang_of(s: str) -> str:
    """Classify an utterance transcript as 'ja' or 'en'.

    Any Japanese script (kana/kanji) means Japanese is the matrix language (true even for code-switched Japanese,
    which always has kana particles); a pure-English utterance has none. So we flag 'ja' on *any* Japanese script
    rather than a character-count majority (English words would otherwise outvote the compact Japanese).
    """
    ja = sum(1 for ch in s if 0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF or 0x3400 <= ord(ch) <= 0x4DBF)
    return "ja" if ja >= 1 else "en"
