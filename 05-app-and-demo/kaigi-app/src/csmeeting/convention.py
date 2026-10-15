"""Script classification + language-span segmentation for the transcription convention.

The whole pipeline rests on one idea (see docs/transcription_convention.md): the *script* of a token encodes
whether it is a naturalized Japanese loanword (katakana/kanji/kana -> "ja") or a true English switch (Latin -> "en").
So we can derive language spans from the gold transcript purely by Unicode script, and render each span with a
script-matched TTS voice. This module is pure stdlib (no MeCab needed for segmentation).
"""

from __future__ import annotations

from typing import Literal, TypedDict

Lang = Literal["ja", "en"]


class Span(TypedDict):
    text: str
    lang: Lang


def classify_char(ch: str) -> Literal["ja", "en", "neu"]:
    """Classify a single character as Japanese, English(Latin), or neutral (digit/space/ascii-punct)."""
    o = ord(ch)
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return "en"
    if (
        0x3040 <= o <= 0x30FF      # hiragana + katakana
        or 0x31F0 <= o <= 0x31FF   # katakana phonetic extensions
        or 0xFF66 <= o <= 0xFF9F   # halfwidth katakana
        or 0x4E00 <= o <= 0x9FFF   # CJK unified
        or 0x3400 <= o <= 0x4DBF   # CJK extension A
        or 0x3000 <= o <= 0x303F   # CJK symbols & punctuation (。、「」 etc.)
        or 0xFF01 <= o <= 0xFF60    # fullwidth forms (Japanese typography)
    ):
        return "ja"
    return "neu"


def has_latin(text: str) -> bool:
    """True if the text contains any Latin letter (i.e., a true English switch under our convention)."""
    return any("a" <= c <= "z" or "A" <= c <= "Z" for c in text)


def is_single_jp(text: str) -> bool:
    """True if there is no Latin -> the utterance is pure Japanese (loanwords stay katakana)."""
    return not has_latin(text)


def segment_into_spans(text: str) -> list[Span]:
    """Split text into consecutive ja / en spans. Neutral chars attach to the adjacent span.

    Guarantee: ''.join(s['text'] for s in segment_into_spans(t)) == t  (lossless).
    """
    runs: list[list[str]] = []  # [lang, text]
    for ch in text:
        c = classify_char(ch)
        if c == "neu":
            if runs:
                runs[-1][1] += ch              # attach to current run, keep its language
            else:
                runs.append(["neu", ch])        # leading neutral; language decided by next real char
        else:
            if runs and runs[-1][0] == c:
                runs[-1][1] += ch
            elif runs and runs[-1][0] == "neu":
                runs[-1][0] = c                 # leading neutral block adopts this language
                runs[-1][1] += ch
            else:
                runs.append([c, ch])

    # collapse any still-neutral run (whole string was neutral) into ja, then merge adjacent same-lang runs.
    merged: list[list[str]] = []
    for lang, t in runs:
        lang = "ja" if lang == "neu" else lang
        if merged and merged[-1][0] == lang:
            merged[-1][1] += t
        else:
            merged.append([lang, t])
    return [{"text": t, "lang": lang} for lang, t in merged]  # type: ignore[misc]


def validate_spans(text: str, spans: list[Span]) -> None:
    """Raise if spans don't losslessly reconstruct text."""
    joined = "".join(s["text"] for s in spans)
    if joined != text:
        raise ValueError(f"spans do not reconstruct text:\n  text={text!r}\n  join={joined!r}")


def infer_style(text: str) -> Literal["splice", "single_jp"]:
    """splice = has a true English (Latin) span needing a native-English voice; single_jp = all Japanese."""
    return "splice" if has_latin(text) else "single_jp"


if __name__ == "__main__":  # quick self-test (no deps)
    cases = {
        "来週のミーティングまでにKPIをfinalizeしておきます。": ["ja", "en", "ja", "en", "ja"],
        "今日のアジェンダはスケジュールの確認です。": ["ja"],
        "let's keep it simple、まずはMVPをshipしてから考えましょう。": ["en", "ja", "en", "ja", "en", "ja"],
        "テストのcoverageを80%まで上げるのがgoalです。": ["ja", "en", "ja", "en", "ja"],
    }
    ok = True
    for text, expected_langs in cases.items():
        spans = segment_into_spans(text)
        validate_spans(text, spans)
        got = [s["lang"] for s in spans]
        status = "ok " if got == expected_langs else "BAD"
        if got != expected_langs:
            ok = False
        print(f"[{status}] {text}")
        print(f"        style={infer_style(text)} spans={[(s['lang'], s['text']) for s in spans]}")
    print("\nself-test:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
