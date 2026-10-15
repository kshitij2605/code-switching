"""Split a JA-EN code-switched sentence into language spans for TTS renderers.

  segments("合金とは、basically a mixture of 2種類以上のmetalsです。")
  -> [{"lang": "ja", "text": "合金とは、"}, {"lang": "en", "text": "basically a mixture of"},
      {"lang": "ja", "text": "2種類以上の"}, {"lang": "en", "text": "metals"},
      {"lang": "ja", "text": "です。"}]

Latin words are English. Digits and number punctuation are English only between
two English words ("magnitude 6.5 earthquake") and are read in Japanese
otherwise ("2種類"). Spaces and ASCII punctuation attach to the current span.
Shared by the twin builder (04-evaluation/eval-ft/data/make_twins.py) and the
text generators (01-data-generation/asr-ft/gen_cs_text_llm.py).
"""
from __future__ import annotations

import re

_CHUNK = re.compile(r"[A-Za-z][A-Za-z'’\-]*|\d+(?:[.,]\d+)*|\s+|[!-/:-@\[-`{-~]|.", re.S)
_LATIN = re.compile(r"[A-Za-z]")


def segments(text: str) -> list[dict]:
    chunks = _CHUNK.findall(text)
    kinds = []
    for c in chunks:
        if _LATIN.match(c):
            kinds.append("en")
        elif c.isspace() or re.fullmatch(r"[!-/:-@\[-`{-~]", c):
            kinds.append("glue")
        elif c[0].isdigit():
            kinds.append("num")
        else:
            kinds.append("ja")
    # numbers between two English words are English, otherwise Japanese
    for i, k in enumerate(kinds):
        if k == "num":
            prev = next((kinds[j] for j in range(i - 1, -1, -1) if kinds[j] not in ("glue", "num")), None)
            nxt = next((kinds[j] for j in range(i + 1, len(kinds)) if kinds[j] not in ("glue", "num")), None)
            kinds[i] = "en" if prev == "en" and nxt == "en" else "ja"
    segs: list[dict] = []
    cur_lang = None
    for c, k in zip(chunks, kinds):
        lang = k if k != "glue" else (cur_lang or "ja")
        if segs and lang == cur_lang:
            segs[-1]["text"] += c
        else:
            segs.append({"lang": lang, "text": c})
            cur_lang = lang
    out = []
    for s in segs:
        t = s["text"].strip()
        if t and not re.fullmatch(r"[\s!-/:-@\[-`{-~、。]+", t):
            if out and out[-1]["lang"] == s["lang"]:
                out[-1]["text"] += " " + t
            else:
                out.append({"lang": s["lang"], "text": t})
    return out
