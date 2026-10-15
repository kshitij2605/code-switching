#!/usr/bin/env python3
"""Code-switching text profile: how a CS corpus switches (Phase 3 calibration).

Synthetic CS text must match the switching statistics of real CS speech:
prior work (e.g. on SEAME) found mismatched text hurt far more
than mismatched accent. This module measures, per corpus:

  cmi            Code-Mixing Index (Gambäck & Das 2014), mean over utterances:
                 100 * (1 - max_lang_tokens / (N - U)), where U = language-independent
                 tokens (numbers, punctuation).
  m_index        Multilingual index (Barnett et al. 2000) over the corpus's
                 JA/EN token counts: 0 = one language, 1 = balanced.
  i_index        Integration index: switch points / (N - 1), mean over utterances.
  switches       Mean switch points per utterance.
  en_span_len    Mean English span length (words). single_word_insert is the
                 share of English spans that are a single word (insertional
                 vs alternational switching).
  burstiness     (sd - mean) / (sd + mean) of language-span lengths (Goh & Barabási).
  span_entropy   Shannon entropy (bits) of the span-length distribution.
  en_func_share  Share of English tokens that are function words: natural CS
                 inserts content words, while LLM text over-inserts function words.
  en_start       Share of utterances that begin in English.

Tokens: Japanese words from fugashi/unidic-lite (the scorer's pinned dictionary),
English words as Latin tokens.

Usage:
  python common/cs_profile.py name=path.jsonl[:field] ...    (default field: reference,
                                                             else transcript, else text)
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

_LATIN = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_JA = re.compile(r"[぀-ヿ㐀-䶿一-鿿々〆ー]")
EN_FUNCTION = set("""a an the of to in on at for from by with about as into like through after over
between out against during without before under around among and or but if than because while so
nor yet is are was were be been being am do does did have has had will would shall should can could
may might must not no this that these those it its it's i me my we us our you your he him his she her
they them their what which who whom whose when where why how all any both each few more most other
some such only own same too very just there here then also don't doesn't isn't aren't wasn't""".split())

_TAGGER = None


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        import os
        import fugashi
        import unidic_lite
        d = unidic_lite.DICDIR
        _TAGGER = fugashi.Tagger(f'-d "{d}" -r "{os.path.join(d, "mecabrc")}"')
    return _TAGGER


def tag_tokens(text: str) -> list[tuple[str, str]]:
    """[(token, lang)] with lang in {ja, en, x}; x = language-independent."""
    out: list[tuple[str, str]] = []
    for w in _tagger()(text):
        s = w.surface
        if _LATIN.fullmatch(s):
            out.append((s.lower(), "en"))
        elif _JA.search(s):
            out.append((s, "ja"))
        elif _LATIN.search(s):              # mixed tokens MeCab left joined
            out.extend((m.group(0).lower(), "en") for m in _LATIN.finditer(s))
        elif s.strip():
            out.append((s, "x"))
    return out


def utterance_stats(tokens: list[tuple[str, str]]) -> dict | None:
    langs = [l for _, l in tokens if l != "x"]
    if not langs:
        return None
    n, u = len(tokens), sum(1 for _, l in tokens if l == "x")
    c = Counter(langs)
    cmi = 100 * (1 - max(c.values()) / (n - u)) if n - u else 0.0
    spans: list[tuple[str, int]] = []
    for l in langs:
        if spans and spans[-1][0] == l:
            spans[-1] = (l, spans[-1][1] + 1)
        else:
            spans.append((l, 1))
    return {"cmi": cmi, "switches": len(spans) - 1,
            "i_index": (len(spans) - 1) / (len(langs) - 1) if len(langs) > 1 else 0.0,
            "spans": spans, "counts": c, "en_start": langs[0] == "en",
            "en_tokens": [t for t, l in tokens if l == "en"]}


def profile(texts: list[str]) -> dict:
    us = [s for s in (utterance_stats(tag_tokens(t)) for t in texts) if s]
    counts = Counter()
    for s in us:
        counts.update(s["counts"])
    p = [counts[l] / sum(counts.values()) for l in ("ja", "en") if counts[l]]
    sq = sum(x * x for x in p)
    m_index = (1 - sq) / ((2 - 1) * sq) if sq else 0.0
    span_lens = [n for s in us for _, n in s["spans"]]
    en_spans = [n for s in us for l, n in s["spans"] if l == "en"]
    mu, sd = statistics.fmean(span_lens), statistics.pstdev(span_lens)
    hist = Counter(span_lens)
    tot = sum(hist.values())
    ent = -sum(v / tot * math.log2(v / tot) for v in hist.values())
    en_tok = [t for s in us for t in s["en_tokens"]]
    cs = [s for s in us if len(s["counts"]) > 1]
    return {"utterances": len(us), "cs_utterances": len(cs),
            "cmi": statistics.fmean(s["cmi"] for s in us),
            "m_index": m_index,
            "i_index": statistics.fmean(s["i_index"] for s in us),
            "switches": statistics.fmean(s["switches"] for s in us),
            "en_span_len": statistics.fmean(en_spans) if en_spans else 0.0,
            "single_word_insert": (sum(1 for n in en_spans if n == 1) / len(en_spans)) if en_spans else 0.0,
            "burstiness": (sd - mu) / (sd + mu) if sd + mu else 0.0,
            "span_entropy": ent,
            "en_func_share": (sum(1 for t in en_tok if t in EN_FUNCTION) / len(en_tok)) if en_tok else 0.0,
            "en_start": sum(1 for s in us if s["en_start"]) / len(us)}


def _read(spec: str) -> tuple[str, list[str]]:
    name, _, rest = spec.partition("=")
    path, _, field = rest.partition(":")
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    key = field or next(k for k in ("reference", "transcript", "text") if k in rows[0])
    return name, [r[key].replace("**", "") for r in rows if r.get(key)]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    cols = ["utterances", "cs_utterances", "cmi", "m_index", "i_index", "switches", "en_span_len",
            "single_word_insert", "burstiness", "span_entropy", "en_func_share", "en_start"]
    print("| corpus | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for spec in argv:
        name, texts = _read(spec)
        p = profile(texts)
        print(f"| {name} | " + " | ".join(f"{p[c]:.3g}" if isinstance(p[c], float) else str(p[c])
                                          for c in cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
