#!/usr/bin/env python3
"""Katakana <-> English loanword lexicon for script-lenient scoring.

Strict MER charges a correct katakana rendering of an English word, e.g.
ミーティング for "meeting", as up to 6 character errors. HiKE-style lenient
scoring accepts either script for loanwords. It must accept only
*transliterations*, never translations: パン "bread" comes from Portuguese
pão, and "bread" must not match it.

The lexicon is built from JMdict (EDRDG, CC BY-SA 4.0; not redistributed
here) in two steps:
  candidates  katakana-only entries paired with (a) their English lsource word
              when present and (b) their English glosses. Only 2,413 of 37,008
              katakana entries carry an English lsource, so the glosses are needed.
  gate        a pair is kept only if the katakana and the English word are
              phonetically similar (consonant-skeleton similarity >= threshold).
              The threshold is calibrated on the lsource-annotated pairs
              (positives) against random mismatched pairs (negatives); see
              `calibrate`.

Usage:
  python eval/loanwords.py build --jmdict JMdict_e.gz --out loanwords_v1.json
  python eval/loanwords.py calibrate --jmdict JMdict_e.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

KATA_RE = re.compile(r"^[ァ-ヺー・]+$")
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
# English candidates: plain ASCII words (lsource and glosses alike; no affixes like "pan-")
_CANDIDATE = re.compile(r"[a-z][a-z' ]*[a-z]|[a-z]")

# ── katakana -> romaji (Hepburn-ish, enough for a phonetic skeleton) ─────────
_BASE = dict(zip(
    "アイウエオカキクケコガギグゲゴサシスセソザジズゼゾタチツテトダヂヅデドナニヌネノ"
    "ハヒフヘホバビブベボパピプペポマミムメモヤユヨラリルレロワヰヱヲンヴ",
    ["a", "i", "u", "e", "o", "ka", "ki", "ku", "ke", "ko", "ga", "gi", "gu", "ge", "go",
     "sa", "shi", "su", "se", "so", "za", "ji", "zu", "ze", "zo", "ta", "chi", "tsu", "te", "to",
     "da", "ji", "zu", "de", "do", "na", "ni", "nu", "ne", "no", "ha", "hi", "fu", "he", "ho",
     "ba", "bi", "bu", "be", "bo", "pa", "pi", "pu", "pe", "po", "ma", "mi", "mu", "me", "mo",
     "ya", "yu", "yo", "ra", "ri", "ru", "re", "ro", "wa", "wi", "we", "wo", "n", "vu"]))
_SMALL_V = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}
_SMALL_Y = {"ャ": "a", "ュ": "u", "ョ": "o"}
_CONS = {"shi": "sh", "chi": "ch", "ji": "j", "tsu": "ts", "fu": "f", "u": "w", "vu": "v"}


def kata_to_romaji(kata: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(kata):
        if ch in ("ー", "・", "ッ"):
            continue
        nxt = kata[i + 1] if i + 1 < len(kata) else ""
        if ch in _SMALL_V or ch in _SMALL_Y:
            continue                              # consumed with the previous kana
        base = _BASE.get(ch, "")
        if nxt in _SMALL_Y and base.endswith("i"):
            cons = _CONS.get(base, base[:-1])
            base = cons + ("" if cons in ("sh", "ch", "j") else "y") + _SMALL_Y[nxt]
        elif nxt in _SMALL_V:
            base = _CONS.get(base, base[:-1] if len(base) > 1 else base) + _SMALL_V[nxt]
        out.append(base)
    return "".join(out)


# ── coarse consonant skeletons ────────────────────────────────────────────────
def _skeleton(s: str) -> str:
    """Consonant classes, vowels dropped: l/r->r, v->b, f/h->h, sh/s/th->s,
    ch/ts->t, j/g(soft)->z, c->k/s, x->ks, q->k, w and y dropped."""
    s = re.sub(r"(.)\1+", r"\1", s)               # collapse doubles
    s = s.replace("sh", "s").replace("ch", "t").replace("ts", "t").replace("th", "s")
    s = s.replace("ph", "h").replace("ck", "k").replace("qu", "k").replace("x", "ks")
    s = re.sub(r"c(?=[eiy])", "s", s).replace("c", "k").replace("q", "k")
    s = s.replace("j", "z").replace("v", "b").replace("f", "h").replace("l", "r")
    s = re.sub(r"[aeiouwy]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def english_skeletons(word: str) -> set[str]:
    """Variants for the ambiguous English spellings (ch = k or t; soft g; silent
    non-prevocalic r; silent final e; gh)."""
    w = re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", word.lower()))
    if not w:
        return set()
    w = re.sub(r"e$", "", w) if len(w) > 3 else w
    forms = {w, w.replace("gh", ""), re.sub(r"r(?![aeiouy])", "", w)}
    forms |= {f.replace("ch", "k") for f in forms}
    forms |= {re.sub(r"g(?=[eiy])", "j", f) for f in forms}
    return {_skeleton(f) for f in forms if f}


def kata_skeleton(kata: str) -> str:
    r = kata_to_romaji(kata)
    r = r.replace("n", "n")                         # syllabic n stays n
    return _skeleton(r)


_NEAR = {frozenset(p) for p in ("kg", "td", "sz", "pb", "hb", "tz", "nm")}


def _dist(a: str, b: str) -> float:
    """Levenshtein with half-cost substitutions for voicing/place neighbours."""
    prev = [float(j) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)]
        for j, cb in enumerate(b, 1):
            sub = 0.0 if ca == cb else (0.5 if frozenset((ca, cb)) in _NEAR else 1.0)
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1]


def phonetic_similarity(kata: str, english: str) -> float:
    """1 - normalised skeleton distance, best over English spelling variants."""
    k = kata_skeleton(kata)
    best = 0.0
    for e in english_skeletons(english.replace(" ", "")):
        if not k and not e:
            continue
        best = max(best, 1 - _dist(k, e) / max(len(k), len(e), 1))
    return best


# ── JMdict ────────────────────────────────────────────────────────────────────
def iter_katakana_entries(jmdict: str):
    """Yield (katakana readings, English lsource words, English glosses) for
    katakana-only JMdict entries."""
    import xml.etree.ElementTree as ET
    opener = gzip.open if jmdict.endswith(".gz") else open
    for _, el in ET.iterparse(opener(jmdict), events=("end",)):
        if el.tag != "entry":
            continue
        rebs = [r.findtext("reb") or "" for r in el.findall("r_ele")]
        if el.find("k_ele") is None and rebs and all(KATA_RE.match(r) for r in rebs):
            src, gloss = [], []
            for s in el.findall("sense"):
                for l in s.findall("lsource"):
                    t = (l.text or "").strip().lower()
                    if l.get(XML_LANG, "eng") == "eng" and _CANDIDATE.fullmatch(t):
                        src.append(t)
                for g in s.findall("gloss"):
                    t = re.sub(r"\(.*?\)", "", g.text or "").strip().lower()
                    if _CANDIDATE.fullmatch(t) and len(t.split()) <= 3:
                        gloss.append(t)
            yield rebs, src, gloss
        el.clear()


def calibrate(jmdict: str, seed: int = 0) -> dict:
    """Precision/recall of the gate at several thresholds: positives are
    (katakana, lsource word) pairs; negatives pair each katakana with the
    lsource word of a random other entry."""
    pos = [(r[0], s[0]) for r, s, _ in iter_katakana_entries(jmdict) if s]
    rng = random.Random(seed)
    words = [w for _, w in pos]
    neg = [(k, words[rng.randrange(len(words))]) for k, _ in pos]
    sp = [phonetic_similarity(k, w) for k, w in pos]
    sn = [phonetic_similarity(k, w) for k, w in neg]
    out = {}
    for t in (0.5, 0.6, 0.67, 0.75, 0.8):
        tp, fp = sum(s >= t for s in sp), sum(s >= t for s in sn)
        out[t] = {"recall": round(tp / len(sp), 3),
                  "false_accept": round(fp / len(sn), 3),
                  "precision_at_1to1": round(tp / max(tp + fp, 1), 3)}
    return {"n_pos": len(pos), "thresholds": out}


def build(jmdict: str, threshold: float) -> dict[str, list[str]]:
    lex: dict[str, set[str]] = {}
    for rebs, src, gloss in iter_katakana_entries(jmdict):
        cands = list(dict.fromkeys(src + gloss))
        for r in rebs:
            key = r.replace("・", "")
            for c in cands:
                if phonetic_similarity(key, c) >= threshold:
                    lex.setdefault(key, set()).add(c)
    return {k: sorted(v) for k, v in sorted(lex.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "calibrate"])
    ap.add_argument("--jmdict", required=True)
    ap.add_argument("--threshold", type=float, default=0.67)
    ap.add_argument("--out", default="loanwords_v1.json")
    args = ap.parse_args()
    if args.cmd == "calibrate":
        print(json.dumps(calibrate(args.jmdict), indent=2))
        return 0
    lex = build(args.jmdict, args.threshold)
    meta = {"source": "JMdict (EDRDG, CC BY-SA 4.0)", "threshold": args.threshold,
            "entries": len(lex)}
    Path(args.out).write_text(json.dumps({"meta": meta, "lexicon": lex}, ensure_ascii=False),
                              encoding="utf-8")
    print(f"wrote {args.out}: {len(lex)} katakana keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
