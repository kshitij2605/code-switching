#!/usr/bin/env python3
"""Code-switching ASR scoring harness (scorer v2.3).

Reads one JSONL of {id, reference, hypothesis} per model and emits a metrics
table with:
  - MER              : Mixed Error Rate over a mixed token stream
                       (each Japanese char = 1 token, each English word = 1 token).
  - JA-CER           : Character Error Rate over the Japanese-only character stream
                       (Japanese has no word boundaries -> chars).
  - JA-WER           : Word Error Rate over fugashi/MeCab-segmented Japanese words
                       (depends on the installed dictionary; recorded in --json).
  - EN-WER           : Word Error Rate over the English (Latin) word stream.
  - ScriptAcc        : (v1, lenient) of the English words in the reference, the
                       fraction the model rendered in Latin script (vs. katakana,
                       other Japanese, or dropped). NOTE: *any* Latin token counts,
                       including misrecognised words, romaji and translations, so a
                       system can raise it by over-romanising. Always read it
                       together with ScriptAcc_strict and KataKeep.
  - ScriptAcc_strict : (v2) fraction of English reference words reproduced exactly
                       in Latin script ("Latin AND correct").
  - KataKeep         : (v2) of the katakana runs (>= 2 chars) in the reference
                       (loanwords, foreign names), the fraction the hypothesis keeps
                       in katakana rather than Latinising or dropping. Two-sided
                       companion to ScriptAcc: romanising everything scores a high
                       ScriptAcc but a low KataKeep.
  - MER_lenient      : (v2.1, with --loanwords) MER after rewriting the hypothesis's
                       loanwords into the reference's script: katakana -> English
                       where the reference has the English word, and English ->
                       katakana where it has the katakana. The lexicon is JMdict-
                       derived and phonetically gated, so translations are never
                       accepted (eval/loanwords.py). The reference is untouched, so
                       the denominator equals MER's, and MER - MER_lenient is the
                       error from script choice alone.
  - MER_capped       : (v2.2) MER with each utterance's errors capped at its number
                       of reference tokens. A runaway hypothesis (a repetition loop,
                       a hallucinated continuation) then costs at most 100% of its
                       own utterance instead of dominating the pooled rate.
  - Runaway          : (v2.2) fraction of utterances whose MER errors exceed twice
                       their reference tokens. Read MER, MER_capped and Runaway
                       together: a large MER - MER_capped gap means a few runaway
                       outputs drive the pooled MER.
  - CER_yv           : (v2.3) the character error rate of Ye & Vickers (Interspeech
                       2026, arXiv 2607.02757), for comparison with their published
                       numbers: on the raw reference and hypothesis, lowercase, strip
                       punctuation (Unicode category P*) and all whitespace, then
                       Levenshtein distance over characters / reference characters.
                       English words therefore count per letter. Their paper does not
                       mention Unicode normalisation, so none is applied.

Identical normalization is applied to references AND every model's hypotheses
before scoring. All error rates are corpus-aggregated (sum of edits / sum of
reference tokens), which is the standard way to pool WER/CER across utterances.

Details (v2.0):
  * normalization: 々 〆 〇 are Japanese characters; the katakana middle dot ・ and
    ゠ are word separators; intra-word apostrophes are kept ("don't" is one
    token); accented Latin letters are kept ("café").
  * switch report: switch-free utterances go to bucket "none";
    distances and downstream errors are indexed by *reference* position; "inter" = the English word sits in a sentence with no
    Japanese characters.
  * bootstrap: NaN resamples are excluded (n_valid reported); optional cluster
    bootstrap (--cluster-key, e.g. speaker); paired bootstrap aligns utterances by
    id and reports a two-sided bootstrap p-value.
  * small-denominator guard: rates over fewer than --min-denominator reference
    tokens are flagged and shown as n/a in the table (raw values stay in JSON).
  * provenance: scorer version and the fugashi dictionary are written to --json.

Known limitation: whether an English word that appears in katakana is a
legitimate loanword (e.g. ミーティング) cannot be decided from the text alone;
ScriptAcc treats every English reference word as "should be Latin" and KataKeep
treats every reference katakana run as "should stay katakana". Pronunciation-
grounded, loanword-aware references (L1/L2/L3 labels) are planned for the
Phase 1 metric suite.

Usage:
    python eval/score.py whisper:preds_whisper.jsonl lfm:preds_lfm.jsonl
    python eval/score.py preds.jsonl                       # label from filename
    python eval/score.py --no-fillers whisper:preds.jsonl  # keep English fillers
    python eval/score.py whisper:preds.jsonl --json out.json
    python eval/score.py model:preds.jsonl --bootstrap 1000 --ci 95
    python eval/score.py a:a.jsonl b:b.jsonl --bootstrap 10000 --paired
    python eval/score.py model:preds.jsonl --bootstrap 1000 --cluster-key speaker
    python eval/score.py model:preds.jsonl --switch-report switch.json --audit audit.json
    python eval/score.py model:preds.jsonl --loanwords loanwords_v1.json   # + MER_lenient
      (build the lexicon: python eval/loanwords.py build --jmdict JMdict_e.gz
       --out loanwords_v1.json; on mactrn01 it is in /data/code-switching/data/lexicon)

Each input line: {"id": "...", "reference": "...", "hypothesis": "..."}
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

try:
    import regex as _re  # better Unicode property support if available
except Exception:  # pragma: no cover - stdlib fallback
    import re as _re

SCORER_VERSION = "2.3"
# An utterance is a runaway when its MER errors exceed this multiple of its reference tokens.
RUNAWAY_FACTOR = 2

# Rates computed over fewer reference tokens than this are flagged as unreliable
# (e.g. EN-WER on a Japanese-only set with ~30 English reference words).
DEFAULT_MIN_DENOMINATOR = 50
# KataKeep over fewer reference katakana runs than this is flagged.
KATA_MIN_RUNS_REPORT = 20
# Minimum length (in characters) of a reference katakana run counted by KataKeep.
KATA_MIN_RUN = 2

METRIC_KEYS = ("MER", "JA_CER", "JA_WER", "EN_WER", "ScriptAcc", "ScriptAcc_strict", "KataKeep",
               "MER_lenient", "MER_capped", "Runaway", "CER_yv")

# ---------------------------------------------------------------------------
# Character / token classification
# ---------------------------------------------------------------------------

# English fillers removed by default (whole-token). We deliberately do NOT strip
# Japanese fillers (あの, えーと, ...) by regex: without morphological analysis
# that corrupts legitimate Japanese substrings. Documented choice.
ENGLISH_FILLERS = {"um", "uh", "uhh", "umm", "er", "err", "erm", "hmm", "mm"}

# Characters outside the kana/Han blocks that behave like Han characters in
# running Japanese text: 々 (iteration mark), 〆, 〇 (ideographic zero), 〻.
_JA_EXTRA = {"\u3005", "\u3006", "\u3007", "\u303b"}
# Katakana-block punctuation that separates words instead of spelling them:
# ゠ (double hyphen) and ・ (katakana middle dot, e.g. ジョン・スミス).
_JA_SEPARATORS = {"\u30a0", "\u30fb"}
# Apostrophes kept when they sit between two Latin letters ("don't").
_APOSTROPHES = {"'", "\u2019", "\u02bc"}


def _is_hiragana(ch: str) -> bool:
    return 0x3041 <= ord(ch) <= 0x309F


def _is_katakana(ch: str) -> bool:
    if ch in _JA_SEPARATORS:
        return False
    o = ord(ch)
    # Katakana (incl. prolonged sound mark ー), half-width katakana.
    return (0x30A1 <= o <= 0x30FF) or (0xFF66 <= o <= 0xFF9D)


def _is_kana(ch: str) -> bool:
    return _is_hiragana(ch) or _is_katakana(ch)


def _is_han(ch: str) -> bool:
    if ch in _JA_EXTRA:
        return True
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF) or (0xF900 <= o <= 0xFAFF)


def _is_japanese_char(ch: str) -> bool:
    return _is_kana(ch) or _is_han(ch)


@functools.lru_cache(maxsize=None)
def _is_latin_letter(ch: str) -> bool:
    """ASCII letters plus accented Latin letters (é, ü, ñ, ...)."""
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return True
    if ord(ch) < 0x80 or not ch.isalpha():
        return False
    return unicodedata.name(ch, "").startswith("LATIN")


def is_english_word(tok: str) -> bool:
    """A Latin-letter token (optionally with digits / internal apostrophes)."""
    if not tok or not _is_latin_letter(tok[0]):
        return False
    return all(_is_latin_letter(c) or c.isdigit() or c == "'" for c in tok)


def is_katakana_token(tok: str) -> bool:
    return len(tok) > 0 and all(_is_katakana(c) for c in tok)


# ---------------------------------------------------------------------------
# Normalization (applied identically to reference and hypothesis)
# ---------------------------------------------------------------------------


def normalize(text: str, remove_fillers: bool = True) -> str:
    """NFKC (folds full/half-width & most number forms) -> lowercase -> strip
    punctuation (keeping intra-word apostrophes) -> collapse whitespace.
    Optionally drop English fillers."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch.isspace():
            out.append(" ")
        elif _is_latin_letter(ch) or ch.isdigit() or _is_japanese_char(ch):
            out.append(ch)
        elif (ch in _APOSTROPHES and 0 < i < n - 1
              and _is_latin_letter(text[i - 1]) and _is_latin_letter(text[i + 1])):
            out.append("'")
        else:
            out.append(" ")
    text = _re.sub(r"\s+", " ", "".join(out)).strip()
    if remove_fillers and text:
        text = " ".join(t for t in text.split(" ") if t not in ENGLISH_FILLERS)
    return text


# ---------------------------------------------------------------------------
# Tokenizers for each metric's token stream
# ---------------------------------------------------------------------------


def mixed_tokens(norm_text: str) -> list[str]:
    """Mixed stream: each Japanese char is its own token; runs of Latin letters
    (and digits / apostrophes attached to them) form word tokens; standalone
    digit runs form number tokens."""
    tokens: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in norm_text:
        if ch == " ":
            flush()
        elif _is_japanese_char(ch):
            flush()
            tokens.append(ch)  # one token per Japanese char
        else:  # latin letter, digit or intra-word apostrophe
            buf.append(ch)
    flush()
    return tokens


def japanese_chars(norm_text: str) -> list[str]:
    """Japanese-only character stream (kana + Han) for CER."""
    return [c for c in norm_text if _is_japanese_char(c)]


def english_words(norm_text: str) -> list[str]:
    """English (Latin) word stream for WER."""
    return [t for t in mixed_tokens(norm_text) if is_english_word(t)]


_TAGGER = None


def _tagger():
    """MeCab tagger pinned to unidic-lite (UniDic 2.1.2), the dictionary in uv.lock.

    A bare ``fugashi.Tagger()`` silently prefers the full ``unidic`` package when
    it is installed, which changes JA-WER segmentation, so
    the dictionary directory is passed explicitly and a missing unidic-lite is an
    error rather than a fallback."""
    global _TAGGER
    if _TAGGER is None:
        import fugashi
        try:
            import unidic_lite
        except ImportError as e:
            raise RuntimeError("JA-WER needs the pinned unidic-lite dictionary; run the scorer "
                               "in the locked environment (docs/environment.md)") from e
        dicdir = unidic_lite.DICDIR
        _TAGGER = fugashi.Tagger(f'-d "{dicdir}" -r "{os.path.join(dicdir, "mecabrc")}"')
    return _TAGGER


def _dist_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def ja_tokenizer_info() -> dict:
    """Which fugashi/MeCab dictionary JA-WER was computed with (provenance)."""
    try:
        info = {"fugashi": _dist_version("fugashi"), "unidic-lite": _dist_version("unidic-lite"),
                "dictionaries": []}
        for d in getattr(_tagger(), "dictionary_info", []) or []:
            info["dictionaries"].append({"filename": d.get("filename"),
                                         "version": d.get("version")})
        return info
    except Exception as e:  # pragma: no cover - depends on local install
        return {"error": f"{type(e).__name__}: {e}"}


def japanese_words(norm_text: str) -> list[str]:
    """Japanese word stream (fugashi/MeCab segmentation) for JA-WER. Keeps tokens
    that contain at least one Japanese char (drops Latin/number tokens)."""
    if not norm_text.strip():
        return []
    return [tok.surface for tok in _tagger()(norm_text)
            if tok.surface and any(_is_japanese_char(c) for c in tok.surface)]


# ---------------------------------------------------------------------------
# Edit distance with backtrace
# ---------------------------------------------------------------------------


@dataclass
class EditCounts:
    sub: int = 0
    dele: int = 0
    ins: int = 0
    ref_len: int = 0

    def __iadd__(self, other: "EditCounts") -> "EditCounts":
        self.sub += other.sub
        self.dele += other.dele
        self.ins += other.ins
        self.ref_len += other.ref_len
        return self

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def rate(self) -> float:
        return self.errors / self.ref_len if self.ref_len else 0.0


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Levenshtein alignment. Returns ops: (kind, ref_tok, hyp_tok) where kind in
    {eq, sub, del, ins}. O(n*m); fine for utterance-length sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j = n, m
    ops: list[tuple[str, str | None, str | None]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            kind = "eq" if ref[i - 1] == hyp[j - 1] else "sub"
            ops.append((kind, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def annotate_ops(ops: Sequence[tuple[str, str | None, str | None]]
                 ) -> list[tuple[int | None, int, str, str | None, str | None]]:
    """Attach reference positions to alignment ops.

    Returns (ref_idx, anchor, kind, ref_tok, hyp_tok). For eq/sub/del ops
    ``ref_idx`` is the reference token index; insertions have ``ref_idx=None``
    and ``anchor`` = index of the next reference token (the insertion sits just
    before it). Use this instead of the op's list position whenever a reference
    index is needed: insertions shift op positions relative to the reference.
    """
    out = []
    ri = 0
    for kind, r, h in ops:
        if r is not None:
            out.append((ri, ri, kind, r, h))
            ri += 1
        else:
            out.append((None, ri, kind, r, h))
    return out


def _counts_from_ops(ops, ref_len: int) -> EditCounts:
    c = EditCounts(ref_len=ref_len)
    for kind, _, _ in ops:
        if kind == "sub":
            c.sub += 1
        elif kind == "del":
            c.dele += 1
        elif kind == "ins":
            c.ins += 1
    return c


def edit_counts(ref: list[str], hyp: list[str]) -> EditCounts:
    return _counts_from_ops(align(ref, hyp), len(ref))


# ---------------------------------------------------------------------------
# Script accuracy (English side) and katakana keep-rate (Japanese side)
# ---------------------------------------------------------------------------

_SCRIPT_FIELDS = ("en_ref", "latin_exact", "latin_other", "katakana", "other_jp", "dropped")


@dataclass
class ScriptCounts:
    en_ref: int = 0          # English words in the reference
    latin_exact: int = 0     # ...rendered in Latin script AND identical to the reference word
    latin_other: int = 0     # ...rendered as some other Latin token (wrong word, romaji, translation)
    katakana: int = 0        # ...forced into katakana
    other_jp: int = 0        # ...replaced by other Japanese (kanji/hiragana)
    dropped: int = 0         # ...dropped entirely (minority language lost)

    def __iadd__(self, other: "ScriptCounts") -> "ScriptCounts":
        for f in _SCRIPT_FIELDS:
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    @property
    def latin(self) -> int:
        """Any Latin rendering (v1 definition, used by ScriptAcc)."""
        return self.latin_exact + self.latin_other

    @property
    def accuracy(self) -> float:
        return self.latin / self.en_ref if self.en_ref else float("nan")

    @property
    def strict_accuracy(self) -> float:
        return self.latin_exact / self.en_ref if self.en_ref else float("nan")


@dataclass
class KataCounts:
    runs: int = 0        # katakana runs (>= KATA_MIN_RUN chars) in the reference
    kept: int = 0        # ...kept (mostly) in katakana by the hypothesis
    latinized: int = 0   # ...replaced by Latin-script tokens
    other: int = 0       # ...dropped or replaced by other Japanese

    def __iadd__(self, other: "KataCounts") -> "KataCounts":
        for f in ("runs", "kept", "latinized", "other"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    @property
    def keep_rate(self) -> float:
        return self.kept / self.runs if self.runs else float("nan")


def _token_script_category(r: str, h: str | None) -> str:
    """Category for one aligned English reference token (v1 categories)."""
    if h is None:
        return "dropped"
    if is_english_word(h):
        return "latin"
    if is_katakana_token(h):
        return "katakana"
    return "other_jp"


def script_counts(ref_mixed: list[str], hyp_mixed: list[str], ops=None) -> ScriptCounts:
    """Align mixed token streams; for every English-word token in the reference,
    inspect what the hypothesis put there."""
    sc = ScriptCounts()
    ops = align(ref_mixed, hyp_mixed) if ops is None else ops
    for kind, r, h in ops:
        if r is None or not is_english_word(r):
            continue
        sc.en_ref += 1
        cat = _token_script_category(r, None if kind == "del" else h)
        if cat == "latin":
            if h == r:
                sc.latin_exact += 1
            else:
                sc.latin_other += 1
        elif cat == "katakana":
            sc.katakana += 1
        elif cat == "other_jp":
            sc.other_jp += 1
        else:
            sc.dropped += 1
    return sc


def script_counts_detailed(ref_mixed: list[str], hyp_mixed: list[str], ops=None) -> list[dict]:
    """Per-English-word script decision for switch / audit analysis."""
    out = []
    ops = align(ref_mixed, hyp_mixed) if ops is None else ops
    for kind, r, h in ops:
        if r is None or not is_english_word(r):
            continue
        hh = None if kind == "del" else h
        out.append({"ref": r, "hyp": hh, "kind": kind,
                    "category": _token_script_category(r, hh),
                    "exact": hh == r})
    return out


def katakana_runs(tokens: list[str], min_len: int = KATA_MIN_RUN) -> list[tuple[int, int]]:
    """Maximal runs [start, end) of consecutive katakana tokens of length >= min_len."""
    runs = []
    i = 0
    while i < len(tokens):
        if is_katakana_token(tokens[i]):
            j = i
            while j < len(tokens) and is_katakana_token(tokens[j]):
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def katakana_counts(ref_mixed: list[str], hyp_mixed: list[str], ops=None) -> KataCounts:
    """For each reference katakana run, did the hypothesis keep it in katakana?

    kept      : at least half of the run's characters align to katakana
                (exact match or katakana substitution);
    latinized : otherwise, if any Latin-script token was placed over the run
                (substitution inside it or insertion adjacent to it);
    other     : otherwise (dropped, kanji/hiragana, ...).
    """
    kc = KataCounts()
    runs = katakana_runs(ref_mixed)
    if not runs:
        return kc
    ops = align(ref_mixed, hyp_mixed) if ops is None else ops
    ann = annotate_ops(ops)
    for s, e in runs:
        kc.runs += 1
        kept_pos = 0
        hyp_toks: list[str] = []
        for ref_idx, anchor, kind, _r, h in ann:
            if ref_idx is not None:
                if s <= ref_idx < e:
                    if kind != "del" and h is not None:
                        hyp_toks.append(h)
                    if kind == "eq" or (kind == "sub" and h is not None and is_katakana_token(h)):
                        kept_pos += 1
            elif s <= anchor <= e and h is not None:
                hyp_toks.append(h)
        if kept_pos * 2 >= (e - s):
            kc.kept += 1
        elif any(is_english_word(t) for t in hyp_toks):
            kc.latinized += 1
        else:
            kc.other += 1
    return kc


# ---------------------------------------------------------------------------
# Switch-point analysis
# ---------------------------------------------------------------------------

def _token_lang(tok: str) -> str:
    if _is_japanese_char(tok[0]):
        return "ja"
    if is_english_word(tok):
        return "en"
    return "other"


def switch_points(tokens: list[str]) -> list[int]:
    """Return boundary positions where language switches between EN and JP.

    A boundary between token i-1 and token i is represented by index i.
    """
    pts = []
    for i in range(1, len(tokens)):
        prev, cur = _token_lang(tokens[i - 1]), _token_lang(tokens[i])
        if (prev == "en" and cur == "ja") or (prev == "ja" and cur == "en"):
            pts.append(i)
    return pts


def _nearest_switch_distance(idx: int, points: list[int]) -> int | None:
    if not points:
        return None
    return min(abs(idx - p) for p in points)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split `text` into sentence-like spans using EN/JA sentence finals."""
    spans = []
    start = 0
    for m in _re.finditer(r"[。！？.!?]+", text):
        end = m.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _english_spans_in_original(original: str) -> list[tuple[str, int, int]]:
    """Find English-word-like spans in the original (cased) reference text.

    Returns tuples of (lowercased_word, start_char, end_char).
    """
    spans = []
    # Keep apostrophes and hyphens inside words; stop on punctuation/space/JP.
    for m in _re.finditer(r"[A-Za-z][A-Za-z0-9'’\-\.]*", original):
        spans.append((m.group(0).lower(), m.start(), m.end()))
    return spans


def _match_english_words_to_original(norm_words: list[str], original: str) -> list[tuple[int, int] | None]:
    """Align normalized English word tokens to original char spans in order.

    Returns a list the same length as `norm_words`; each element is either
    (start, end) char offsets in `original`, or None if no span matched.
    """
    orig_spans = _english_spans_in_original(original)
    mapping: list[tuple[int, int] | None] = []
    oi = 0
    for w in norm_words:
        # A normalized word may differ from the original (digits, full-width,
        # punctuation removed). Fall back to the next English span whose
        # lowercased form starts with the same letters, then advance.
        found = None
        while oi < len(orig_spans):
            ow, s, e = orig_spans[oi]
            if ow.startswith(w) or w.startswith(ow.replace("-", "").replace("'", "")):
                found = (s, e)
                oi += 1
                break
            oi += 1
        mapping.append(found)
    return mapping


def _switch_type(span: tuple[int, int] | None, original: str,
                 sentences: list[tuple[int, int]]) -> str:
    """'inter' if the English word sits in a sentence with no Japanese characters
    (the switch happens at a sentence boundary), 'intra' if its sentence also
    contains Japanese, 'unknown' if the word could not be located."""
    if span is None:
        return "unknown"
    s, _e = span
    for a, b in sentences:
        if a <= s < b:
            sent = original[a:b]
            return "intra" if any(_is_japanese_char(c) for c in sent) else "inter"
    return "unknown"


def _build_distance_buckets(max_dist: int = 3) -> dict:
    keys = [str(d) for d in range(max_dist)] + [f">={max_dist}", "none"]
    return {k: {"tokens": 0, "correct": 0} for k in keys}


def _bucket(d: int | None, max_dist: int) -> str:
    if d is None:
        return "none"  # no EN<->JP switch in the utterance
    if d < max_dist:
        return str(d)
    return f">={max_dist}"


def analyze_switch_points(reference: str, hypothesis: str, original: str,
                          remove_fillers: bool = True) -> dict:
    """Stratified analysis of errors around EN/JP switch points.

    Requires the *original* reference text (before normalization) so that
    sentence boundaries can be detected for inter/intra-sentential typing.
    """
    rn = normalize(reference, remove_fillers)
    hn = normalize(hypothesis, remove_fillers)
    ref_toks = mixed_tokens(rn)
    hyp_toks = mixed_tokens(hn)
    ops = align(ref_toks, hyp_toks)
    ann = annotate_ops(ops)

    points = switch_points(ref_toks)
    max_dist = 3
    all_by_dist = _build_distance_buckets(max_dist)
    en_by_dist = _build_distance_buckets(max_dist)
    ja_by_dist = _build_distance_buckets(max_dist)

    # Token-level correctness, indexed by *reference* position.
    ref_kind: dict[int, str] = {}
    for ref_idx, _anchor, kind, r, _h in ann:
        if ref_idx is None:
            continue
        ref_kind[ref_idx] = kind
        lang = _token_lang(r)
        b = _bucket(_nearest_switch_distance(ref_idx, points), max_dist)
        correct = kind == "eq"
        all_by_dist[b]["tokens"] += 1
        all_by_dist[b]["correct"] += int(correct)
        if lang == "en":
            en_by_dist[b]["tokens"] += 1
            en_by_dist[b]["correct"] += int(correct)
        elif lang == "ja":
            ja_by_dist[b]["tokens"] += 1
            ja_by_dist[b]["correct"] += int(correct)

    # Switch-type and span-length for English-script accuracy.
    sentences = _sentence_spans(original)
    en_word_indices = [i for i, t in enumerate(ref_toks) if is_english_word(t)]
    orig_spans = _match_english_words_to_original([ref_toks[i] for i in en_word_indices], original)

    script_detail = script_counts_detailed(ref_toks, hyp_toks, ops)
    type_counts = {"intra": {"en_ref": 0, "latin": 0},
                   "inter": {"en_ref": 0, "latin": 0},
                   "unknown": {"en_ref": 0, "latin": 0}}
    span_counts = {"single": {"en_ref": 0, "latin": 0},
                   "phrase": {"en_ref": 0, "latin": 0}}

    # Run lengths of consecutive English tokens.
    run_lengths = {}
    i = 0
    while i < len(ref_toks):
        if is_english_word(ref_toks[i]):
            j = i
            while j < len(ref_toks) and is_english_word(ref_toks[j]):
                j += 1
            for k in range(i, j):
                run_lengths[k] = j - i
            i = j
        else:
            i += 1

    for detail, idx, span in zip(script_detail, en_word_indices, orig_spans):
        stype = _switch_type(span, original, sentences)
        type_counts[stype]["en_ref"] += 1
        if detail["category"] == "latin":
            type_counts[stype]["latin"] += 1

        slen = "single" if run_lengths.get(idx, 1) == 1 else "phrase"
        span_counts[slen]["en_ref"] += 1
        if detail["category"] == "latin":
            span_counts[slen]["latin"] += 1

    # Downstream effect of a katakana substitution vs an exact Latin rendering:
    # error rate on the next 5 *reference* tokens.
    after_kat = {"tokens": 0, "errors": 0}
    after_lat = {"tokens": 0, "errors": 0}
    for det, idx in zip(script_detail, en_word_indices):
        if det["category"] == "katakana":
            entry = after_kat
        elif det["category"] == "latin" and det["exact"]:
            entry = after_lat
        else:
            continue
        for offset in range(1, 6):
            nxt = idx + offset
            if nxt >= len(ref_toks):
                break
            entry["tokens"] += 1
            if ref_kind.get(nxt) != "eq":
                entry["errors"] += 1

    return {
        "distance": all_by_dist,
        "en_distance": en_by_dist,
        "ja_distance": ja_by_dist,
        "switch_type": type_counts,
        "span_length": span_counts,
        "downstream": {"after_katakana": after_kat, "after_correct_latin": after_lat},
    }


def _acc_from_counts(counts: dict) -> dict:
    return {k: round(v["correct"] / v["tokens"], 4) if v["tokens"] else None
            for k, v in counts.items()}


def _scriptacc_from_counts(counts: dict) -> dict:
    return {k: round(v["latin"] / v["en_ref"], 4) if v["en_ref"] else None
            for k, v in counts.items()}


def _rate_from_counts(entry: dict) -> float | None:
    return round(entry["errors"] / entry["tokens"], 4) if entry["tokens"] else None


def switch_report_from_counts(total: dict) -> dict:
    """Turn aggregated raw counts into the published switch-point report."""
    downstream = total["downstream"]
    return {
        "token_accuracy_by_distance": _acc_from_counts(total["distance"]),
        "en_token_accuracy_by_distance": _acc_from_counts(total["en_distance"]),
        "ja_token_accuracy_by_distance": _acc_from_counts(total["ja_distance"]),
        "script_accuracy_by_switch_type": _scriptacc_from_counts(total["switch_type"]),
        "script_accuracy_by_span_length": _scriptacc_from_counts(total["span_length"]),
        "downstream_error_after_katakana": _rate_from_counts(downstream["after_katakana"]),
        "downstream_error_after_correct_latin": _rate_from_counts(downstream["after_correct_latin"]),
        "counts": total,
    }


def aggregate_switch_analysis(records: Iterable[dict], remove_fillers: bool = True) -> dict:
    """Aggregate switch-point counts over many utterances and return the report."""
    max_dist = 3
    total = {
        "distance": _build_distance_buckets(max_dist),
        "en_distance": _build_distance_buckets(max_dist),
        "ja_distance": _build_distance_buckets(max_dist),
        "switch_type": {"intra": {"en_ref": 0, "latin": 0},
                        "inter": {"en_ref": 0, "latin": 0},
                        "unknown": {"en_ref": 0, "latin": 0}},
        "span_length": {"single": {"en_ref": 0, "latin": 0},
                        "phrase": {"en_ref": 0, "latin": 0}},
        "downstream": {"after_katakana": {"tokens": 0, "errors": 0},
                       "after_correct_latin": {"tokens": 0, "errors": 0}},
    }
    for rec in records:
        counts = analyze_switch_points(rec["reference"], rec["hypothesis"],
                                       rec["reference"], remove_fillers)
        for group, buckets in counts.items():
            for k, v in buckets.items():
                for f, x in v.items():
                    total[group][k][f] += x
    return switch_report_from_counts(total)


# ---------------------------------------------------------------------------
# Per-model scoring
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    label: str
    n: int = 0
    mer: EditCounts = field(default_factory=EditCounts)
    ja: EditCounts = field(default_factory=EditCounts)      # JA char-level (CER)
    jaw: EditCounts = field(default_factory=EditCounts)     # JA word-level (WER)
    en: EditCounts = field(default_factory=EditCounts)
    script: ScriptCounts = field(default_factory=ScriptCounts)
    kata: KataCounts = field(default_factory=KataCounts)
    mer_len: EditCounts = field(default_factory=EditCounts)  # script-lenient MER (if lexicon)
    cer_yv: EditCounts = field(default_factory=EditCounts)   # Ye & Vickers CER
    lenient: bool = False
    cap_err: int = 0        # MER errors, each utterance capped at its reference tokens
    runaway: int = 0        # utterances with MER errors > RUNAWAY_FACTOR x reference tokens
    min_denominator: int = DEFAULT_MIN_DENOMINATOR

    @staticmethod
    def _ec(ec: EditCounts) -> dict:
        return {"sub": ec.sub, "del": ec.dele, "ins": ec.ins, "ref": ec.ref_len}

    def flags(self) -> dict:
        return {
            "en_small_denominator": self.script.en_ref < self.min_denominator,
            "ja_small_denominator": self.ja.ref_len < self.min_denominator,
            "kata_small_denominator": self.kata.runs < KATA_MIN_RUNS_REPORT,
        }

    def as_row(self) -> dict:
        return {
            "model": self.label,
            "n": self.n,
            "MER": round(self.mer.rate, 4),
            "JA_CER": round(self.ja.rate, 4),
            "JA_WER": round(self.jaw.rate, 4),
            "EN_WER": round(self.en.rate, 4),
            "ScriptAcc": (None if self.script.en_ref == 0 else round(self.script.accuracy, 4)),
            "ScriptAcc_strict": (None if self.script.en_ref == 0
                                 else round(self.script.strict_accuracy, 4)),
            "KataKeep": (None if self.kata.runs == 0 else round(self.kata.keep_rate, 4)),
            "MER_lenient": round(self.mer_len.rate, 4) if self.lenient else None,
            "MER_capped": round(self.cap_err / self.mer.ref_len, 4) if self.mer.ref_len else None,
            "CER_yv": round(self.cer_yv.rate, 4),
            "Runaway": round(self.runaway / self.n, 4) if self.n else None,
            "runaway_utts": self.runaway,
            "en_ref_words": self.script.en_ref,
            "ja_ref_chars": self.ja.ref_len,
            "ja_ref_words": self.jaw.ref_len,
            "kata_ref_runs": self.kata.runs,
            # per-metric edit decomposition (sub/del/ins/ref) — for UI drill-down
            "edits": {"mer": self._ec(self.mer), "ja_cer": self._ec(self.ja),
                      "ja_wer": self._ec(self.jaw), "en_wer": self._ec(self.en)},
            "script_breakdown": {
                "latin": self.script.latin,
                "latin_exact": self.script.latin_exact,
                "latin_other": self.script.latin_other,
                "katakana": self.script.katakana,
                "other_jp": self.script.other_jp,
                "dropped": self.script.dropped,
            },
            "kata_breakdown": {"kept": self.kata.kept, "latinized": self.kata.latinized,
                               "other": self.kata.other},
            "flags": self.flags(),
            "scorer_version": SCORER_VERSION,
        }


# ---------------------------------------------------------------------------
# Script-lenient MER (loanword transliterations accepted in either script)
# ---------------------------------------------------------------------------
# Strict MER charges a correct katakana rendering of an English reference word
# (ミーティング for "meeting") as several character errors, and the reverse too.
# MER_lenient keeps the reference fixed. It rewrites only those hypothesis spans
# whose script counterpart occurs in the reference, using a phonetically gated
# katakana <-> English loanword lexicon (eval/loanwords.py; JMdict-derived). The
# denominator therefore equals strict MER's, and MER - MER_lenient is the error
# due to script choice alone. Translations are never accepted, because the
# lexicon holds transliterations only.

_LOANWORDS: tuple[dict, dict, int] | None = None     # (kata->eng, eng->kata, max key length)
_LOANWORDS_META: dict | None = None
_KATA_RUN = re.compile(r"[ァ-ヺー]+")
_LATIN_SPAN = re.compile(r"[a-z0-9']+(?:\s+[a-z0-9']+)*")


def set_loanwords(lexicon: dict[str, list[str]] | None, meta: dict | None = None) -> None:
    """Install a katakana -> [english phrase] lexicon (None disables MER_lenient)."""
    global _LOANWORDS, _LOANWORDS_META
    if not lexicon:
        _LOANWORDS, _LOANWORDS_META = None, None
        return
    rev: dict[str, list[str]] = {}
    for k, cands in lexicon.items():
        for c in cands:
            rev.setdefault(c, []).append(k)
    _LOANWORDS = ({k: v for k, v in lexicon.items()}, rev, max(len(k) for k in lexicon))
    _LOANWORDS_META = meta or {"entries": len(lexicon)}


def load_loanwords(path: str) -> None:
    import hashlib
    raw = open(path, "rb").read()
    data = json.loads(raw.decode("utf-8"))
    meta = dict(data.get("meta", {}), path=path, sha256=hashlib.sha256(raw).hexdigest()[:16])
    set_loanwords(data["lexicon"], meta)


def lenient_hypothesis(rn: str, hn: str) -> str:
    """Rewrite hypothesis loanwords into the reference's script where the
    reference contains the counterpart (both inputs already normalised)."""
    if _LOANWORDS is None:
        return hn
    lex, rev, maxlen = _LOANWORDS
    ref_latin = set(english_words(rn))

    def kata_to_latin(m: re.Match) -> str:
        run, out, i = m.group(0), [], 0
        while i < len(run):
            for L in range(min(maxlen, len(run) - i), 1, -1):
                seg = run[i:i + L]
                hit = next((c for c in lex.get(seg, ()) if set(c.split()) <= ref_latin), None)
                if hit:
                    out.append(f" {hit} ")
                    i += L
                    break
            else:
                out.append(run[i])
                i += 1
        return "".join(out)

    def latin_to_kata(m: re.Match) -> str:
        toks, out, i = m.group(0).split(), [], 0
        while i < len(toks):
            for n in (3, 2, 1):
                phrase = " ".join(toks[i:i + n])
                if len(toks) - i >= n and phrase in rev:
                    k = next((k for k in rev[phrase] if k in rn), None)
                    if k:
                        out.append(f" {k} ")
                        i += n
                        break
            else:
                out.append(f" {toks[i]} ")
                i += 1
        return "".join(out)

    return _LATIN_SPAN.sub(latin_to_kata, _KATA_RUN.sub(kata_to_latin, hn))


def yv_chars(text: str) -> list[str]:
    """Ye & Vickers CER units: lowercased characters without punctuation or whitespace."""
    return [c for c in text.lower() if not c.isspace() and not unicodedata.category(c).startswith("P")]


def score_pair(reference: str, hypothesis: str, remove_fillers: bool = True) -> dict:
    """Score a single utterance; returns the per-metric counts for aggregation."""
    rn = normalize(reference, remove_fillers)
    hn = normalize(hypothesis, remove_fillers)
    r_mixed, h_mixed = mixed_tokens(rn), mixed_tokens(hn)
    ops = align(r_mixed, h_mixed)
    parts = {
        "mer": _counts_from_ops(ops, len(r_mixed)),
        "ja": edit_counts(japanese_chars(rn), japanese_chars(hn)),
        "jaw": edit_counts(japanese_words(rn), japanese_words(hn)),
        "en": edit_counts(english_words(rn), english_words(hn)),
        "script": script_counts(r_mixed, h_mixed, ops),
        "kata": katakana_counts(r_mixed, h_mixed, ops),
        "cer_yv": edit_counts(yv_chars(reference), yv_chars(hypothesis)),
    }
    if _LOANWORDS is not None:
        hl = mixed_tokens(lenient_hypothesis(rn, hn))
        parts["mer_len"] = _counts_from_ops(align(r_mixed, hl), len(r_mixed))
    return parts


def _is_runaway(err: int, ref: int) -> bool:
    return err > RUNAWAY_FACTOR * max(ref, 1)


def _int_counts(parts: dict) -> dict:
    """Flatten one utterance's counts to plain ints."""
    def _ec(ec: EditCounts) -> dict:
        return {"sub": ec.sub, "del": ec.dele, "ins": ec.ins, "ref": ec.ref_len,
                "err": ec.errors}
    sc, kc = parts["script"], parts["kata"]
    e, r = parts["mer"].errors, parts["mer"].ref_len
    out = {
        "mer_cap": {"err": min(e, r), "ref": r},
        "cer_yv": _ec(parts["cer_yv"]),
        "runaway": int(_is_runaway(e, r)), "n_hyp": 1,
        "mer": _ec(parts["mer"]), "ja_cer": _ec(parts["ja"]),
        "ja_wer": _ec(parts["jaw"]), "en_wer": _ec(parts["en"]),
        "script": {
            "en_ref": sc.en_ref, "latin": sc.latin, "latin_exact": sc.latin_exact,
            "latin_other": sc.latin_other, "katakana": sc.katakana,
            "other_jp": sc.other_jp, "dropped": sc.dropped,
        },
        "kata": {"runs": kc.runs, "kept": kc.kept, "latinized": kc.latinized,
                 "other": kc.other},
    }
    if "mer_len" in parts:
        out["mer_len"] = _ec(parts["mer_len"])
    return out


def score_records(records: Iterable[dict], label: str, remove_fillers: bool = True,
                  min_denominator: int = DEFAULT_MIN_DENOMINATOR) -> ModelMetrics:
    mm = ModelMetrics(label=label, min_denominator=min_denominator)
    for rec in records:
        parts = score_pair(rec.get("reference", ""), rec.get("hypothesis", ""), remove_fillers)
        mm.n += 1
        mm.mer += parts["mer"]
        e, r = parts["mer"].errors, parts["mer"].ref_len
        mm.cap_err += min(e, r)
        mm.cer_yv += parts["cer_yv"]
        mm.runaway += _is_runaway(e, r)
        mm.ja += parts["ja"]
        mm.jaw += parts["jaw"]
        mm.en += parts["en"]
        mm.script += parts["script"]
        mm.kata += parts["kata"]
        if "mer_len" in parts:
            mm.mer_len += parts["mer_len"]
            mm.lenient = True
    return mm


def per_utterance_counts(records: Iterable[dict], remove_fillers: bool = True,
                         cluster_key: str | None = None) -> list[dict]:
    """Per-utterance integer counts (+ id / cluster) for bootstrap resampling."""
    out = []
    for rec in records:
        c = _int_counts(score_pair(rec.get("reference", ""),
                                   rec.get("hypothesis", ""), remove_fillers))
        c["id"] = rec.get("id")
        c["cluster"] = rec.get(cluster_key) if cluster_key else None
        out.append(c)
    return out


def _aggregate_rates(counts: Sequence[dict]) -> dict:
    """Pool a list of per-utterance int-count dicts into metric rates."""
    def _rate(err: int, ref: int) -> float:
        return err / ref if ref else float("nan")

    tot = {k: 0 for k in ("mer_err", "mer_ref", "ja_err", "ja_ref", "jaw_err", "jaw_ref",
                          "en_err", "en_ref", "latin", "latin_exact", "en_words",
                          "kata_runs", "kata_kept", "len_err", "len_ref",
                          "cap_err", "cap_ref", "runaway", "n_hyp", "yv_err", "yv_ref")}
    has_yv = bool(counts) and all("cer_yv" in c for c in counts)
    has_len = bool(counts) and all("mer_len" in c for c in counts)
    has_cap = bool(counts) and all("mer_cap" in c for c in counts)
    for c in counts:
        if has_len:
            tot["len_err"] += c["mer_len"]["err"]; tot["len_ref"] += c["mer_len"]["ref"]
        if has_yv:
            tot["yv_err"] += c["cer_yv"]["err"]; tot["yv_ref"] += c["cer_yv"]["ref"]
        if has_cap:
            tot["cap_err"] += c["mer_cap"]["err"]; tot["cap_ref"] += c["mer_cap"]["ref"]
            tot["runaway"] += c["runaway"]; tot["n_hyp"] += c["n_hyp"]
        tot["mer_err"] += c["mer"]["err"]; tot["mer_ref"] += c["mer"]["ref"]
        tot["ja_err"] += c["ja_cer"]["err"]; tot["ja_ref"] += c["ja_cer"]["ref"]
        tot["jaw_err"] += c["ja_wer"]["err"]; tot["jaw_ref"] += c["ja_wer"]["ref"]
        tot["en_err"] += c["en_wer"]["err"]; tot["en_ref"] += c["en_wer"]["ref"]
        tot["latin"] += c["script"]["latin"]; tot["en_words"] += c["script"]["en_ref"]
        tot["latin_exact"] += c["script"].get("latin_exact", 0)
        kata = c.get("kata", {})
        tot["kata_runs"] += kata.get("runs", 0); tot["kata_kept"] += kata.get("kept", 0)
    return {
        "MER": _rate(tot["mer_err"], tot["mer_ref"]),
        "JA_CER": _rate(tot["ja_err"], tot["ja_ref"]),
        "JA_WER": _rate(tot["jaw_err"], tot["jaw_ref"]),
        "EN_WER": _rate(tot["en_err"], tot["en_ref"]),
        "ScriptAcc": _rate(tot["latin"], tot["en_words"]),
        "ScriptAcc_strict": _rate(tot["latin_exact"], tot["en_words"]),
        "KataKeep": _rate(tot["kata_kept"], tot["kata_runs"]),
        "MER_lenient": _rate(tot["len_err"], tot["len_ref"]) if has_len else float("nan"),
        "MER_capped": _rate(tot["cap_err"], tot["cap_ref"]) if has_cap else float("nan"),
        "Runaway": _rate(tot["runaway"], tot["n_hyp"]) if has_cap else float("nan"),
        "CER_yv": _rate(tot["yv_err"], tot["yv_ref"]) if has_yv else float("nan"),
    }


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile on an already-sorted sequence (0..100)."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _summarize(dists: dict, ci: float, with_p: bool = False) -> dict:
    """Percentile CIs per metric; NaN resamples (zero denominators) are excluded."""
    lo_p = (100.0 - ci) / 2.0
    hi_p = 100.0 - lo_p
    out = {}
    for k, vals in dists.items():
        s = sorted(v for v in vals if not math.isnan(v))
        if not s:
            out[k] = {"mean": None, "lo": None, "hi": None, "n_valid": 0}
            continue
        d = {"mean": round(sum(s) / len(s), 4), "lo": round(_percentile(s, lo_p), 4),
             "hi": round(_percentile(s, hi_p), 4), "n_valid": len(s)}
        if with_p:
            le = sum(1 for v in s if v <= 0) / len(s)
            ge = sum(1 for v in s if v >= 0) / len(s)
            d["p_boot"] = round(min(1.0, 2.0 * min(le, ge)), 4)
        out[k] = d
    return out


def _cluster_groups(per_utts: Sequence[dict]) -> list[list[int]]:
    groups: dict = {}
    for i, c in enumerate(per_utts):
        groups.setdefault(c.get("cluster"), []).append(i)
    return list(groups.values())


def _resample_indices(n: int, groups: list[list[int]] | None, rng: random.Random) -> list[int]:
    if groups is None:
        return [rng.randrange(n) for _ in range(n)]
    idx: list[int] = []
    for _ in range(len(groups)):
        idx.extend(groups[rng.randrange(len(groups))])
    return idx


def bootstrap_ci(per_utts: Sequence[dict], n_samples: int, ci: float,
                 rng: random.Random, clustered: bool = False) -> dict:
    """Bootstrap percentile CIs for every rate over `per_utts`.

    With ``clustered=True`` whole clusters (the ``cluster`` field of each
    per-utterance dict, e.g. speaker) are resampled instead of utterances.
    Returns metric -> {"mean", "lo", "hi", "n_valid"}.
    """
    n = len(per_utts)
    groups = _cluster_groups(per_utts) if clustered else None
    dists = {k: [] for k in METRIC_KEYS}
    for _ in range(n_samples):
        sample = [per_utts[i] for i in _resample_indices(n, groups, rng)]
        for k, v in _aggregate_rates(sample).items():
            dists[k].append(v)
    return _summarize(dists, ci)


def align_records_by_id(rows_a: Sequence[dict], rows_b: Sequence[dict],
                        allow_mismatch: bool = False) -> tuple[list[dict], list[dict]]:
    """Pair two prediction files by utterance id (order of `rows_a`)."""
    ids_a = [r.get("id") for r in rows_a]
    ids_b = [r.get("id") for r in rows_b]
    for name, ids in (("first", ids_a), ("second", ids_b)):
        if any(i is None for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{name} file: paired scoring needs unique, non-null ids")
    by_b = {r["id"]: r for r in rows_b}
    common = [i for i in ids_a if i in by_b]
    if not allow_mismatch and (len(common) != len(ids_a) or len(common) != len(ids_b)):
        raise ValueError(f"id sets differ ({len(ids_a)} vs {len(ids_b)} rows, "
                         f"{len(common)} shared); use --allow-id-mismatch to score the "
                         f"intersection")
    by_a = {r["id"]: r for r in rows_a}
    return [by_a[i] for i in common], [by_b[i] for i in common]


def paired_bootstrap(per_utts_a: Sequence[dict], per_utts_b: Sequence[dict],
                     n_samples: int, ci: float, rng: random.Random,
                     clustered: bool = False) -> dict:
    """Paired bootstrap of (model_a - model_b) over the same utterances.

    Inputs must be aligned by utterance id (see align_records_by_id). Reports a
    percentile CI and a two-sided bootstrap p-value per metric.
    """
    if len(per_utts_a) != len(per_utts_b):
        raise ValueError("paired bootstrap requires the same number of utterances")
    ids_a = [c.get("id") for c in per_utts_a]
    ids_b = [c.get("id") for c in per_utts_b]
    if any(i is not None for i in ids_a + ids_b) and ids_a != ids_b:
        raise ValueError("paired bootstrap requires utterances aligned by id "
                         "(use align_records_by_id first)")
    n = len(per_utts_a)
    groups = _cluster_groups(per_utts_a) if clustered else None
    dists = {k: [] for k in METRIC_KEYS}
    for _ in range(n_samples):
        idx = _resample_indices(n, groups, rng)
        rates_a = _aggregate_rates([per_utts_a[i] for i in idx])
        rates_b = _aggregate_rates([per_utts_b[i] for i in idx])
        for k in METRIC_KEYS:
            dists[k].append(rates_a[k] - rates_b[k])
    return _summarize(dists, ci, with_p=True)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: invalid JSON: {e}")
    return rows


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _pct(v: float | None, hide: bool = False) -> str:
    if v is None or hide:
        return "n/a"
    return f"{v * 100:5.1f}%"


def render_table(models: list[ModelMetrics]) -> str:
    lenient = any(mm.lenient for mm in models)
    headers = ["Model", "N", "MER↓"] + (["MER-len↓"] if lenient else []) + [
               "MER-cap↓", "Runaway↓", "CER-YV↓", "JA-CER↓", "JA-WER↓", "EN-WER↓", "ScriptAcc↑",
               "SA-strict↑", "KataKeep↑", "EN-w", "JA-ch", "Kata"]
    rows = []
    for mm in models:
        r = mm.as_row()
        fl = r["flags"]
        en_small, ja_small = fl["en_small_denominator"], fl["ja_small_denominator"]
        rows.append([
            r["model"], str(r["n"]), _pct(r["MER"]),
            *([_pct(r["MER_lenient"])] if lenient else []),
            _pct(r["MER_capped"]), _pct(r["Runaway"]), _pct(r["CER_yv"]),
            _pct(r["JA_CER"], ja_small), _pct(r["JA_WER"], ja_small),
            _pct(r["EN_WER"], en_small), _pct(r["ScriptAcc"], en_small),
            _pct(r["ScriptAcc_strict"], en_small),
            _pct(r["KataKeep"], fl["kata_small_denominator"]),
            str(r["en_ref_words"]), str(r["ja_ref_chars"]), str(r["kata_ref_runs"]),
        ])
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    sep = "  "
    line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, sep.join("-" * widths[i] for i in range(len(headers)))]
    for row in rows:
        out.append(sep.join(c.ljust(widths[i]) for i, c in enumerate(row)))
    out.append(f"(n/a = fewer than the minimum reference tokens/runs for a stable rate; "
               f"raw values are in --json. scorer v{SCORER_VERSION})")
    return "\n".join(out)


def parse_spec(spec: str) -> tuple[str, str]:
    """'label:path' -> (label, path). Bare path -> (filename-stem, path).
    Handles Windows-style 'C:\\...' minimally by requiring the label part to be
    non-empty and not look like a drive letter when a colon split is ambiguous."""
    if ":" in spec:
        label, path = spec.split(":", 1)
        if label and path:
            return label, path
    import os
    return os.path.splitext(os.path.basename(spec))[0], spec


def _format_ci(ci: dict) -> str:
    if ci.get("mean") is None:
        return "n/a (no valid resamples)"
    return f"{ci['mean']*100:5.1f}% [{ci['lo']*100:5.1f}%, {ci['hi']*100:5.1f}%]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Code-switching ASR scoring harness")
    ap.add_argument("inputs", nargs="+", help="One or more label:path (or path) JSONL prediction files")
    ap.add_argument("--no-fillers", action="store_true",
                    help="Keep English fillers (default removes um/uh/...)")
    ap.add_argument("--json", metavar="OUT", help="Also write metrics rows as JSON to OUT")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="Bootstrap N resamples over utterances and report CIs")
    ap.add_argument("--ci", type=float, default=95, metavar="P",
                    help="Confidence level for bootstrap intervals (default 95)")
    ap.add_argument("--bootstrap-seed", type=int, default=42, metavar="S",
                    help="Random seed for reproducible bootstrap resampling")
    ap.add_argument("--cluster-key", metavar="FIELD",
                    help="Resample clusters (e.g. 'speaker') instead of utterances")
    ap.add_argument("--paired", action="store_true",
                    help="Paired bootstrap between the first two models, aligned by id")
    ap.add_argument("--allow-id-mismatch", action="store_true",
                    help="With --paired, score the intersection of ids instead of failing")
    ap.add_argument("--min-denominator", type=int, default=DEFAULT_MIN_DENOMINATOR,
                    metavar="K", help="Flag EN/JA rates over fewer than K reference tokens "
                                      f"(default {DEFAULT_MIN_DENOMINATOR})")
    ap.add_argument("--switch-report", metavar="OUT",
                    help="Write switch-point stratified analysis JSON for the first model")
    ap.add_argument("--audit", metavar="OUT",
                    help="Write per-English-word script audit JSON for the first model")
    ap.add_argument("--loanwords", metavar="JSON", default=os.environ.get("JAENCS_LOANWORDS"),
                    help="Katakana<->English loanword lexicon (eval/loanwords.py build); "
                         "enables MER_lenient. Default: $JAENCS_LOANWORDS")
    args = ap.parse_args(argv)
    if args.loanwords:
        load_loanwords(args.loanwords)

    if args.paired and len(args.inputs) != 2:
        raise SystemExit("--paired requires exactly two input files")
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap must be non-negative")

    remove_fillers = not args.no_fillers
    models: list[ModelMetrics] = []
    per_model_counts: list[list[dict]] = []
    all_rows: list[list[dict]] = []
    for spec in args.inputs:
        label, path = parse_spec(spec)
        rows = load_jsonl(path)
        all_rows.append(rows)
        models.append(score_records(rows, label, remove_fillers, args.min_denominator))
        if args.bootstrap:
            per_model_counts.append(per_utterance_counts(rows, remove_fillers, args.cluster_key))

    print(render_table(models))

    bootstrap_results: dict = {}
    if args.bootstrap:
        clustered = bool(args.cluster_key)
        rng = random.Random(args.bootstrap_seed)
        unit = f"clusters of '{args.cluster_key}'" if clustered else "utterances"
        print(f"\nBootstrap CIs ({args.bootstrap} resamples over {unit}, {args.ci}% level, "
              f"seed={args.bootstrap_seed}):")
        for mm, counts in zip(models, per_model_counts):
            if clustered:
                k = len(_cluster_groups(counts))
                if k < 2:
                    print(f"  WARNING: {mm.label}: only {k} cluster(s) for '{args.cluster_key}'; "
                          f"cluster CIs are degenerate")
            cis = bootstrap_ci(counts, args.bootstrap, args.ci, rng, clustered)
            bootstrap_results[mm.label] = cis
            print(f"  {mm.label}")
            for metric, ci in cis.items():
                print(f"    {metric:16s} {_format_ci(ci)}")

        if args.paired:
            rows_a, rows_b = align_records_by_id(all_rows[0], all_rows[1],
                                                 args.allow_id_mismatch)
            counts_a = per_utterance_counts(rows_a, remove_fillers, args.cluster_key)
            counts_b = per_utterance_counts(rows_b, remove_fillers, args.cluster_key)
            paired = paired_bootstrap(counts_a, counts_b, args.bootstrap, args.ci, rng, clustered)
            bootstrap_results["paired_diff"] = {"n_paired": len(rows_a), **paired}
            print(f"\nPaired bootstrap ({models[0].label} - {models[1].label}, "
                  f"{len(rows_a)} utterances aligned by id):")
            for metric, ci in paired.items():
                if ci.get("mean") is None:
                    print(f"    {metric:16s} n/a")
                    continue
                sign = "includes 0" if ci["lo"] <= 0 <= ci["hi"] else "excludes 0"
                print(f"    {metric:16s} Δ {_format_ci(ci)} ({sign}, p_boot={ci['p_boot']})")

    if args.switch_report or args.audit:
        if len(args.inputs) != 1:
            raise SystemExit("--switch-report and --audit require exactly one input file")
        rows = all_rows[0]
        label = models[0].label

        if args.switch_report:
            payload = {"model": label, "scorer_version": SCORER_VERSION,
                       "analysis": aggregate_switch_analysis(rows, remove_fillers)}
            with open(args.switch_report, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nWrote switch-point report -> {args.switch_report}")

        if args.audit:
            audit_records = []
            for rec in rows:
                rn = normalize(rec["reference"], remove_fillers)
                hn = normalize(rec["hypothesis"], remove_fillers)
                for det in script_counts_detailed(mixed_tokens(rn), mixed_tokens(hn)):
                    latin_mismatch = det["category"] == "latin" and not det["exact"]
                    audit_records.append({
                        "id": rec["id"], "ref_word": det["ref"],
                        "hyp_word": det["hyp"], "category": det["category"],
                        "latin_mismatch": latin_mismatch,
                    })
            counts = {"en_ref": 0, "latin": 0, "katakana": 0, "other_jp": 0, "dropped": 0,
                      "latin_mismatch": 0}
            for r in audit_records:
                counts["en_ref"] += 1
                counts[r["category"]] += 1
                if r["latin_mismatch"]:
                    counts["latin_mismatch"] += 1
            payload = {"model": label, "scorer_version": SCORER_VERSION,
                       "counts": counts, "records": audit_records}
            with open(args.audit, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nWrote script audit -> {args.audit}")

    if args.json:
        tok_info = ja_tokenizer_info()
        payload = []
        for m in models:
            row = m.as_row()
            row["ja_tokenizer"] = tok_info
            row["normalization"] = {"remove_english_fillers": remove_fillers,
                                    "loanword_lexicon": _LOANWORDS_META}
            payload.append(row)
        if bootstrap_results:
            payload = {"models": payload, "bootstrap": bootstrap_results}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
