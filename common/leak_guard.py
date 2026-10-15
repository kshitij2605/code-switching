#!/usr/bin/env python3
"""Test-leak guard for training and validation data.

Every manifest that feeds training or checkpoint selection must pass through
``LeakGuard`` first. A row is rejected if it matches protected evaluation data by

  * id            (only for sources with globally unique ids, e.g. CS-FLEURS),
  * audio name    (basename of the audio file; same restriction),
  * exact text    (script-agnostic normalised text, see ``norm_text``),
  * n-gram overlap (fraction of the row's character n-grams that occur in a
                    protected *strict* sentence >= threshold).

Protected data has two roles:

  test  frozen test sets (CS-FLEURS read/test, FLEURS test, local eval sets):
        never in training, never used for checkpoint selection;
  dev   selection sets (CS-FLEURS xtts/test1, FLEURS validation): never in
        training, but allowed as the validation set.

Training rows are checked against both roles (the default); dev/validation rows
against ``roles=("test",)`` only.

The blocklist (``common/eval_blocklist.json``) holds only ids, file basenames and
truncated SHA-1 hashes — no evaluation text — and is rebuilt with
``common/build_eval_blocklist.py``. It keeps the frozen benchmarks (e.g. CS-FLEURS
read/test) out of every training mix.

CLI:
    python common/leak_guard.py data/training/balanced_train.jsonl [more.jsonl ...]
    python common/leak_guard.py --roles test dev_manifest.jsonl
exits 1 if any row overlaps protected evaluation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_BLOCKLIST = Path(__file__).with_name("eval_blocklist.json")
NGRAM_N = 12
NGRAM_THRESHOLD = 0.5
ROLES = ("test", "dev")

TEXT_KEYS = ("target", "transcript", "reference", "text", "sentence")
ID_KEYS = ("id", "source_id", "utt_id", "clean_id")
AUDIO_KEYS = ("audio_file", "audio_path", "audio", "file_name", "path")


def norm_text(text: str | None) -> str:
    """Script-agnostic matching key: NFKC, lowercase, keep only letters/digits.

    Spaces, punctuation and markup (e.g. the ``**`` markers in CS-FLEURS xtts
    text or MeCab spacing) are dropped so that trivially reformatted copies of a
    test sentence still match.
    """
    t = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(ch for ch in t if ch.isalnum())


def _h(s: str, n_hex: int = 16) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n_hex]


def text_hash(text: str | None) -> str | None:
    k = norm_text(text)
    return _h(k) if k else None


def ngram_hashes(text: str | None, n: int = NGRAM_N) -> set[str]:
    k = norm_text(text)
    if not k:
        return set()
    if len(k) < n:
        return {_h(k, 12)}
    return {_h(k[i:i + n], 12) for i in range(len(k) - n + 1)}


def audio_basename(path: str) -> str:
    return os.path.basename(str(path)).lower()


class LeakError(RuntimeError):
    """Raised when training data overlaps protected evaluation data."""


def _describe(row: dict) -> str:
    for k in ID_KEYS + AUDIO_KEYS:
        if row.get(k):
            return f"{k}={row[k]}"
    return "<row without id/audio>"


class _RoleSets:
    def __init__(self, d: dict) -> None:
        self.ids = set(d.get("ids", []))
        self.audio = set(d.get("audio_basenames", []))
        self.text = set(d.get("text_hashes", []))
        self.ngrams = set(d.get("ngram_hashes", []))


class LeakGuard:
    def __init__(self, blocklist_path: str | os.PathLike = DEFAULT_BLOCKLIST,
                 ngram_threshold: float = NGRAM_THRESHOLD,
                 roles: Iterable[str] = ROLES) -> None:
        path = Path(blocklist_path)
        if not path.exists():
            raise FileNotFoundError(
                f"leak-guard blocklist not found: {path}. Build it with "
                f"common/build_eval_blocklist.py (it must exist before any training run).")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.path = str(path)
        self.meta = {k: data.get(k) for k in ("version", "created", "sources", "ngram_n")}
        self.ngram_n = int(data.get("ngram_n", NGRAM_N))
        self.ngram_threshold = ngram_threshold
        stored = data.get("roles") or {"test": data}      # v1 files had no roles
        self.roles = tuple(roles)
        unknown = [r for r in self.roles if r not in ROLES]
        if unknown:
            raise ValueError(f"unknown roles {unknown}; use {ROLES}")
        self._sets = {r: _RoleSets(stored.get(r, {})) for r in self.roles}

    def check_row(self, row: dict) -> list[str]:
        """Reasons this row overlaps protected evaluation data ([] if clean)."""
        reasons = []
        texts = [(k, row[k]) for k in TEXT_KEYS if isinstance(row.get(k), str) and row[k].strip()]
        for role, s in self._sets.items():
            for k in ID_KEYS:
                v = row.get(k)
                if v is not None and str(v) in s.ids:
                    reasons.append(f"id:{k}@{role}")
            for k in AUDIO_KEYS:
                v = row.get(k)
                if isinstance(v, str) and audio_basename(v) in s.audio:
                    reasons.append(f"audio:{k}@{role}")
            for k, v in texts:
                th = text_hash(v)
                if th and th in s.text:
                    reasons.append(f"text:{k}:exact@{role}")
                    continue
                if s.ngrams:
                    grams = ngram_hashes(v, self.ngram_n)
                    if grams:
                        frac = len(grams & s.ngrams) / len(grams)
                        if frac >= self.ngram_threshold:
                            reasons.append(f"text:{k}:ngram={frac:.2f}@{role}")
        return reasons

    def filter_rows(self, rows: Iterable[dict], on_leak: str = "raise",
                    context: str = "") -> list[dict]:
        """Check rows. on_leak: 'raise' (default) | 'drop' (discard, report) |
        'warn' (report, keep all)."""
        if on_leak not in ("raise", "drop", "warn"):
            raise ValueError(f"on_leak must be raise|drop|warn, got {on_leak!r}")
        rows = list(rows)
        clean, leaks = [], []
        for r in rows:
            why = self.check_row(r)
            if why:
                leaks.append((r, why))
            else:
                clean.append(r)
        tag = f"[leak-guard{': ' + context if context else ''}]"
        if not leaks:
            print(f"{tag} {len(rows)} rows checked against {'+'.join(self.roles)}, "
                  f"no overlap with protected eval data")
            return rows
        examples = "; ".join(f"{_describe(r)} <- {','.join(w)}" for r, w in leaks[:5])
        msg = (f"{tag} {len(leaks)} of {len(rows)} rows overlap protected evaluation "
               f"data (e.g. {examples})")
        if on_leak == "raise":
            raise LeakError(msg + ". Fix the data source; pass on_leak='drop' only to "
                                  "discard them deliberately.")
        print(msg + (" -> dropped" if on_leak == "drop" else " -> KEPT (warn mode)"),
              file=sys.stderr)
        return clean if on_leak == "drop" else rows


def group_split(rows: list[dict], eval_frac: float, rng, key_fn) -> tuple[list[dict], list[dict]]:
    """Split rows so that all rows sharing a group key land on the same side.

    Use it instead of a row-level shuffle whenever several rows share content
    (the same transcript rendered by several TTS engines, augmented copies of a
    clip, several task prompts over one clip): a row-level split leaks them into
    validation.
    """
    groups: dict = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    keys = sorted(groups, key=lambda k: str(k))
    rng.shuffle(keys)
    n_eval = max(1, round(len(keys) * eval_frac)) if eval_frac > 0 else 0
    eval_keys = set(keys[:n_eval])
    train = [r for k in keys if k not in eval_keys for r in groups[k]]
    evl = [r for k in keys if k in eval_keys for r in groups[k]]
    return train, evl


def transcript_group_key(row: dict) -> str:
    """Group key: normalised transcript (falls back to audio path)."""
    for k in TEXT_KEYS:
        if isinstance(row.get(k), str) and row[k].strip():
            return text_hash(row[k]) or ""
    for k in AUDIO_KEYS:
        if row.get(k):
            return str(row[k])
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def row_group_key(row: dict) -> str:
    """Default group key: the row's explicit ``group`` field when the data script
    set one (e.g. one CS-FLEURS sentence rendered by several voices, or all
    augmented copies of a clip), otherwise the normalised transcript."""
    g = row.get("group")
    return f"group:{g}" if g not in (None, "") else transcript_group_key(row)


def _audio_key(row: dict) -> str | None:
    for k in AUDIO_KEYS:
        if isinstance(row.get(k), str) and row[k]:
            return os.path.normpath(row[k]).replace("\\", "/")
    return None


def check_disjoint(train_rows: list[dict], eval_rows: list[dict],
                   key_fn: Callable[[dict], str] = row_group_key, context: str = "") -> None:
    """Raise LeakError if train and validation share a group or an audio file.

    The audio check catches multi-task data where one clip carries several
    targets (native / all-Japanese / all-English) and therefore several
    transcript groups."""
    tag = f"[split-check{': ' + context if context else ''}]"
    g_train = {key_fn(r) for r in train_rows}
    shared_groups = {key_fn(r) for r in eval_rows} & g_train
    a_train = {a for a in map(_audio_key, train_rows) if a}
    shared_audio = {a for a in map(_audio_key, eval_rows) if a} & a_train
    if shared_groups or shared_audio:
        raise LeakError(f"{tag} train/validation overlap: {len(shared_groups)} shared groups "
                        f"(e.g. {sorted(shared_groups)[:3]}), {len(shared_audio)} shared audio "
                        f"files (e.g. {sorted(shared_audio)[:3]})")
    print(f"{tag} train ({len(train_rows)}) and validation ({len(eval_rows)}) are disjoint "
          f"by group and audio file")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check JSONL manifests against the eval blocklist")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--blocklist", default=str(DEFAULT_BLOCKLIST))
    ap.add_argument("--threshold", type=float, default=NGRAM_THRESHOLD)
    ap.add_argument("--roles", default=",".join(ROLES),
                    help="Comma-separated roles to check against: test,dev (training data, "
                         "default) or test (a validation/dev manifest)")
    args = ap.parse_args(argv)
    guard = LeakGuard(args.blocklist, args.threshold, roles=args.roles.split(","))
    bad = 0
    for f in args.files:
        rows = [json.loads(l) for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()]
        leaks = [(r, guard.check_row(r)) for r in rows]
        leaks = [(r, w) for r, w in leaks if w]
        bad += len(leaks)
        print(f"{f}: {len(rows)} rows, {len(leaks)} overlap protected eval data ({args.roles})")
        for r, w in leaks[:10]:
            print(f"   {_describe(r)} <- {','.join(w)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
