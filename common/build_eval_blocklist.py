#!/usr/bin/env python3
"""Build common/eval_blocklist.json — the protected evaluation data for LeakGuard.

Only ids, audio basenames and truncated SHA-1 hashes are stored (no text), so
the file can be committed even though several test sets are CC BY-NC.

Roles (see common/leak_guard.py):
  test  frozen test sets: never trained on, never used for checkpoint selection
  dev   selection sets:   never trained on, allowed as validation data

Match levels per source:
  strict      exact normalised-text hash + character n-gram hashes
  exact       exact normalised-text hash only (for legacy synthetic eval sets whose
              template segments are shared with the training corpus by design)
  +ids        additionally block the source's ids and audio basenames — only for
              sources with globally unique ids (CS-FLEURS "jpn_1662_SS"); locally
              numbered ids such as "ja_jp_1" collide with training manifests.

Default sources (HF cache, downloaded if missing unless --offline):
  test  CS-FLEURS jpn-eng read/test (strict+ids); FLEURS ja_jp + en_us test (strict);
        CS-YODAS jpn (strict+ids). The 214 YouTube videos are also blocked as ids
        "yt:<video>". Any training row derived from YODAS must carry
        source_id="yt:<video>", so that whole test videos can never reach training,
        even with different captions.
  dev   CS-FLEURS jpn-eng xtts/test1, xtts/test2 (strict+ids; test2 has no jpn-eng
        pairs today); FLEURS ja_jp + en_us validation ("dev.tsv", strict)
CS-FLEURS mms/test has no Japanese pairs. Add local eval sets with
--manifest PATH:LEVEL (role test) or --dev-manifest PATH:LEVEL (role dev).

Only list evaluation sets, not the candidate pools they were drawn from:
eval-synth/gen777.jsonl is the seed-777 generator pool from which
eval_transcripts.jsonl (200 rows) was selected after removing overlap with the
training templates, so 19 of its rows legitimately occur in training.

Usage (mactrn01) — this exact command built the committed blocklist:
  D=/data/code-switching/data
  python common/build_eval_blocklist.py \
     --hf-cache /data/code-switching/hf_cache --hf-cache ~/.cache/huggingface \
     --manifest $D/eval-real/csfleurs/read_test/manifest.jsonl:strict+ids \
     --manifest $D/eval-mono/ja_jp_test/manifest.jsonl:strict \
     --manifest $D/eval-mono/en_us_test/manifest.jsonl:strict \
     --manifest $D/eval-synth/csartificial_manifest.jsonl:exact \
     --manifest $D/eval-synth/eval_transcripts.jsonl:exact \
     --manifest $D/eval-synth/edge/manifest.jsonl:exact \
     --manifest $D/eval-diss-synth/refs.jsonl:exact
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.leak_guard import (DEFAULT_BLOCKLIST, NGRAM_N, ROLES, audio_basename,  # noqa: E402
                               ngram_hashes, text_hash)

CSF_REPO = "byan/cs-fleurs"
CSY_REPO = "byan/cs-yodas"
CSF_SPLITS = {"read/test": "test", "xtts/test1": "dev", "xtts/test2": "dev"}
FLEURS_REPO = "google/fleurs"
FLEURS_CONFIGS = ("ja_jp", "en_us")
FLEURS_SPLITS = {"test": "test", "dev": "dev"}      # TSV name -> role
TEXT_KEYS = ("reference", "text", "transcript", "target", "transcription", "raw_transcription")
LEVELS = ("strict", "exact", "strict+ids", "exact+ids")


class Blocklist:
    def __init__(self, ngram_n: int):
        self.ngram_n = ngram_n
        self.sets = {r: {"ids": set(), "audio": set(), "text": set(), "ngrams": set()} for r in ROLES}
        self.sources: dict = {}

    def add(self, name: str, rows: list[dict], level: str, role: str = "test") -> None:
        if level not in LEVELS or role not in ROLES:
            raise ValueError(f"bad level/role {level!r}/{role!r}")
        s = self.sets[role]
        strict = level.startswith("strict")
        with_ids = level.endswith("+ids")
        n_text = 0
        for r in rows:
            texts = [r[k] for k in TEXT_KEYS if isinstance(r.get(k), str) and r[k].strip()]
            for t in texts:
                h = text_hash(t)
                if h:
                    s["text"].add(h)
                    n_text += 1
                if strict:
                    s["ngrams"] |= ngram_hashes(t, self.ngram_n)
            if with_ids:
                if r.get("id") is not None:
                    s["ids"].add(str(r["id"]))
                for k in ("file_name", "audio", "audio_path", "audio_file"):
                    if isinstance(r.get(k), str):
                        s["audio"].add(audio_basename(r[k]))
        self.sources[name] = {"rows": len(rows), "texts": n_text, "level": level, "role": role}
        note = "  <- no rows for this language pair" if not rows else ""
        print(f"  [{role}] {name}: {len(rows)} rows ({level}){note}")

    def dump(self, path: Path) -> None:
        payload = {
            "version": 2,
            "created": time.strftime("%Y-%m-%d"),
            "ngram_n": self.ngram_n,
            "sources": self.sources,
            "roles": {r: {"ids": sorted(s["ids"]), "audio_basenames": sorted(s["audio"]),
                          "text_hashes": sorted(s["text"]), "ngram_hashes": sorted(s["ngrams"])}
                      for r, s in self.sets.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
                        encoding="utf-8")
        print(f"wrote {path}")
        for r, s in self.sets.items():
            print(f"  {r}: {len(s['ids'])} ids, {len(s['audio'])} audio, "
                  f"{len(s['text'])} text hashes, {len(s['ngrams'])} n-gram hashes")


def _cached_file(hf_caches: list[str], repo: str, relpath: str) -> str | None:
    """First snapshot copy of repo/relpath in any cache (one file even if several
    caches or snapshots hold it)."""
    name = "datasets--" + repo.replace("/", "--")
    for cache in hf_caches:
        for root in (os.path.join(cache, "hub"), cache):
            hits = sorted(glob.glob(os.path.join(root, name, "snapshots", "*", relpath)))
            if hits:
                return hits[0]
    return None


def _fetch(hf_caches: list[str], repo: str, relpath: str, offline: bool) -> str | None:
    path = _cached_file(hf_caches, repo, relpath)
    if path or offline:
        if not path:
            print(f"  WARNING: {repo}/{relpath} not in cache (offline)")
        return path
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, relpath, repo_type="dataset")
    except Exception as e:  # pragma: no cover - network dependent
        print(f"  WARNING: could not fetch {repo}/{relpath}: {type(e).__name__}: {e}")
        return None


def load_csfleurs(hf_caches: list[str], split: str, offline: bool) -> list[dict] | None:
    path = _fetch(hf_caches, CSF_REPO, f"{split}/metadata.jsonl", offline)
    if not path:
        return None
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("language") == "jpn-eng"]


def load_fleurs(hf_caches: list[str], config: str, split: str, offline: bool) -> list[dict] | None:
    """FLEURS text from data/<config>/<split>.tsv (id, file_name, raw_transcription,
    transcription, ...). The ids are FLEURS sentence ids, so they are not blocked."""
    path = _fetch(hf_caches, FLEURS_REPO, f"data/{config}/{split}.tsv", offline)
    if not path:
        return None
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) >= 4:
            rows.append({"raw_transcription": cols[2], "transcription": cols[3]})
    return rows


def load_csyodas_jpn(hf_caches: list[str], offline: bool) -> list[dict] | None:
    """CS-YODAS jpn utterances (id + text), plus one id-only row per YouTube video."""
    path = _fetch(hf_caches, CSY_REPO, "jpn.jsonl", offline)
    if not path:
        return None
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [{"id": r["id"], "text": r["text"]} for r in recs]
    videos = {r["id"].split("_jpn_asr_")[0].rsplit("_", 2)[0] for r in recs}
    return rows + [{"id": f"yt:{v}"} for v in sorted(videos)]


def _parse_spec(spec: str) -> tuple[str, str]:
    path, _, level = spec.rpartition(":")
    if not path or level not in LEVELS:
        raise SystemExit(f"bad manifest spec {spec!r}; use PATH:strict|exact[+ids]")
    return path, level


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-cache", action="append", default=[],
                    help="HF cache root(s) to search (default: $HF_HOME and ~/.cache/huggingface)")
    ap.add_argument("--manifest", action="append", default=[], metavar="PATH:LEVEL",
                    help="Extra TEST-role eval manifest; LEVEL is strict | exact, optionally +ids")
    ap.add_argument("--dev-manifest", action="append", default=[], metavar="PATH:LEVEL",
                    help="Extra DEV-role (checkpoint-selection) manifest")
    ap.add_argument("--offline", action="store_true", help="Do not download missing metadata")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Build even if a default source is unavailable (not recommended)")
    ap.add_argument("--ngram-n", type=int, default=NGRAM_N)
    ap.add_argument("--out", default=str(DEFAULT_BLOCKLIST))
    args = ap.parse_args(argv)

    caches = args.hf_cache or [c for c in (os.environ.get("HF_HOME"),
                                           os.path.expanduser("~/.cache/huggingface")) if c]
    bl = Blocklist(args.ngram_n)
    missing = []
    print("default sources:")
    for split, role in CSF_SPLITS.items():
        rows = load_csfleurs(caches, split, args.offline)
        if rows is None:
            missing.append(f"cs-fleurs/{split}")
        else:
            bl.add(f"cs-fleurs/{split}/jpn-eng", rows, "strict+ids", role)
    for cfg in FLEURS_CONFIGS:
        for split, role in FLEURS_SPLITS.items():
            rows = load_fleurs(caches, cfg, split, args.offline)
            if rows is None:
                missing.append(f"fleurs/{cfg}/{split}")
            else:
                bl.add(f"fleurs/{cfg}/{split}", rows, "strict", role)
    rows = load_csyodas_jpn(caches, args.offline)
    if rows is None:
        missing.append("cs-yodas/jpn")
    else:
        bl.add("cs-yodas/jpn", rows, "strict+ids", "test")
    if args.manifest or args.dev_manifest:
        print("extra manifests:")
    for role, specs in (("test", args.manifest), ("dev", args.dev_manifest)):
        for spec in specs:
            path, level = _parse_spec(spec)
            rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
            bl.add(path, rows, level, role)
    if missing and not args.allow_missing:
        raise SystemExit(f"missing default sources: {missing} (use --allow-missing to override)")
    bl.dump(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
