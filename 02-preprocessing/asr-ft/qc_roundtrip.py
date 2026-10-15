#!/usr/bin/env python3
"""Round-trip ASR quality control for synthetic code-switched speech (Phase 3).

A synthetic clip is kept only if independent ASR "judges" can read back what it
was supposed to say (Phase 3 QC). Inputs:
  * the synthetic manifest (training contract: id, audio_path, transcript,
    segments [{lang, text}], duration_s);
  * one or more judge prediction files ({id, reference, hypothesis}), e.g.
    Whisper-v3 LoRA, kotoba-whisper and Qwen3-ASR. Never use only the model
    being trained.

Per clip:
  qc_mer      best (lowest) round-trip MER over the judges, after the lenient
              loanword rewrite (a katakana rendering of an English word is not
              a TTS failure);
  qc_en_recall fraction of the English spans whose words all appear, in either
              script, in the best judge's hypothesis;
  qc_cps      reference characters per second, which catches truncated or
              stalled audio.
Buckets:
  pass    qc_mer <= --max-mer, all English spans recovered, cps within bounds
  hard    fails only by a margin (qc_mer <= --hard-mer, recall >= 0.5). Kept up
          to --hard-frac of the output: realistic difficult audio, not failures
  reject  everything else

Usage:
  python qc_roundtrip.py --manifest synth/manifest.jsonl \
      --judge whisper_lora.jsonl --judge kotoba.jsonl --loanwords loanwords_v1.json \
      --out synth/manifest.qc.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04-evaluation" / "eval-ft" / "eval"))
import score as S  # noqa: E402


def clip_qc(row: dict, hyps: list[str]) -> dict:
    rn = S.normalize(row["transcript"])
    best_mer, best_h = float("inf"), ""
    for h in hyps:
        hn = S.lenient_hypothesis(rn, S.normalize(h))
        r_tok, h_tok = S.mixed_tokens(rn), S.mixed_tokens(hn)
        ec = S._counts_from_ops(S.align(r_tok, h_tok), len(r_tok))
        mer = ec.errors / max(ec.ref_len, 1)
        if mer < best_mer:
            best_mer, best_h = mer, hn
    en_spans = [s["text"] for s in row.get("segments", []) if s.get("lang") == "en"]
    hyp_words = set(S.english_words(best_h))
    recovered = sum(1 for sp in en_spans
                    if set(S.english_words(S.normalize(sp))) <= hyp_words)
    dur = float(row.get("duration_s") or 0)
    chars = len(rn.replace(" ", ""))
    return {"qc_mer": round(best_mer, 4),
            "qc_en_recall": round(recovered / len(en_spans), 4) if en_spans else 1.0,
            "qc_cps": round(chars / dur, 2) if dur else None}


def bucket(q: dict, max_mer: float, hard_mer: float, cps: tuple[float, float]) -> str:
    cps_ok = q["qc_cps"] is None or cps[0] <= q["qc_cps"] <= cps[1]
    if q["qc_mer"] <= max_mer and q["qc_en_recall"] == 1.0 and cps_ok:
        return "pass"
    if q["qc_mer"] <= hard_mer and q["qc_en_recall"] >= 0.5 and cps_ok:
        return "hard"
    return "reject"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--judge", action="append", required=True, help="judge predictions JSONL")
    ap.add_argument("--loanwords", required=True)
    ap.add_argument("--max-mer", type=float, default=0.15)
    ap.add_argument("--hard-mer", type=float, default=0.35)
    ap.add_argument("--hard-frac", type=float, default=0.15)
    ap.add_argument("--cps", type=float, nargs=2, default=(3.0, 25.0), metavar=("MIN", "MAX"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S.load_loanwords(args.loanwords)

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    judged: dict[str, list[str]] = {}
    for jf in args.judge:
        for r in map(json.loads, open(jf, encoding="utf-8")):
            judged.setdefault(r["id"], []).append(r["hypothesis"])
    missing = [r["id"] for r in rows if r["id"] not in judged]
    if missing:
        raise SystemExit(f"{len(missing)} clips have no judge prediction, e.g. {missing[:3]}")

    out_rows, counts = [], Counter()
    for r in rows:
        q = clip_qc(r, judged[r["id"]])
        q["qc_bucket"] = bucket(q, args.max_mer, args.hard_mer, tuple(args.cps))
        q["qc_judges"] = len(judged[r["id"]])
        counts[q["qc_bucket"]] += 1
        out_rows.append(dict(r, **q))
    # cap the hard bucket at hard_frac of what is kept (seeded)
    rng = random.Random(args.seed)
    passed = [r for r in out_rows if r["qc_bucket"] == "pass"]
    hard = [r for r in out_rows if r["qc_bucket"] == "hard"]
    rng.shuffle(hard)
    cap = int(args.hard_frac / (1 - args.hard_frac) * len(passed)) if args.hard_frac < 1 else len(hard)
    kept = passed + hard[:cap]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"QC {len(rows)} clips: {dict(counts)}; kept {len(kept)} "
          f"(pass {len(passed)} + hard {min(cap, len(hard))}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
