#!/usr/bin/env python3
"""Sample transcript pairs for the human validation of the script-aware metrics
(pre-registered metric-validation protocol).

For each utterance of the chosen test sets, every pair of systems with
different hypotheses is a candidate. Seeded configurations contribute seed 1
only. The pool is the ladder's pre-registered systems (protocol: "Phase 0 ladder
outputs, zero-shot and fine-tuned"); its exploratory runs (epoch checkpoints,
ja+rp decoding of the fine-tuned adapters) are left out. Candidates are
stratified as follows:
  1 metric_flip  strict MER and MER_lenient prefer different transcripts
  2 script_flip  ScriptAcc and KataKeep move in opposite directions
  3 close        |ΔMER| < 0.05
  4 random       everything else
Up to --per-stratum pairs are drawn from each stratum, at most 3 per
utterance. A/B order is randomised. Two files are written:
  packet.jsonl  what raters see: pair id, audio, transcript A, transcript B
                (no systems, no reference, no metrics)
  key.jsonl     systems, stratum and per-transcript metrics, for the analysis

Usage:
  $PY human_eval/make_pairs.py --runs /data/code-switching/runs/phase0 \
      --loanwords /data/code-switching/data/lexicon/loanwords_v1.json \
      --manifest csf_read=/data/code-switching/data/eval-real/csfleurs/read_test/manifest.jsonl \
      --out human_eval/pairs_v1
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import score as S  # noqa: E402

STRATA = ("metric_flip", "script_flip", "close", "random")
PHASE0_EXPLORATORY = re.compile(r"_ep\d|^ev_ft_.*_ja_rp$")


def utt_metrics(ref: str, hyp: str) -> dict:
    p = S.score_pair(ref, hyp)
    rate = lambda ec: ec.errors / ec.ref_len if ec.ref_len else 0.0
    sc, kc = p["script"], p["kata"]
    return {"MER": rate(p["mer"]), "MER_lenient": rate(p["mer_len"]),
            "ScriptAcc": sc.latin / sc.en_ref if sc.en_ref else None,
            "ScriptAcc_strict": sc.latin_exact / sc.en_ref if sc.en_ref else None,
            "KataKeep": kc.kept / kc.runs if kc.runs else None}


def resolve_audio(manifest: str, audio: str) -> str:
    """Manifest 'audio' fields are absolute, or relative to the eval set's audio
    root (the manifest's directory or its parent)."""
    if os.path.isabs(audio):
        return audio
    for root in (Path(manifest).parent, Path(manifest).parent.parent):
        if (root / audio).exists():
            return str(root / audio)
    raise FileNotFoundError(f"{audio} not found next to {manifest} or its parent")


def stratum(ma: dict, mb: dict) -> str:
    d, dl = ma["MER"] - mb["MER"], ma["MER_lenient"] - mb["MER_lenient"]
    if d * dl < 0 or (d == 0) != (dl == 0):
        return "metric_flip"
    sa = (ma["ScriptAcc"], mb["ScriptAcc"])
    kk = (ma["KataKeep"], mb["KataKeep"])
    if None not in sa + kk and (sa[0] - sa[1]) * (kk[0] - kk[1]) < 0:
        return "script_flip"
    if abs(d) < 0.05:
        return "close"
    return "random"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--manifest", action="append", required=True, metavar="TEST=PATH",
                    help="Test set name as used in the job names, and its manifest (audio paths)")
    ap.add_argument("--loanwords", required=True)
    ap.add_argument("--per-stratum", type=int, default=125)
    ap.add_argument("--max-per-utt", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S.load_loanwords(args.loanwords)
    rng = random.Random(args.seed)
    runs = Path(args.runs)

    cands = {s: [] for s in STRATA}
    for spec in args.manifest:
        test, path = spec.split("=", 1)
        man = {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8")) if r}
        hyps: dict[str, dict[str, str]] = {}
        for p in sorted(runs.glob(f"*__{test}/preds.jsonl")):
            label = p.parent.name.rsplit("__", 1)[0]
            m = re.search(r"_s(\d+)", label)
            if not (p.parent / "DONE").exists() or (m and m.group(1) != "1") or PHASE0_EXPLORATORY.search(label):
                continue
            for r in map(json.loads, open(p, encoding="utf-8")):
                hyps.setdefault(r["id"], {})[re.sub(r"_s\d+", "", label)] = r["hypothesis"]
        for uid, by_sys in hyps.items():
            ref = man[uid]["reference"]
            mets = {s: utt_metrics(ref, h) for s, h in by_sys.items()}
            for a, b in itertools.combinations(sorted(by_sys), 2):
                if S.normalize(by_sys[a]) == S.normalize(by_sys[b]):
                    continue
                cands[stratum(mets[a], mets[b])].append(
                    {"test": test, "utt_id": uid, "audio": resolve_audio(path, man[uid]["audio"]),
                     "sys": (a, b), "hyp": (by_sys[a], by_sys[b]), "met": (mets[a], mets[b])})
    print({s: len(v) for s, v in cands.items()}, "candidate pairs")

    per_utt: dict[str, int] = {}
    packet, key = [], []
    for s in STRATA:
        pool = cands[s][:]
        rng.shuffle(pool)
        taken = 0
        for c in pool:
            if taken >= args.per_stratum:
                break
            u = f"{c['test']}/{c['utt_id']}"
            if per_utt.get(u, 0) >= args.max_per_utt:
                continue
            per_utt[u] = per_utt.get(u, 0) + 1
            flip = rng.random() < 0.5
            ia, ib = (1, 0) if flip else (0, 1)
            pid = f"p{len(packet) + 1:04d}"
            packet.append({"pair_id": pid, "audio": c["audio"],
                           "transcript_A": c["hyp"][ia], "transcript_B": c["hyp"][ib]})
            key.append({"pair_id": pid, "stratum": s, "test": c["test"], "utt_id": c["utt_id"],
                        "system_A": c["sys"][ia], "system_B": c["sys"][ib],
                        "metrics_A": c["met"][ia], "metrics_B": c["met"][ib]})
            taken += 1
    # shuffle presentation order so strata are interleaved
    order = list(range(len(packet)))
    rng.shuffle(order)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "packet.jsonl", "w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(packet[i], ensure_ascii=False) + "\n")
    with open(out / "key.jsonl", "w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(key[i], ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(packet)} pairs to {out}: {dict(Counter(k['stratum'] for k in key))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
