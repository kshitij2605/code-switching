#!/usr/bin/env python3
"""Phase 1 leaderboard statistics: every system against zero-shot Whisper-large-v3.

For each real code-switching test set:
  * per-system 95% bootstrap CIs of MER, MER_capped and MER_lenient;
  * a paired bootstrap (aligned by utterance id) of system - Whisper-large-v3
    (auto language), with Holm correction across systems per test set and metric.
Seeds of the Phase 0 fine-tuned systems are pooled per utterance, as in
ladder_stats.py; the ladder's exploratory runs are left out, as in the leaderboard.
Every system and comparison has its own seeded random stream.

Usage:
  $PY eval/leaderboard_stats.py [--n 5000]
Writes RUNS/scores/leaderboard_stats.md and .json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402
from ladder_stats import fmt_p, holm, load_configs  # noqa: E402

REFERENCE = "zs_whisper_auto"
PHASE0_EXPLORATORY = re.compile(r"_ep\d|^ev_ft_.*_ja_rp$")
TESTED = ("MER", "MER_capped", "MER_lenient")   # paired tests vs Whisper-v3, Holm per metric
METRICS = TESTED + ("CER_yv",)                    # CER_yv (Ye & Vickers): per-system CIs only


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/data/code-switching/runs/phase1_leaderboard")
    ap.add_argument("--phase0-runs", default="/data/code-switching/runs/phase0")
    ap.add_argument("--tests", default="csf_read,csyodas,switchlingua")
    ap.add_argument("--n", type=int, default=5000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loanwords", default="/data/code-switching/data/lexicon/loanwords_v1.json")
    args = ap.parse_args()
    S.load_loanwords(args.loanwords)
    out: dict = {"reference": REFERENCE, "n_boot": args.n, "tests": {}}
    md = ["# Phase 1 leaderboard statistics\n",
          f"Every system against zero-shot Whisper-large-v3 (`{REFERENCE}`): paired bootstrap by "
          f"utterance id ({args.n} resamples; Phase 0 seeds pooled), Holm-adjusted across systems "
          f"per test set and metric. A bootstrap p of 0 is shown as its bound (<m·2/n).\n"]

    for test in [t for t in args.tests.split(",") if t]:
        cfgs: dict[str, list[dict]] = {}
        for runs in (Path(args.runs), Path(args.phase0_runs)):
            cfgs.update({k: v for k, v in load_configs(runs, test).items()
                         if not PHASE0_EXPLORATORY.search(k)})
        if REFERENCE not in cfgs:
            continue
        common = set.intersection(*({c["id"] for c in v} for v in cfgs.values()))
        cfgs = {k: [c for c in v if c["id"] in common] for k, v in cfgs.items()}
        rows = {}
        for name, counts in cfgs.items():
            rng = random.Random(f"{args.seed}:lb:{test}:{name}")
            ci = S.bootstrap_ci(counts, args.n, 95, rng)
            agg = S._aggregate_rates(counts)
            row = {"n": len(counts), "point": {m: agg[m] for m in METRICS}, "ci": {m: ci[m] for m in METRICS}}
            if name != REFERENCE:
                row["vs_ref"] = S.paired_bootstrap(counts, cfgs[REFERENCE], args.n, 95,
                                                   random.Random(f"{args.seed}:lbpair:{test}:{name}"))
            rows[name] = row
        others = [k for k in rows if k != REFERENCE]
        for m in TESTED:
            ps = [rows[k]["vs_ref"][m].get("p_boot") for k in others]
            ok = [(k, p) for k, p in zip(others, ps) if p is not None]
            for (k, _), ph in zip(ok, holm([p for _, p in ok])):
                rows[k]["vs_ref"][m]["p_holm"] = round(ph, 4)
                rows[k]["vs_ref"][m]["m"] = len(ok)
        out["tests"][test] = rows

        md.append(f"\n## {test} (n = {len(common)})\n\n| system | MER [95% CI] | ΔMER vs Whisper-v3, p_holm "
                  f"| MER_capped [95% CI] | ΔMER_capped, p_holm | MER_lenient | CER_yv [95% CI] |"
                  f"\n|---|---|---|---|---|---|---|")
        for name in sorted(rows, key=lambda k: rows[k]["point"]["MER"]):
            r = rows[name]
            cells = []
            for m in ("MER", "MER_capped"):
                cells.append(f"{100 * r['point'][m]:.1f} [{100 * r['ci'][m]['lo']:.1f}, {100 * r['ci'][m]['hi']:.1f}]")
                if name == REFERENCE:
                    cells.append("reference")
                else:
                    d = r["vs_ref"][m]
                    cells.append(f"{100 * (r['point'][m] - rows[REFERENCE]['point'][m]):+.1f} "
                                 f"[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] "
                                 f"{fmt_p(d.get('p_holm'), d.get('m', 1), d['n_valid'])}")
            cells.append(f"{100 * r['point']['MER_lenient']:.1f}")
            cells.append(f"{100 * r['point']['CER_yv']:.1f} [{100 * r['ci']['CER_yv']['lo']:.1f}, "
                         f"{100 * r['ci']['CER_yv']['hi']:.1f}]")
            md.append(f"| {name} | " + " | ".join(cells) + " |")

    dest = Path(args.runs) / "scores"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "leaderboard_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (dest / "leaderboard_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
