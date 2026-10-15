#!/usr/bin/env python3
"""Pre-registered statistics for the Phase 0 ladder
(pre-registered protocol).

For every configuration, the per-utterance error counts of its seeds are pooled
(numerators and denominators summed per utterance). A configuration is thus
compared on the same utterances with its seed noise averaged. Then:

  * paired bootstrap (aligned by id) for the pre-registered comparisons, with
    Holm correction across the comparisons of one test set and metric;
  * Kendall tau-b between the system rankings on the synthetic set and on
    CS-FLEURS read/test (MER and ScriptAcc), with a bootstrap CI that resamples
    utterances in both sets.

Exploratory (not pre-registered; reported in separate sections):
  * the ja+rp decoding (forced ja + repetition penalty) applied to the fine-tuned
    Whisper adapters (own Holm family);
  * ranking fidelity for further set pairs (--rank-pairs), and the
    model-selection regret of picking the best system on the proxy set;
  * the runaway-robust MER_capped with the Runaway share (--capped-sets);
  * with --loanwords, the script-choice gap MER - MER_lenient.
Exploratory sections use their own seeded random streams, per comparison or
per system, so adding systems never changes any other number.

Usage:
  $PY eval/ladder_stats.py --runs /data/code-switching/runs/phase0 [--n 5000]
Writes RUNS/stats.json and RUNS/stats.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402

# (label, system A, system B): A - B is reported; configs are seed-pooled labels
COMPARISONS = {
    "csf_read": [
        ("replay effect (Whisper cs+mono - cs)", "ev_ft_whisper_csmono_auto", "ev_ft_whisper_cs_auto"),
        ("fine-tuning (Whisper cs - zero-shot)", "ev_ft_whisper_cs_auto", "zs_whisper_auto"),
        ("fine-tuning (Whisper cs+mono - zero-shot)", "ev_ft_whisper_csmono_auto", "zs_whisper_auto"),
        ("LFM encoder+LM - LM only", "ev_ft_lfm_both", "ev_ft_lfm_lm"),
        ("LFM encoder only - LM only", "ev_ft_lfm_encoder", "ev_ft_lfm_lm"),
        ("fine-tuning (LFM both - zero-shot p1)", "ev_ft_lfm_both", "zs_lfm_p1"),
        ("architecture, same data (Whisper cs+mono - LFM both)", "ev_ft_whisper_csmono_auto", "ev_ft_lfm_both"),
        ("ja+rp decoding (Whisper ja+rp - ja)", "zs_whisper_ja_rp", "zs_whisper_ja"),
    ],
    "mono_ja": [
        ("forgetting (Whisper cs - zero-shot)", "ev_ft_whisper_cs_auto", "zs_whisper_auto"),
        ("forgetting (Whisper cs+mono - zero-shot)", "ev_ft_whisper_csmono_auto", "zs_whisper_auto"),
        ("forgetting (LFM both - zero-shot p1)", "ev_ft_lfm_both", "zs_lfm_p1"),
        ("forgetting (LFM LM only - zero-shot p1)", "ev_ft_lfm_lm", "zs_lfm_p1"),
        ("forgetting (LFM encoder only - zero-shot p1)", "ev_ft_lfm_encoder", "zs_lfm_p1"),
    ],
}
COMPARISONS["mono_en"] = COMPARISONS["mono_ja"]
# Exploratory: the ja+rp decoding (forced ja + repetition penalty 1.3 + 3-gram
# block) on the fine-tuned adapters isolates the decoding effect
EXPLORATORY = {
    "csf_read": [
        ("ja+rp decoding on Whisper cs (ja+rp - auto)", "ev_ft_whisper_cs_ja_rp", "ev_ft_whisper_cs_auto"),
        ("ja+rp decoding on Whisper cs+mono (ja+rp - auto)",
         "ev_ft_whisper_csmono_ja_rp", "ev_ft_whisper_csmono_auto"),
        ("replay effect under ja+rp decoding (cs+mono - cs, both ja+rp)",
         "ev_ft_whisper_csmono_ja_rp", "ev_ft_whisper_cs_ja_rp"),
        # --extra-epochs: dev-loss selection keeps epoch 1
        ("training length (Whisper cs, epoch 3 - epoch 1)", "ev_ft_whisper_cs_ep3_auto", "ev_ft_whisper_cs_ep1_auto"),
        ("training length (Whisper cs+mono, epoch 3 - epoch 1)",
         "ev_ft_whisper_csmono_ep3_auto", "ev_ft_whisper_csmono_ep1_auto"),
        ("replay effect at epoch 3 (cs+mono - cs)", "ev_ft_whisper_csmono_ep3_auto", "ev_ft_whisper_cs_ep3_auto"),
    ],
    "mono_ja": [
        ("training length (Whisper cs, epoch 3 - epoch 1)", "ev_ft_whisper_cs_ep3_auto", "ev_ft_whisper_cs_ep1_auto"),
        ("training length (Whisper cs+mono, epoch 3 - epoch 1)",
         "ev_ft_whisper_csmono_ep3_auto", "ev_ft_whisper_csmono_ep1_auto"),
    ],
}
EXPLORATORY["mono_en"] = EXPLORATORY["mono_ja"]
# Training reproducibility: the epoch-1 checkpoints of the --extra-epochs repeats are the
# ladder's dev-selected models trained again with the same seeds
REPRO = {t: [(f"repeat - ladder (Whisper {c}, dev-selected epoch 1)", f"ev_ft_whisper_{c}_ep1_auto",
              f"ev_ft_whisper_{c}_auto") for c in ("cs", "csmono")] for t in ("csf_read", "mono_ja", "mono_en")}
METRICS = {"csf_read": ("MER", "ScriptAcc", "ScriptAcc_strict", "KataKeep"),
           "mono_ja": ("JA_CER",), "mono_en": ("EN_WER",)}
RANK_SETS = ("synth_art", "csf_read")


def _add(a, b):
    """Sum two nested per-utterance count dicts (numeric leaves only)."""
    if isinstance(a, dict):
        return {k: _add(a[k], b.get(k, 0)) if k not in ("id", "cluster") else a[k] for k in a}
    return a + b if isinstance(a, (int, float)) else a


def load_configs(runs: Path, test: str) -> dict[str, list[dict]]:
    """Seed-pooled per-utterance counts per configuration on one test set."""
    per_cfg: dict[str, list[list[dict]]] = {}
    for p in sorted(runs.glob(f"*__{test}/preds.jsonl")):
        if not (p.parent / "DONE").exists():
            continue
        cfg = re.sub(r"_s\d+", "", p.parent.name.rsplit("__", 1)[0])
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows.sort(key=lambda r: r["id"])
        per_cfg.setdefault(cfg, []).append(S.per_utterance_counts(rows))
    pooled = {}
    for cfg, seeds in per_cfg.items():
        ids = [c["id"] for c in seeds[0]]
        if any([c["id"] for c in s] != ids for s in seeds):
            raise ValueError(f"{cfg}/{test}: seeds disagree on utterance ids")
        acc = seeds[0]
        for s in seeds[1:]:
            acc = [_add(a, b) for a, b in zip(acc, s)]
        pooled[cfg] = acc
    # every system on one test set must cover the same utterances (paired tests)
    if pooled:
        common = set.intersection(*({c["id"] for c in v} for v in pooled.values()))
        for cfg, v in pooled.items():
            if len(v) != len(common):
                print(f"WARNING {test}/{cfg}: {len(v) - len(common)} utterances not shared by "
                      f"all systems are dropped", file=sys.stderr)
            pooled[cfg] = [c for c in v if c["id"] in common]
    return pooled


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall's tau-b: tied pairs stay in the denominator (as scipy.stats.kendalltau)."""
    n0 = conc = disc = tie_x = tie_y = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        n0 += 1
        tie_x += dx == 0
        tie_y += dy == 0
        conc += dx * dy > 0
        disc += dx * dy < 0
    denom = math.sqrt((n0 - tie_x) * (n0 - tie_y))
    return (conc - disc) / denom if denom else float("nan")


def fmt_p(p: float | None, m: int, n_valid: int) -> str:
    """A bootstrap p of 0 only bounds p below 2/n; after Holm, below m * 2/n."""
    if p is None:
        return "n/a"
    if p > 0:
        return f"{p:.3g}"
    bound = min(1.0, m * 2 / max(n_valid, 1))
    step = 10 ** math.floor(math.log10(bound))
    return f"<{math.ceil(bound / step) * step:.1g}"


def compare(sets: dict, comps_by_test: dict, n: int, rng: random.Random | str, md: list, out: dict,
            metrics: dict | None = None) -> None:
    """Paired bootstrap per comparison; Holm across the comparisons of one test set and metric.
    rng: one shared stream (the pre-registered analysis), or a string that seeds a
    separate stream per comparison, so adding comparisons never shifts the others.
    metrics: test set -> metric names (default METRICS)."""
    metrics = metrics or METRICS
    for test, comps in comps_by_test.items():
        cfgs = sets(test)
        rows = []
        for label, a, b in comps:
            if a not in cfgs or b not in cfgs:
                rows.append({"label": label, "a": a, "b": b, "missing": True})
                continue
            r = random.Random(f"{rng}:{test}:{a}:{b}") if isinstance(rng, str) else rng
            res = S.paired_bootstrap(cfgs[a], cfgs[b], n, 95, r)
            pt = {k: S._aggregate_rates(cfgs[a])[k] - S._aggregate_rates(cfgs[b])[k] for k in metrics[test]}
            rows.append({"label": label, "a": a, "b": b, "diff": pt,
                         "ci": {k: res[k] for k in metrics[test]}})
        for k in metrics[test]:
            ok = [r for r in rows if not r.get("missing") and r["ci"][k].get("p_boot") is not None]
            for r, p in zip(ok, holm([r["ci"][k]["p_boot"] for r in ok])):
                r["ci"][k]["p_holm"] = round(p, 4)
                r["ci"][k]["m"] = len(ok)
        out[test] = rows
        md.append(f"\n## {test}\n\n| comparison (A - B) | " + " | ".join(
            f"Δ{k} [95% CI] p_holm" for k in metrics[test]) + " |\n|---|" + "---|" * len(metrics[test]))
        for r in rows:
            if r.get("missing"):
                md.append(f"| {r['label']} | " + " | ".join("missing" for _ in metrics[test]) + " |")
                continue
            cells = [f"{100 * r['diff'][k]:+.1f} [{100 * r['ci'][k]['lo']:+.1f}, "
                     f"{100 * r['ci'][k]['hi']:+.1f}] "
                     f"{fmt_p(r['ci'][k].get('p_holm'), r['ci'][k].get('m', 1), r['ci'][k]['n_valid'])}"
                     if r["ci"][k].get("lo") is not None else "n/a" for k in metrics[test]]
            md.append(f"| {r['label']} | " + " | ".join(cells) + " |")


def ranking(proxy: dict, real: dict, n: int, rng: random.Random) -> dict:
    """Kendall tau-b of the system rankings on two sets, with a bootstrap CI that
    resamples utterances in each set independently, plus the model-selection regret
    of picking the lowest-MER system on the proxy set (ties: every tied pick)."""
    common = sorted(set(proxy) & set(real))
    res: dict = {"systems": common}
    if len(common) < 3:
        return res
    # MER_capped last, so it never shifts the pre-registered MER / ScriptAcc draws
    for metric in ("MER", "ScriptAcc", "MER_capped"):
        point = kendall_tau(*[[S._aggregate_rates(s[c])[metric] for c in common] for s in (proxy, real)])
        dist = []
        for _ in range(min(n, 2000)):
            vals = []
            for s in (proxy, real):
                m = len(s[common[0]])
                idx = [rng.randrange(m) for _ in range(m)]
                vals.append([S._aggregate_rates([s[c][i] for i in idx])[metric] for c in common])
            tau = kendall_tau(*vals)
            if not math.isnan(tau):
                dist.append(tau)
        dist.sort()
        res[metric] = {"tau": point, "lo": dist[int(0.025 * len(dist))],
                       "hi": dist[int(0.975 * len(dist)) - 1]}
    for scope, pool in (("all systems", common), ("fine-tuned only", [c for c in common if c.startswith("ev_ft_")])):
        if not pool:
            continue
        p = {c: S._aggregate_rates(proxy[c])["MER"] for c in pool}
        r = {c: S._aggregate_rates(real[c])["MER"] for c in pool}
        picks = sorted(c for c in pool if p[c] == min(p.values()))
        res.setdefault("regret", {})[scope] = {
            "picks": picks, "real_best": min(r, key=r.get),
            "regret": [r[c] - min(r.values()) for c in picks]}
    return res


def ranking_md(name: str, res: dict) -> list[str]:
    lines = [f"\n### {name}, {len(res['systems'])} systems\n"]
    for metric in ("MER", "ScriptAcc", "MER_capped"):
        if metric in res:
            t = res[metric]
            note = " (exploratory)" if metric == "MER_capped" else ""
            lines.append(f"- Kendall τ-b ({metric}){note}: {t['tau']:+.2f} "
                         f"[95% CI {t['lo']:+.2f}, {t['hi']:+.2f}]")
    for scope, g in res.get("regret", {}).items():
        rg = g["regret"]
        span = f"{100 * min(rg):.1f}" if min(rg) == max(rg) else f"{100 * min(rg):.1f}–{100 * max(rg):.1f}"
        lines.append(f"- Selection regret, {scope} (exploratory): proxy picks {', '.join(g['picks'])}; "
                     f"real best {g['real_best']}; regret {span} MER")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/data/code-switching/runs/phase0")
    ap.add_argument("--n", type=int, default=5000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loanwords", default=None,
                    help="Loanword lexicon: adds the exploratory script-choice gap (MER - MER_lenient)")
    ap.add_argument("--gap-sets", default="csf_read,csyodas,twins_kokoro,twins_edge",
                    help="CS test sets for the script-choice gap table")
    ap.add_argument("--rank-pairs", default="synth_diss:csf_read,twins_kokoro:csf_read,"
                                            "twins_edge:csf_read,csyodas:csf_read",
                    help="Further PROXY:REFERENCE set pairs for the ranking-fidelity section")
    ap.add_argument("--capped-sets", default="csf_read,csyodas,twins_kokoro,twins_edge",
                    help="Sets for the runaway-robust table (MER_capped, Runaway; scorer >= 2.2)")
    args = ap.parse_args()
    runs = Path(args.runs)
    if args.loanwords:
        S.load_loanwords(args.loanwords)
    cache: dict[str, dict] = {}

    def sets(test: str) -> dict:
        if test not in cache:
            cache[test] = load_configs(runs, test)
        return cache[test]

    out: dict = {"n_boot": args.n, "comparisons": {}, "ranking": {}}
    md = [f"# Phase 0 ladder statistics\n", f"Paired bootstrap ({args.n} resamples, seeds pooled "
          f"per utterance); p-values Holm-adjusted per test set and metric. A bootstrap p of 0 "
          f"is shown as its bound (<m·2/n).\n"]

    # pre-registered: comparisons, then the synth_art vs csf_read ranking (one RNG stream)
    rng = random.Random(args.seed)
    compare(sets, COMPARISONS, args.n, rng, md, out["comparisons"])
    res = ranking(sets(RANK_SETS[0]), sets(RANK_SETS[1]), args.n, rng)
    out["ranking"][f"{RANK_SETS[0]}:{RANK_SETS[1]}"] = res
    md.append(f"\n## Ranking fidelity (pre-registered Q4)")
    md += ranking_md(f"{RANK_SETS[0]} vs {RANK_SETS[1]}", res)

    # exploratory sections: own RNG streams, so they never shift the pre-registered draws
    md.append("\n# Exploratory analyses (not pre-registered)")
    for key, family, note in (
            ("exploratory_comparisons", EXPLORATORY,
             "Own Holm family, separate from the pre-registered comparisons."),
            ("reproducibility", REPRO,
             "\n## Training reproducibility\n\nThe same trainings run again with the same seeds; "
             "a CI that covers 0 means the ladder's numbers reproduce.")):
        present = {t: [c for c in comps if all(x in sets(t) for x in c[1:])] for t, comps in family.items()}
        present = {t: c for t, c in present.items() if c}
        if present:
            md.append(note)
            out[key] = {}
            compare(sets, present, args.n, f"{args.seed}:{key}", md, out[key])

    pairs = [tuple(p.split(":", 1)) for p in args.rank_pairs.split(",") if ":" in p]
    pairs = [(a, b) for a, b in pairs if sets(a) and sets(b)]
    if pairs:
        md.append("\n## Ranking fidelity, further set pairs\n\nsynth_diss (n=30) is a pre-registered "
                  "ranking set, but only the synth_art τ was named in the protocol; the twins "
                  "(same text as csf_read, synthetic audio) and csyodas are exploratory sets.")
        for a, b in pairs:
            res = ranking(sets(a), sets(b), args.n, random.Random(f"{args.seed}:rank:{a}:{b}"))
            out["ranking"][f"{a}:{b}"] = res
            md += ranking_md(f"{a} vs {b}", res)

    # how much of each system's MER is a few runaway hypotheses?
    capped_sets = [t for t in args.capped_sets.split(",") if t and sets(t)]
    if capped_sets:
        out["capped"] = {}
        md.append("\n## Runaway-robust MER (each hypothesis's errors capped at its reference "
                  "length; seeds pooled)\n\nRunaway = share of hypotheses with errors above "
                  "2x their reference tokens.")
        for test in capped_sets:
            cfgs = sets(test)
            md.append(f"\n### {test}\n\n| system | MER | MER_capped [95% CI] | Runaway |\n|---|---|---|---|")
            out["capped"][test] = {}
            for cfg in sorted(cfgs, key=lambda c: S._aggregate_rates(cfgs[c])["MER_capped"]):
                counts = cfgs[cfg]
                agg = S._aggregate_rates(counts)
                crng = random.Random(f"{args.seed}:capped:{test}:{cfg}")
                dist = sorted(S._aggregate_rates([counts[crng.randrange(len(counts))] for _ in counts])
                              ["MER_capped"] for _ in range(min(args.n, 2000)))
                lo, hi = dist[int(0.025 * len(dist))], dist[int(0.975 * len(dist)) - 1]
                out["capped"][test][cfg] = {"MER": agg["MER"], "MER_capped": agg["MER_capped"],
                                            "lo": lo, "hi": hi, "Runaway": agg["Runaway"]}
                md.append(f"| {cfg} | {100 * agg['MER']:.1f} | {100 * agg['MER_capped']:.1f} "
                          f"[{100 * lo:.1f}, {100 * hi:.1f}] | {100 * agg['Runaway']:.1f}% |")

    # how much of each system's MER is script choice alone?
    if args.loanwords:
        out["script_gap"] = {}
        md.append("\n## Script-choice gap (MER - MER_lenient, seeds pooled)\n")
        for test in [t for t in args.gap_sets.split(",") if t]:
            cfgs = sets(test)
            if not cfgs:
                continue
            md.append(f"\n### {test}\n\n| system | MER | MER_lenient | gap [95% CI] |\n|---|---|---|---|")
            out["script_gap"][test] = {}
            for cfg in sorted(cfgs):
                counts = cfgs[cfg]
                agg = S._aggregate_rates(counts)
                grng = random.Random(f"{args.seed}:gap:{test}:{cfg}")
                gaps = []
                for _ in range(min(args.n, 2000)):
                    idx = [grng.randrange(len(counts)) for _ in counts]
                    a = S._aggregate_rates([counts[i] for i in idx])
                    gaps.append(a["MER"] - a["MER_lenient"])
                gaps.sort()
                lo, hi = gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps)) - 1]
                gap = agg["MER"] - agg["MER_lenient"]
                out["script_gap"][test][cfg] = {"MER": agg["MER"], "MER_lenient": agg["MER_lenient"],
                                                "gap": gap, "lo": lo, "hi": hi}
                md.append(f"| {cfg} | {100 * agg['MER']:.1f} | {100 * agg['MER_lenient']:.1f} | "
                          f"{100 * gap:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] |")

    (runs / "stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (runs / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
