#!/usr/bin/env python3
"""Pre-registered analysis of the human validation of the script-aware metrics
(pre-registered metric-validation protocol). Written and committed before
any rating exists.

Inputs
  --key      key.jsonl from make_pairs.py (systems, stratum, per-transcript metrics)
  --ratings  one JSON object per rating:
             {"pair_id", "rater", "q1": "A"|"B"|"equal",
              "q2": "none"|"A"|"B"|"both", "q3_A": 1-5, "q3_B": 1-5}

Analyses (protocol section "Analysis")
  * Inter-rater agreement on Q1: Cohen's kappa per rater pair, Fleiss' kappa on
    pairs rated by 3 or more raters. Target: kappa >= 0.6.
  * Metric-human agreement: the human label of a pair is the strict majority of
    its Q1 answers (no majority, or a majority for "equal", is a human tie).
    Over pairs with a non-tied human label, a metric agrees when it prefers the
    same transcript; a metric tie or an undefined value counts as not agreeing.
    Metrics: MER, MER_lenient (lower is better), ScriptAcc, ScriptAcc_strict,
    KataKeep (higher is better), and MER_lenient + lambda * (1 - KataKeep) with
    lambda chosen by 2-fold cross-fitting over utterances (fit on one fold,
    scored on the other). Bootstrap 95% CIs over pairs; exact McNemar test of
    MER against MER_lenient on the same pairs. Reported overall and per stratum.
  * Rank correlation: Kendall tau-b over systems between each metric (mean over
    the system's rated transcripts) and its mean Q3 adequacy.
  * H1: on stratum 1 (metric_flip), MER_lenient agrees with humans more often
    than MER (McNemar, two-sided, alpha 0.05, and the direction must match).
  * H2: ScriptAcc alone does not predict Q2 unnaturalness; ScriptAcc_strict and
    KataKeep together do. A transcript is unnatural when a strict majority of its
    raters flag it (Q2 = it or "both"). Measured by 2-fold cross-fitted AUC of a
    logistic model on (a) ScriptAcc and (b) ScriptAcc_strict + KataKeep, over
    transcripts where all three are defined, with bootstrap CIs.

Usage
  $PY human_eval/analyse.py --key pairs_v1/key.jsonl --ratings ratings.jsonl --out report
  $PY human_eval/analyse.py --key pairs_v1/key.jsonl --selftest   # simulated raters
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

LOWER_BETTER = ("MER", "MER_lenient")
HIGHER_BETTER = ("ScriptAcc", "ScriptAcc_strict", "KataKeep")
METRICS = LOWER_BETTER + HIGHER_BETTER
LAMBDAS = [i / 20 for i in range(21)]          # composite weight grid, 0 .. 1


# ── rater agreement ───────────────────────────────────────────────────────────

def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss_kappa(items: list[list[str]], cats=("A", "B", "equal")) -> float:
    """items: the answers per item; every item must have the same number of raters."""
    if not items:
        return float("nan")
    m = len(items[0])
    n = len(items)
    p_j = {c: sum(it.count(c) for it in items) / (n * m) for c in cats}
    p_i = [(sum(it.count(c) ** 2 for c in cats) - m) / (m * (m - 1)) for it in items]
    pbar, pe = sum(p_i) / n, sum(v * v for v in p_j.values())
    return (pbar - pe) / (1 - pe) if pe < 1 else float("nan")


# ── labels and metric preferences ─────────────────────────────────────────────

def majority(answers: list[str]) -> str | None:
    """Strict majority, or None when there is none."""
    c = Counter(answers).most_common()
    if c and c[0][1] * 2 > len(answers):
        return c[0][0]
    return None


def metric_pref(metric: str, a: float | None, b: float | None) -> str | None:
    if a is None or b is None or a == b:
        return None
    if metric in HIGHER_BETTER:
        return "A" if a > b else "B"
    return "A" if a < b else "B"


def composite(m: dict, lam: float) -> float:
    kk = m.get("KataKeep")
    return m["MER_lenient"] + lam * (1 - kk if kk is not None else 0.0)


def agreement(pairs: list[dict], pref) -> tuple[float, int]:
    """Share of pairs (non-tied human label) where pref(pair) equals the human label."""
    hits = [pref(p) == p["human"] for p in pairs]
    return (sum(hits) / len(hits) if hits else float("nan")), len(hits)


def boot_ci(values: list[float], n: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    n0 = conc = disc = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        n0 += 1
        tx += dx == 0
        ty += dy == 0
        conc += dx * dy > 0
        disc += dx * dy < 0
    d = math.sqrt((n0 - tx) * (n0 - ty))
    return (conc - disc) / d if d else float("nan")


# ── H2: cross-fitted logistic AUC ─────────────────────────────────────────────

def fit_logistic(X: list[list[float]], y: list[int], iters: int = 50) -> list[float]:
    """Newton-Raphson logistic regression with an intercept and a small ridge."""
    k = len(X[0]) + 1
    w = [0.0] * k
    for _ in range(iters):
        g = [0.0] * k
        H = [[1e-3 * (i == j) for j in range(k)] for i in range(k)]
        for xi, yi in zip(X, y):
            v = [1.0] + xi
            z = sum(a * b for a, b in zip(w, v))
            p = 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
            for i in range(k):
                g[i] += (yi - p) * v[i]
                for j in range(k):
                    H[i][j] += p * (1 - p) * v[i] * v[j]
        step = _solve(H, g)
        w = [a + b for a, b in zip(w, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    return w


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[c][c]:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] if M[i][i] else 0.0 for i in range(n)]


def auc(scores: list[float], y: list[int]) -> float:
    """Mann-Whitney AUC from average ranks (ties count one half)."""
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, t in zip(ranks, y) if t)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def crossfit_scores(rows: list[dict], feats: tuple[str, ...], folds: dict[str, int]) -> list[float]:
    out = [0.0] * len(rows)
    for f in (0, 1):
        train = [r for r in rows if folds[r["utt"]] != f]
        w = fit_logistic([[r[x] for x in feats] for r in train], [r["y"] for r in train])
        for i, r in enumerate(rows):
            if folds[r["utt"]] == f:
                out[i] = sum(a * b for a, b in zip(w, [1.0] + [r[x] for x in feats]))
    return out


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse(key: list[dict], ratings: list[dict], n_boot: int, seed: int) -> tuple[dict, list[str]]:
    rng = random.Random(seed)
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in ratings:
        by_pair[r["pair_id"]].append(r)
    keyd = {k["pair_id"]: k for k in key}
    res: dict = {"n_pairs_key": len(key), "n_ratings": len(ratings), "n_pairs_rated": len(by_pair)}
    md = ["# Human validation of the script-aware metrics: analysis\n",
          f"{len(ratings)} ratings of {len(by_pair)} / {len(key)} pairs.\n"]

    # inter-rater agreement (Q1)
    raters = sorted({r["rater"] for r in ratings})
    kappas = {}
    for a, b in itertools.combinations(raters, 2):
        both = [(next(x["q1"] for x in rs if x["rater"] == a), next(x["q1"] for x in rs if x["rater"] == b))
                for rs in by_pair.values() if {a, b} <= {x["rater"] for x in rs}]
        if both:
            kappas[f"{a}~{b}"] = {"kappa": cohen_kappa(*map(list, zip(*both))), "n": len(both)}
    multi = [[x["q1"] for x in rs][:3] for rs in by_pair.values() if len(rs) >= 3]
    res["agreement"] = {"cohen": kappas, "fleiss_3": fleiss_kappa(multi), "n_fleiss": len(multi)}
    md.append("## Inter-rater agreement (Q1)\n")
    md += [f"- Cohen's κ {k}: {v['kappa']:.2f} (n = {v['n']})" for k, v in kappas.items()]
    md.append(f"- Fleiss' κ, pairs with 3 raters: {res['agreement']['fleiss_3']:.2f} (n = {len(multi)}); "
              f"target ≥ 0.6")

    # human labels
    pairs = []
    for pid, rs in by_pair.items():
        k = keyd[pid]
        lab = majority([x["q1"] for x in rs])
        pairs.append({"pid": pid, "stratum": k["stratum"], "utt": f"{k['test']}/{k['utt_id']}",
                      "human": lab if lab in ("A", "B") else None,
                      "mA": k["metrics_A"], "mB": k["metrics_B"], "k": k, "rs": rs})
    decided = [p for p in pairs if p["human"]]
    res["human_ties"] = len(pairs) - len(decided)

    # composite lambda by 2-fold cross-fitting over utterances
    utts = sorted({p["utt"] for p in pairs})
    frng = random.Random(seed + 1)
    folds = {u: frng.randrange(2) for u in utts}
    comp_pref = {}
    lam_chosen = {}
    for f in (0, 1):
        fit = [p for p in decided if folds[p["utt"]] != f]
        best = max(LAMBDAS, key=lambda lam: (agreement(fit, lambda p: metric_pref(
            "MER_lenient", composite(p["mA"], lam), composite(p["mB"], lam)))[0], -lam))
        lam_chosen[f] = best
        for p in pairs:
            if folds[p["utt"]] == f:
                comp_pref[p["pid"]] = metric_pref("MER_lenient", composite(p["mA"], best),
                                                  composite(p["mB"], best))
    res["composite_lambda_by_fold"] = lam_chosen

    prefs = {m: (lambda p, m=m: metric_pref(m, p["mA"].get(m), p["mB"].get(m))) for m in METRICS}
    prefs["composite"] = lambda p: comp_pref.get(p["pid"])
    md.append("\n## Metric–human agreement (Q1, non-tied human labels)\n")
    md.append("| subset | n | " + " | ".join(prefs) + " | McNemar MER vs MER_lenient (b, c, p) |")
    md.append("|---|---|" + "---|" * len(prefs) + "---|")
    res["metric_agreement"] = {}
    for name, subset in [("all", decided)] + [(s, [p for p in decided if p["stratum"] == s])
                                               for s in ("metric_flip", "script_flip", "close", "random")]:
        row = {}
        cells = []
        for m, pref in prefs.items():
            hits = [float(pref(p) == p["human"]) for p in subset]
            a = sum(hits) / len(hits) if hits else float("nan")
            lo, hi = boot_ci(hits, n_boot, rng)
            row[m] = {"agree": a, "lo": lo, "hi": hi}
            cells.append(f"{100 * a:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]" if hits else "n/a")
        b = sum(prefs["MER"](p) == p["human"] and prefs["MER_lenient"](p) != p["human"] for p in subset)
        c = sum(prefs["MER"](p) != p["human"] and prefs["MER_lenient"](p) == p["human"] for p in subset)
        row["mcnemar"] = {"b_mer_only": b, "c_lenient_only": c, "p": mcnemar_exact(b, c)}
        res["metric_agreement"][name] = row
        md.append(f"| {name} | {len(subset)} | " + " | ".join(cells) +
                  f" | {b}, {c}, {row['mcnemar']['p']:.3g} |")
    md.append(f"\nComposite = MER_lenient + λ·(1 − KataKeep); λ per fold (fit on the other fold): {lam_chosen}.")

    h1 = res["metric_agreement"]["metric_flip"]["mcnemar"]
    res["H1"] = {"supported": h1["c_lenient_only"] > h1["b_mer_only"] and h1["p"] < 0.05, **h1}
    md.append(f"\n**H1** (stratum 1: MER_lenient agrees more often than MER): "
              f"{'supported' if res['H1']['supported'] else 'not supported'} "
              f"(lenient-only {h1['c_lenient_only']}, MER-only {h1['b_mer_only']}, p = {h1['p']:.3g}).")

    # rank correlation with Q3 adequacy, over systems
    per_sys = defaultdict(lambda: {"q3": [], **{m: [] for m in METRICS}})
    for p in pairs:
        for side in ("A", "B"):
            s = per_sys[p["k"][f"system_{side}"]]
            s["q3"] += [x[f"q3_{side}"] for x in p["rs"]]
            for m in METRICS:
                v = p[f"m{side}"].get(m)
                if v is not None:
                    s[m].append(v)
    systems = sorted(k for k, v in per_sys.items() if v["q3"])
    q3 = [sum(per_sys[s]["q3"]) / len(per_sys[s]["q3"]) for s in systems]
    res["rank_correlation"] = {}
    md.append("\n## Kendall τ-b over systems: metric (mean over rated transcripts) vs mean Q3 adequacy\n")
    for m in METRICS:
        ok = [i for i, s in enumerate(systems) if per_sys[s][m]]
        vals = [sum(per_sys[systems[i]][m]) / len(per_sys[systems[i]][m]) for i in ok]
        sign = -1 if m in LOWER_BETTER else 1       # oriented so that +1 = agrees with humans
        tau = sign * kendall_tau_b(vals, [q3[i] for i in ok])
        res["rank_correlation"][m] = {"tau_b": tau, "n_systems": len(ok)}
        md.append(f"- {m}: {tau:+.2f} ({len(ok)} systems; +1 = same order as humans)")

    # H2: Q2 unnaturalness
    rows = []
    for p in pairs:
        for side in ("A", "B"):
            m = p[f"m{side}"]
            if None in (m.get("ScriptAcc"), m.get("ScriptAcc_strict"), m.get("KataKeep")):
                continue
            flags = [x["q2"] in (side, "both") for x in p["rs"]]
            rows.append({"utt": p["utt"], "y": int(sum(flags) * 2 > len(flags)),
                         "ScriptAcc": m["ScriptAcc"], "ScriptAcc_strict": m["ScriptAcc_strict"],
                         "KataKeep": m["KataKeep"]})
    res["H2"] = {"n_transcripts": len(rows), "n_unnatural": sum(r["y"] for r in rows)}
    md.append(f"\n## H2: predicting Q2 unnaturalness ({len(rows)} transcripts with all three metrics, "
              f"{res['H2']['n_unnatural']} flagged)\n")
    if len(rows) >= 20 and 0 < res["H2"]["n_unnatural"] < len(rows):
        for label, feats in (("ScriptAcc", ("ScriptAcc",)),
                             ("ScriptAcc_strict + KataKeep", ("ScriptAcc_strict", "KataKeep"))):
            s = crossfit_scores(rows, feats, folds)
            y = [r["y"] for r in rows]
            point = auc(s, y)
            dist = []
            for _ in range(min(n_boot, 1000)):
                idx = [rng.randrange(len(rows)) for _ in rows]
                v = auc([s[i] for i in idx], [y[i] for i in idx])
                if not math.isnan(v):
                    dist.append(v)
            dist.sort()
            lo, hi = dist[int(0.025 * len(dist))], dist[int(0.975 * len(dist)) - 1]
            res["H2"][label] = {"auc": point, "lo": lo, "hi": hi}
            md.append(f"- {label}: cross-fitted AUC {point:.2f} [{lo:.2f}, {hi:.2f}]")
        a, j = res["H2"]["ScriptAcc"], res["H2"]["ScriptAcc_strict + KataKeep"]
        res["H2"]["supported"] = a["lo"] <= 0.5 and j["lo"] > 0.5
        md.append(f"\n**H2**: {'supported' if res['H2']['supported'] else 'not supported'} "
                  f"(ScriptAcc CI covers 0.5 and the joint model's CI excludes it).")
    else:
        md.append("- too few transcripts or no variation in the flags; H2 not evaluated")
    return res, md


def simulate(key: list[dict], seed: int) -> list[dict]:
    """Self-test ratings: 3 raters who prefer the lower MER_lenient with 15% noise,
    call near-equal pairs 'equal', and flag a transcript as unnatural when its
    KataKeep is low and its ScriptAcc_strict is below its ScriptAcc."""
    rng = random.Random(seed)
    out = []
    for i, k in enumerate(key):
        a, b = k["metrics_A"], k["metrics_B"]
        n_raters = 3 if i % 5 == 0 else 2
        for r in range(n_raters):
            d = a["MER_lenient"] - b["MER_lenient"]
            q1 = "equal" if abs(d) < 0.01 else ("A" if d < 0 else "B")
            if rng.random() < 0.15:
                q1 = rng.choice(["A", "B", "equal"])
            odd = [s for s, m in (("A", a), ("B", b)) if m.get("KataKeep") is not None
                   and m["KataKeep"] < 0.5 and (m.get("ScriptAcc_strict") or 0) < (m.get("ScriptAcc") or 0)]
            q2 = "both" if len(odd) == 2 else (odd[0] if odd else "none")
            q3 = {s: max(1, min(5, round(5 - 6 * m["MER_lenient"] + rng.gauss(0, 0.5)))) for s, m in (("A", a), ("B", b))}
            out.append({"pair_id": k["pair_id"], "rater": f"r{(i + r) % 3 + 1}", "q1": q1, "q2": q2,
                        "q3_A": q3["A"], "q3_B": q3["B"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True)
    ap.add_argument("--ratings")
    ap.add_argument("--selftest", action="store_true", help="Analyse simulated ratings instead")
    ap.add_argument("--out", default=None, help="Writes OUT.json and OUT.md")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    key = [json.loads(l) for l in open(args.key, encoding="utf-8") if l.strip()]
    if args.selftest:
        ratings = simulate(key, args.seed)
    elif args.ratings:
        ratings = [json.loads(l) for l in open(args.ratings, encoding="utf-8") if l.strip()]
    else:
        raise SystemExit("--ratings or --selftest is required")
    unknown = {r["pair_id"] for r in ratings} - {k["pair_id"] for k in key}
    if unknown:
        raise SystemExit(f"{len(unknown)} rated pair ids are not in the key")
    res, md = analyse(key, ratings, args.n, args.seed)
    if args.selftest:
        md.insert(1, "**SELF-TEST on simulated raters — not a result.**\n")
    print("\n".join(md))
    if args.out:
        Path(args.out + ".json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str),
                                            encoding="utf-8")
        Path(args.out + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
