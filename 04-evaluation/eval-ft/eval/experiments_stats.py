#!/usr/bin/env python3
"""Pre-registered statistics for Phase 1 experiments v1
(pre-registered protocol).

Configurations are seed-pooled per utterance (ladder_stats.load_configs) and drawn
from three run directories: the experiments, the Phase 0 ladder (large-v3 LoRA
references) and the Phase 1 leaderboard (turbo zero-shot). Paired bootstrap by
id; Holm across the comparisons of one experiment, test set and metric; one
seeded random stream per comparison.

Usage:
  $PY eval/experiments_stats.py [--n 5000]
Writes EXP_RUNS/stats.md and stats.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402
from ladder_stats import compare, load_configs  # noqa: E402

E1 = {
    "csf_read": [(f"turbo - large-v3 LoRA, {c}", f"ev_ft_turbo_{c}_auto", f"ev_ft_whisper_{c}_auto") for c in ("cs", "csmono")]
                + [(f"turbo LoRA - turbo zero-shot, {c}", f"ev_ft_turbo_{c}_auto", "lb_whisper_turbo_auto") for c in ("cs", "csmono")],
}
for t in ("csyodas", "mono_ja", "mono_en"):
    E1[t] = E1["csf_read"]
E1_METRICS = {"csf_read": ("MER", "MER_capped", "ScriptAcc_strict", "KataKeep"), "csyodas": ("MER_capped",),
              "mono_ja": ("JA_CER",), "mono_en": ("EN_WER",)}
E2 = {
    "csf_read": [(f"none - auto labels, {c}, {d} decoding", f"ev_ft_nolang_{c}_{d}", f"ev_ft_whisper_{c}_{d}")
                 for d in ("auto", "ja") for c in ("cs", "csmono")],
    "mono_ja": [(f"none - auto labels, {c}", f"ev_ft_nolang_{c}_auto", f"ev_ft_whisper_{c}_auto") for c in ("cs", "csmono")],
}
E2["mono_en"] = E2["mono_ja"]
E2_METRICS = {"csf_read": ("MER", "ScriptAcc_strict", "KataKeep"), "mono_ja": ("JA_CER",), "mono_en": ("EN_WER",)}
E3 = {"csf_read": [(f"Qwen3-ASR-0.6B FT - large-v3 LoRA, {c}", f"ev_ft_qwen06_{c}_auto", f"ev_ft_whisper_{c}_auto")
                   for c in ("cs", "csmono")]
                  + [(f"Qwen3-ASR-0.6B FT - zero-shot, {c}", f"ev_ft_qwen06_{c}_auto", "lb_qwen3asr_0.6b_auto")
                     for c in ("cs", "csmono")]}
for t in ("csyodas", "mono_ja", "mono_en"):
    E3[t] = E3["csf_read"]
E4 = {"csf_read": [(f"Granite FT {v} - large-v3 LoRA, {c}", f"ev_ft_granite{v}_{c}_auto", f"ev_ft_whisper_{c}_auto")
                   for v in ("a", "b") for c in ("cs", "csmono")]
                  + [(f"Granite FT {v} - zero-shot, {c}", f"ev_ft_granite{v}_{c}_auto", "lb_granite4_1b")
                     for v in ("a", "b") for c in ("cs", "csmono")]}
for t in ("csyodas", "mono_ja", "mono_en"):
    E4[t] = E4["csf_read"]
E5 = {"csf_read": [(f"{lab} {lr} - {ref_lab}", f"ev_ft_{back}{lr}_csmono_auto", ref)
                   for back, lab, zs in (("qwen06lora", "Qwen3-ASR-0.6B LoRA", "lb_qwen3asr_0.6b_auto"),
                                         ("granitelora", "Granite LoRA", "lb_granite4_1b"))
                   for lr in ("1e4", "2e5")
                   for ref_lab, ref in (("zero-shot", zs), ("large-v3 LoRA csmono", "ev_ft_whisper_csmono_auto"),
                                        ("turbo LoRA csmono", "ev_ft_turbo_csmono_auto"))]}
for t in ("csyodas", "mono_ja", "mono_en"):
    E5[t] = E5["csf_read"]
# exploratory: E2 on CS-YODAS
X = {"csyodas": [(f"none - auto labels, {c} (exploratory)", f"ev_ft_nolang_{c}_auto", f"ev_ft_whisper_{c}_auto")
                 for c in ("cs", "csmono")]}
X_METRICS = {"csyodas": ("MER_capped",)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/data/code-switching/runs/phase1_exp")
    ap.add_argument("--phase0-runs", default="/data/code-switching/runs/phase0")
    ap.add_argument("--leaderboard-runs", default="/data/code-switching/runs/phase1_leaderboard")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dirs = [Path(args.runs), Path(args.phase0_runs), Path(args.leaderboard_runs)]
    cache: dict[str, dict] = {}

    def sets(test: str) -> dict:
        if test not in cache:
            merged: dict = {}
            for d in dirs:
                merged.update(load_configs(d, test))
            common = set.intersection(*({c["id"] for c in v} for v in merged.values()))
            cache[test] = {k: [c for c in v if c["id"] in common] for k, v in merged.items()}
        return cache[test]

    out: dict = {"n_boot": args.n}
    md = ["# Phase 1 experiments v1: statistics\n",
          f"Paired bootstrap by id ({args.n} resamples, seeds pooled); Holm per experiment, test set "
          f"and metric. A bootstrap p of 0 is shown as its bound (<m·2/n).\n"]
    for key, title, fam, metrics in (
            ("E1", "E1: Whisper-large-v3-turbo with the Phase 0 recipe", E1, E1_METRICS),
            ("E2", "E2: labels without a language token", E2, E2_METRICS),
            ("E3", "E3: Qwen3-ASR-0.6B fine-tuned", E3, E1_METRICS),
            ("E4", "E4: Granite-4.0-1b-speech fine-tuned", E4, E1_METRICS),
            ("E5", "E5: LLM-only LoRA for Qwen3-ASR-0.6B and Granite", E5, E1_METRICS),
            ("exploratory", "Exploratory", X, X_METRICS)):
        md.append(f"\n# {title}")
        out[key] = {}
        compare(sets, fam, args.n, f"{args.seed}:{key}", md, out[key], metrics)
    runs = Path(args.runs)
    (runs / "stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (runs / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
