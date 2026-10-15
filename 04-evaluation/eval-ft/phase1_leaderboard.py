#!/usr/bin/env python3
"""Phase 1 open-model leaderboard.

Every baseline system is evaluated zero-shot on every test set, with the same
normaliser and scorer (score.py v2.2: MER_lenient via the loanword lexicon, and
the runaway-robust MER_capped). Systems are pinned to the model snapshots
downloaded by the first job. Results are merged with the Phase 0 ladder's
pre-registered zero-shot and fine-tuned systems into one leaderboard; the
ladder's exploratory runs (epoch checkpoints, ja+rp decoding) are left out.

Test sets:
  real        csf_read (CS-FLEURS read/test), csyodas (in the wild), switchlingua
              (SwitchLingua Japanese, read, 290 speakers; research-only licence),
              mono_ja / mono_en (FLEURS controls)
  synthetic   twins_kokoro / twins_edge (the csf_read text, TTS-rendered),
              synth_art / synth_diss (legacy template sets)

Uses the shared memory-aware scheduler (common/jobs.py). Resumable.

Usage (mactrn01, repo root):
  PY=/data/code-switching/envs/liquid-ai-research/bin/python
  $PY 04-evaluation/eval-ft/phase1_leaderboard.py --dry-run
  $PY 04-evaluation/eval-ft/phase1_leaderboard.py --gpus 0,1,2,3
  $PY 04-evaluation/eval-ft/phase1_leaderboard.py --score-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "04-evaluation" / "eval-ft"
sys.path.insert(0, str(REPO))
from common.jobs import Job, is_done, run_jobs, select  # noqa: E402

PY_MAIN = "/data/code-switching/envs/liquid-ai-research/bin/python"
PY_QWEN = "/data/code-switching/envs/qwen3-asr/bin/python"
PY_FUNASR = "/data/code-switching/envs/funasr/bin/python"
PY_QWEN3OMNI = "/data/code-switching/envs/qwen3-omni/bin/python"
D = Path("/data/code-switching/data")
TESTS = {   # name -> (manifest, audio root)
    "csf_read": (D / "eval-real/csfleurs/read_test/manifest.jsonl", D / "eval-real/csfleurs"),
    "csyodas": (D / "eval-real/csyodas/manifest.jsonl", D / "eval-real/csyodas"),
    # research-only licence: evaluation for papers, never product decisions
    "switchlingua": (D / "eval-real/switchlingua/manifest.jsonl", D / "eval-real/switchlingua"),
    "mono_ja": (D / "eval-mono/ja_jp_test/manifest.jsonl", D / "eval-mono"),
    "mono_en": (D / "eval-mono/en_us_test/manifest.jsonl", D / "eval-mono"),
    "twins_kokoro": (D / "eval-twins/kokoro/eval.jsonl", Path("/")),
    "twins_edge": (D / "eval-twins/edge/eval.jsonl", Path("/")),
    "synth_art": (D / "eval-synth/csartificial_manifest.jsonl", Path("/")),
    "synth_diss": (D / "eval-diss-synth/manifest.jsonl", Path("/")),
}
# system name -> (python, runner, model id, extra args)
SYSTEMS = {
    "whisper_turbo_auto": (PY_MAIN, "eval/run_whisper_hf.py", "openai/whisper-large-v3-turbo", []),
    "kotoba_bilingual_auto": (PY_MAIN, "eval/run_whisper_hf.py", "kotoba-tech/kotoba-whisper-bilingual-v1.0", []),
    "kotoba_v22_ja": (PY_MAIN, "eval/run_whisper_hf.py", "kotoba-tech/kotoba-whisper-v2.2", ["--language", "ja"]),
    "qwen3asr_1.7b_auto": (PY_QWEN, "eval/run_qwen3_asr.py", "Qwen/Qwen3-ASR-1.7B", []),
    "qwen3asr_1.7b_ja": (PY_QWEN, "eval/run_qwen3_asr.py", "Qwen/Qwen3-ASR-1.7B", ["--language", "Japanese"]),
    "qwen3asr_0.6b_auto": (PY_QWEN, "eval/run_qwen3_asr.py", "Qwen/Qwen3-ASR-0.6B", []),
    "qwen3asr_0.6b_ja": (PY_QWEN, "eval/run_qwen3_asr.py", "Qwen/Qwen3-ASR-0.6B", ["--language", "Japanese"]),
    "granite4_1b": (PY_MAIN, "eval/run_audio_llm.py", "ibm-granite/granite-4.0-1b-speech",
                    ["--family", "granite"]),
    "funasr_nano_auto": (PY_FUNASR, "eval/run_funasr.py", "FunAudioLLM/Fun-ASR-Nano-2512", []),
    "funasr_nano_ja": (PY_FUNASR, "eval/run_funasr.py", "FunAudioLLM/Fun-ASR-Nano-2512",
                       ["--language", "日文"]),
    # Transformers runs Qwen3-Omni's 128 experts in a Python loop, so batching matters: the
    # leaderboard run took 3.0 s/utt on csf_read at batch 8 on a whole H100. Batch-8 greedy
    # output differs from batch 1 on hard clips (left padding, bf16), so the batch size is part
    # of this system's definition. vLLM wheels need CUDA >= 12.8, which the host's driver
    # (565, CUDA 12.7) lacks.
    "qwen3omni_30b_b8": (PY_QWEN3OMNI, "eval/run_audio_llm.py", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                         ["--family", "qwen3omni", "--batch-size", "8"]),
}
MEM_GB = {"qwen3omni_30b_b8": 80}   # declared peak GPU memory for systems that need a whole GPU
SYSTEM_TESTS = {"qwen3omni_30b_b8": ("csf_read", "csyodas", "switchlingua")}   # expensive systems: real CS sets only
LB_METRICS = {"csf_read": ("MER", "MER_lenient", "MER_capped", "ScriptAcc_strict", "KataKeep"),
              "csyodas": ("MER", "MER_lenient", "MER_capped", "ScriptAcc_strict", "KataKeep"),
              "switchlingua": ("MER", "MER_lenient", "MER_capped", "CER_yv", "ScriptAcc_strict", "KataKeep"),
              "twins_kokoro": ("MER", "MER_lenient"), "twins_edge": ("MER", "MER_lenient"),
              "mono_ja": ("JA_CER",), "mono_en": ("EN_WER",),
              "synth_art": ("MER",), "synth_diss": ("MER",)}
# Phase 0 exploratory runs stay out of the leaderboard: epoch-checkpoint repeats and the
# ja+rp decoding of the fine-tuned adapters (zero-shot zs_whisper_ja_rp stays in)
PHASE0_EXPLORATORY = re.compile(r"_ep\d|^ev_ft_.*_ja_rp$")


def build_jobs(runs: Path) -> dict[str, Job]:
    jobs: dict[str, Job] = {}
    models = sorted({m for _, _, m, _ in SYSTEMS.values()})
    marker = runs / "prep_models" / "snapshots.json"
    code = ("import json,sys; from huggingface_hub import snapshot_download as s; from pathlib import Path; "
            f"d={{m: Path(s(m)).name for m in {models!r}}}; "
            f"Path({str(marker)!r}).write_text(json.dumps(d, indent=2)); print(d)")
    jobs["prep_models"] = Job("prep_models", [PY_MAIN, "-c", code], REPO, "cpu", outputs=[marker])
    for sysname, (py, runner, model, extra) in SYSTEMS.items():
        for test, (man, root) in TESTS.items():
            if test not in SYSTEM_TESTS.get(sysname, TESTS):
                continue
            name = f"lb_{sysname}__{test}"
            out = runs / name / "preds.jsonl"
            jobs[name] = Job(name, [py, runner, "--model-id", model, "--manifest", str(man),
                                    "--audio-root", str(root), "--out", str(out), *extra],
                             EVAL_DIR, "gpu", deps=["prep_models"], outputs=[out],
                             mem_gb=MEM_GB.get(sysname))
    return jobs


def score(runs_list: list[Path], out_dir: Path, lexicon: str, bootstrap: int) -> dict:
    """score.py per test set over every finished prediction file in runs_list.
    Phase 0 seeded systems contribute all seeds (score.py pools nothing; the
    leaderboard averages seeds below)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for test in TESTS:
        preds = []
        for runs in runs_list:
            preds += [p for p in sorted(runs.glob(f"*__{test}/preds.jsonl")) if (p.parent / "DONE").exists()
                      and not PHASE0_EXPLORATORY.search(p.parent.name.rsplit("__", 1)[0])]
        if not preds:
            continue
        labels = [p.parent.name.rsplit("__", 1)[0] for p in preds]
        js = out_dir / f"{test}.json"
        cmd = [PY_MAIN, "eval/score.py", *[f"{l}:{p}" for l, p in zip(labels, preds)],
               "--json", str(js), "--loanwords", lexicon, "--bootstrap", str(bootstrap)]
        with open(out_dir / f"{test}.txt", "w", encoding="utf-8") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=EVAL_DIR)
        payload = json.loads(js.read_text(encoding="utf-8"))
        results[test] = payload["models"] if isinstance(payload, dict) else payload
        print(f"[score] {test}: {len(preds)} prediction files")
    return results


def leaderboard(results: dict, out_dir: Path) -> None:
    """Mean over seeds per system; one row per system, sorted by CS-FLEURS MER."""
    import statistics
    table: dict = {}
    for test, rows in results.items():
        for r in rows:
            sysname = re.sub(r"_s\d+", "", r["model"])
            for m in LB_METRICS[test]:
                if isinstance(r.get(m), (int, float)):
                    table.setdefault(sysname, {}).setdefault(f"{test}:{m}", []).append(100 * r[m])
    cols = [f"{t}:{m}" for t in TESTS for m in LB_METRICS[t]]
    cols = [c for c in cols if any(c in v for v in table.values())]
    order = sorted(table, key=lambda s: statistics.fmean(table[s].get("csf_read:MER", [1e9])))
    lines = ["| system | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for s in order:
        cells = []
        for c in cols:
            v = table[s].get(c)
            cells.append("" if not v else f"{statistics.fmean(v):.1f}" + (f" ±{statistics.stdev(v):.1f}"
                                                                          if len(v) > 1 else ""))
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    (out_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "leaderboard.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/data/code-switching/runs/phase1_leaderboard")
    ap.add_argument("--phase0-runs", default="/data/code-switching/runs/phase0",
                    help="Also score the Phase 0 ladder's systems into the leaderboard")
    ap.add_argument("--lexicon", default="/data/code-switching/data/lexicon/loanwords_v1.json")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--eval-slots", type=int, default=3)
    ap.add_argument("--eval-mem-gb", type=float, default=12)
    ap.add_argument("--headroom-gb", type=float, default=8)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    jobs = select(build_jobs(runs), args.only)
    if args.dry_run:
        for j in jobs.values():
            print(f"{'DONE ' if is_done(j, runs) else '     '}{j.kind:4s} {j.name}")
        print(f"{len(jobs)} jobs")
        return 0
    if not args.score_only:
        from common.run_registry import write_run_record
        write_run_record(runs, vars(args), extra={"systems": {k: v[2] for k, v in SYSTEMS.items()}},
                         filename=f"leaderboard_{time.strftime('%Y%m%d_%H%M%S')}.json")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env.setdefault("HF_HOME", "/data/code-switching/hf_cache")
        env.setdefault("HF_TOKEN_PATH", os.path.expanduser("~/.cache/huggingface/token"))
        gib = lambda x: int(x * 1024)
        # models are fetched once by prep_models (online); every eval then runs offline
        run_jobs({"prep_models": jobs["prep_models"]}, runs, args.gpus.split(","), args.eval_slots,
                 1, env, {"gpu": gib(args.eval_mem_gb), "train": 0, "cpu": 0}, gib(args.headroom_gb))
        env["HF_HUB_OFFLINE"] = "1"
        run_jobs(jobs, runs, args.gpus.split(","), args.eval_slots, 1, env,
                 {"gpu": gib(args.eval_mem_gb), "train": 0, "cpu": 0}, gib(args.headroom_gb))
    runs_list = [runs] + ([Path(args.phase0_runs)] if args.phase0_runs else [])
    leaderboard(score(runs_list, runs / "scores", args.lexicon, args.bootstrap), runs / "scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
