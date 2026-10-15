#!/usr/bin/env python3
"""Phase 0 baseline ladder: a resumable job
runner over the GPUs of one host.

Stages (every job writes into RUNS/<job>/ and is skipped once RUNS/<job>/DONE exists):

  prep   mono_train.jsonl with licence tags; the CS corpus augmented with the
         RIR pool (data/augmented_v2); LFM chat JSONL for train (CS+mono)
         and dev; one preprocessed LFM dataset shared by all LFM runs.
  zs     zero-shot, every test set:
           Whisper large-v3 x {auto, ja, ja_rp}, where ja_rp is forced-ja decoding
             with repetition penalty 1.3 + 3-gram block;
           LFM2.5-Audio-1.5B-JP x 3 system prompts.
  ft     3 seeds each:
           Whisper large-v3 LoRA r8 on {cs, cs+mono};
           LFM LoRA on cs+mono, placed on {lm, encoder, both}.
         The LFM recipe: r8 LM /
         r32 encoder, lr 3e-5 / 6e-6, batch 8, 2000 steps.
         Checkpoints are selected on the dev sets only: CS-FLEURS xtts/test1
         minus test overlap, plus FLEURS ja/en validation.
  eval   each selected checkpoint on every test set with the pre-registered
         decoding: greedy, no repetition penalty. Whisper uses auto language
         detection, plus forced ja on CS-FLEURS; LFM uses the "Perform ASR." prompt.
  score  score.py per test set with bootstrap CIs, then summary.json /
         summary.md with mean ± sd over seeds.

Exploratory additions (after registration; they only add evaluation jobs):
  --extra-test NAME=MANIFEST[@ROOT]   every system on a further test set
  --extra-decoding                    the Whisper adapters on CS-FLEURS with the
                                      ja_rp decoding
  --extra-epochs                      the Whisper trainings repeated, keeping every
                                      epoch checkpoint (selection sensitivity)

Training jobs take a whole GPU. Eval and preprocessing jobs take one of
--eval-slots slots on a GPU. CPU jobs run up to --cpu-slots at a time. A failed
job blocks only its dependants; rerun the same command to retry what is missing.

Usage (mactrn01, from the repo root, with the locked environment):
  PY=/data/code-switching/envs/liquid-ai-research/bin/python
  $PY 03-training/asr-ft/phase0_ladder.py --dry-run          # list the jobs
  $PY 03-training/asr-ft/phase0_ladder.py --gpus 0,1,2,3     # run everything
  $PY 03-training/asr-ft/phase0_ladder.py --only 'zs_*'      # fnmatch subset (+ its deps)
  $PY 03-training/asr-ft/phase0_ladder.py --score-only       # re-score what exists
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "04-evaluation" / "eval-ft"
PREP = REPO / "02-preprocessing" / "asr-ft"
TRAIN = REPO / "03-training" / "asr-ft"

sys.path.insert(0, str(REPO))
from common.jobs import Job, gpu_memory, is_done, run_jobs, select  # noqa: E402,F401

SEEDS = (1, 2, 3)
WHISPER = "openai/whisper-large-v3"
LFM = "LiquidAI/LFM2.5-Audio-1.5B-JP"
LFM_PROMPTS = {
    "p1": "Perform ASR.",
    "p2": "Transcribe the audio.",
    "p3": ("Transcribe the audio verbatim. Write English words in English (Latin script) "
           "and Japanese words in Japanese."),
}
WHISPER_ZS = {"auto": [], "ja": ["--language", "ja"],
              "ja_rp": ["--language", "ja", "--repetition-penalty", "1.3",
                        "--no-repeat-ngram-size", "3"]}


def build_jobs(py: str, data: Path, runs: Path,
               extra_tests: dict[str, tuple[Path, Path]] | None = None,
               extra_decoding: bool = False, extra_epochs: bool = False) -> dict[str, Job]:
    """extra_tests: exploratory (post-registration) sets, name -> (manifest, audio root);
    extra_decoding: exploratory csf_read decodes of the Whisper adapters with the
    ja_rp decoding;
    extra_epochs: exploratory repeats of the Whisper trainings that keep every epoch
    checkpoint, each evaluated on csf_read and the mono controls.
    All three only add jobs; the pre-registered jobs are unchanged."""
    tests = {
        "csf_read": (data / "eval-real/csfleurs/read_test/manifest.jsonl", data / "eval-real/csfleurs"),
        "mono_ja": (data / "eval-mono/ja_jp_test/manifest.jsonl", data / "eval-mono"),
        "mono_en": (data / "eval-mono/en_us_test/manifest.jsonl", data / "eval-mono"),
        "synth_art": (data / "eval-synth/csartificial_manifest.jsonl", Path("/")),
        "synth_diss": (data / "eval-diss-synth/manifest.jsonl", Path("/")),
    }
    for name, spec in (extra_tests or {}).items():
        if name in tests:
            raise SystemExit(f"--extra-test {name!r} clashes with a pre-registered test set")
        tests[name] = spec
    dev = [data / "dev/csfleurs/xtts_test1/manifest.jsonl",
           data / "dev/fleurs/ja_jp_dev/manifest.jsonl",
           data / "dev/fleurs/en_us_dev/manifest.jsonl"]
    cs = [data / f"synth/{e}/manifest.jsonl" for e in ("edge_tts", "kokoro", "melo")] + \
         [data / "augmented_v2/manifest.jsonl"]
    mono = [data / "mono/mono_train.jsonl"]
    jobs: dict[str, Job] = {}

    def add(j: Job) -> None:
        jobs[j.name] = j

    # ── prep ──────────────────────────────────────────────────────────────────
    add(Job("prep_mono", [py, str(PREP / "build_mono_train.py")], data.parent, "cpu",
            outputs=[data / "mono/mono_train.jsonl"]))
    add(Job("prep_aug", [py, str(PREP / "augment_multi.py"), "--manifests", *map(str, cs[:3]),
                         "--musan", str(data / "noise_sources/musan"),
                         "--rir", str(data / "noise_sources/RIRS_NOISES"),
                         "--out", str(data / "augmented_v2"), "--seed", "42"],
            REPO, "cpu", outputs=[data / "augmented_v2/manifest.jsonl"]))
    lfm_dir = runs / "prep_lfm"
    add(Job("prep_lfm_jsonl", ["bash", "-c",
            f"{py} {PREP / 'manifest_to_lfm_jsonl.py'} --out {lfm_dir / 'train.jsonl'} "
            + " ".join(map(str, cs + mono))
            + f" && {py} {PREP / 'manifest_to_lfm_jsonl.py'} --out {lfm_dir / 'dev.jsonl'} "
            + " ".join(map(str, dev))],
            REPO, "cpu", deps=["prep_mono", "prep_aug"],
            outputs=[lfm_dir / "train.jsonl", lfm_dir / "dev.jsonl"]))
    add(Job("prep_lfm_pre", [py, str(PREP / "preprocess_asr_data.py"),
                             "--model_id", LFM,
                             "--train_jsonl", str(lfm_dir / "train.jsonl"),
                             "--eval_jsonl", str(lfm_dir / "dev.jsonl"),
                             "--output_dir", str(lfm_dir / "preprocessed"), "--overwrite"],
            REPO, "gpu", deps=["prep_lfm_jsonl"],
            outputs=[lfm_dir / "preprocessed/train", lfm_dir / "preprocessed/eval"]))

    def whisper_eval(name, test, extra, adapter=None, deps=()):
        m, root = tests[test]
        out = runs / name / "preds.jsonl"
        cmd = [py, "eval/run_whisper_hf.py", "--model-id", WHISPER, "--manifest", str(m),
               "--audio-root", str(root), "--out", str(out), *extra]
        if adapter:
            cmd += ["--lora-adapter", str(adapter)]
        add(Job(name, cmd, EVAL_DIR, "gpu", deps=list(deps), outputs=[out]))

    def lfm_eval(name, test, prompt, adapter=None, deps=()):
        m, root = tests[test]
        out = runs / name / "preds.jsonl"
        cmd = [py, "eval/run_baseline.py", "--model", "lfm", "--model-id", LFM,
               "--manifest", str(m), "--audio-root", str(root), "--out", str(out),
               "--system-prompt", prompt]
        if adapter:
            cmd += ["--lora-adapter", str(adapter)]
        add(Job(name, cmd, EVAL_DIR, "gpu", deps=list(deps), outputs=[out]))

    # ── zero-shot ─────────────────────────────────────────────────────────────
    for test in tests:
        for cfg, extra in WHISPER_ZS.items():
            whisper_eval(f"zs_whisper_{cfg}__{test}", test, extra)
        for pid, prompt in LFM_PROMPTS.items():
            lfm_eval(f"zs_lfm_{pid}__{test}", test, prompt)

    # ── fine-tuning + evaluation ──────────────────────────────────────────────
    for seed in SEEDS:
        for cfg, manifests in (("cs", cs), ("csmono", cs + mono)):
            name = f"ft_whisper_{cfg}_s{seed}"
            model = runs / name / "model"
            add(Job(name, [py, str(TRAIN / "train_whisper.py"), "--model-id", WHISPER,
                           "--manifests", *map(str, manifests), "--dev-manifests", *map(str, dev),
                           "--out", str(model), "--epochs", "3", "--batch-size", "16",
                           "--lr", "1e-4", "--lora-r", "8", "--seed", str(seed)],
                    REPO, "train", deps=["prep_mono", "prep_aug"],
                    outputs=[model / "final/adapter_model.safetensors"]))
            for test in tests:
                whisper_eval(f"ev_{name}_auto__{test}", test, [], model / "final", [name])
            whisper_eval(f"ev_{name}_ja__csf_read", "csf_read", ["--language", "ja"],
                         model / "final", [name])
            if extra_decoding:
                whisper_eval(f"ev_{name}_ja_rp__csf_read", "csf_read", WHISPER_ZS["ja_rp"],
                             model / "final", [name])
            if extra_epochs:
                ep, ep_model = f"{name}_ep", runs / f"{name}_ep" / "model"
                add(Job(ep, [str(ep_model) if c == str(model) else c for c in jobs[name].cmd]
                        + ["--keep-checkpoints"], REPO, "train", deps=["prep_mono", "prep_aug"],
                        outputs=[ep_model / "epoch-3/adapter_model.safetensors"]))
                for k in (1, 2, 3):
                    for test in ("csf_read", "mono_ja", "mono_en"):
                        whisper_eval(f"ev_{name}_ep{k}_auto__{test}", test, [], ep_model / f"epoch-{k}", [ep])
        for scope in ("lm", "encoder", "both"):
            name = f"ft_lfm_{scope}_s{seed}"
            model = runs / name / "model"
            add(Job(name, [py, str(TRAIN / "train_lora.py"), "--model_id", LFM,
                           "--data", str(lfm_dir / "preprocessed/train"),
                           "--val_data", str(lfm_dir / "preprocessed/eval"),
                           "--output_dir", str(model), "--lora_scope", scope,
                           "--max_steps", "2000", "--warmup_steps", "100",
                           "--lr", "3e-5", "--encoder_lr", "6e-6",
                           "--lora_r", "8", "--encoder_lora_r", "32", "--batch_size", "8",
                           "--save_interval", "1000", "--val_interval", "200",
                           "--log_interval", "50", "--seed", str(seed)],
                    REPO, "train", deps=["prep_lfm_pre"],
                    outputs=[model / "best/adapter_model.safetensors"]))
            for test in tests:
                lfm_eval(f"ev_{name}__{test}", test, LFM_PROMPTS["p1"], model / "best", [name])
    return jobs


# ── scoring and summary ───────────────────────────────────────────────────────

TEST_ORDER = ("csf_read", "mono_ja", "mono_en", "synth_art", "synth_diss")
KEY_METRICS = {"csf_read": ("MER", "ScriptAcc", "ScriptAcc_strict", "KataKeep"),
               "synth_art": ("MER", "ScriptAcc", "KataKeep"),
               "synth_diss": ("MER", "ScriptAcc"),
               "mono_ja": ("JA_CER", "MER"), "mono_en": ("EN_WER", "MER")}
EXTRA_METRICS = ("MER", "ScriptAcc", "ScriptAcc_strict", "KataKeep")   # --extra-test sets (CS)


def score_all(py: str, runs: Path, bootstrap: int, extra: tuple[str, ...] = ()) -> dict:
    """score.py per test set over every prediction file that exists."""
    out_dir = runs / "scores"
    out_dir.mkdir(exist_ok=True)
    results = {}
    for test in TEST_ORDER + tuple(extra):
        preds = sorted(p for p in runs.glob(f"*__{test}/preds.jsonl")
                       if (p.parent / "DONE").exists())
        if not preds:
            continue
        labels = [p.parent.name.rsplit("__", 1)[0] for p in preds]
        js = out_dir / f"{test}.json"
        cmd = [py, str(EVAL_DIR / "eval/score.py"), *[f"{l}:{p}" for l, p in zip(labels, preds)],
               "--json", str(js), "--bootstrap", str(bootstrap)]
        with open(out_dir / f"{test}.txt", "w", encoding="utf-8") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=EVAL_DIR)
        payload = json.loads(js.read_text(encoding="utf-8"))
        results[test] = payload["models"] if isinstance(payload, dict) else payload
        print(f"[score] {test}: {len(preds)} systems -> {js}")
    return results


def summarize(results: dict, runs: Path) -> None:
    """Mean ± sd over seeds per configuration (label minus its _s<seed> part)."""
    table: dict = {}
    order = list(TEST_ORDER) + [t for t in results if t not in TEST_ORDER]
    for test, rows in results.items():
        for r in rows:
            cfg = re.sub(r"_s\d+", "", r["model"])
            for metric in KEY_METRICS.get(test, EXTRA_METRICS):
                v = r.get(metric)
                if isinstance(v, (int, float)):
                    table.setdefault(cfg, {}).setdefault(f"{test}:{metric}", []).append(100 * v)
    summary = {cfg: {k: {"mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
                         "n": len(v)} for k, v in cols.items()} for cfg, cols in table.items()}
    (runs / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    cols = [f"{t}:{m}" for t in order for m in KEY_METRICS.get(t, EXTRA_METRICS)]
    cols = [c for c in cols if any(c in v for v in summary.values())]
    lines = ["| config | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for cfg in sorted(summary):
        cells = []
        for c in cols:
            s = summary[cfg].get(c)
            cells.append("" if s is None else (f"{s['mean']:.1f}" if s["n"] == 1
                                               else f"{s['mean']:.1f} ± {s['sd']:.1f} (n={s['n']})"))
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    (runs / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/code-switching/data")
    ap.add_argument("--runs", default="/data/code-switching/runs/phase0")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--eval-slots", type=int, default=3, help="Concurrent eval jobs per GPU")
    ap.add_argument("--cpu-slots", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=[], help="fnmatch patterns of jobs to run")
    ap.add_argument("--no-train", action="store_true",
                    help="Hold all training jobs (and so their evals); run prep and zero-shot only")
    ap.add_argument("--extra-test", action="append", default=[], metavar="NAME=MANIFEST[@ROOT]",
                    help="Exploratory (post-registration) test set: evaluate every system on it. "
                         "ROOT (audio root) defaults to the manifest's directory")
    ap.add_argument("--extra-decoding", action="store_true",
                    help="Exploratory: also decode the Whisper adapters on csf_read with the "
                         "ja_rp decoding (forced ja + repetition penalty 1.3 + 3-gram block)")
    ap.add_argument("--extra-epochs", action="store_true",
                    help="Exploratory: repeat the Whisper trainings, keeping every epoch checkpoint, "
                         "and evaluate each on csf_read, mono_ja and mono_en")
    ap.add_argument("--eval-mem-gb", type=float, default=10,
                    help="Declared peak GPU memory of an eval/prep job")
    ap.add_argument("--train-mem-gb", type=float, default=60,
                    help="Declared peak GPU memory of a training job")
    ap.add_argument("--headroom-gb", type=float, default=8,
                    help="Free memory kept on every GPU for other users' jobs to grow into")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    py = sys.executable
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    extra = {}
    for spec in args.extra_test:
        name, _, rest = spec.partition("=")
        manifest, _, root = rest.partition("@")
        if not name or not manifest or not Path(manifest).exists():
            raise SystemExit(f"bad --extra-test {spec!r}: use NAME=MANIFEST[@ROOT] with an existing manifest")
        extra[name] = (Path(manifest), Path(root) if root else Path(manifest).parent)
    jobs = select(build_jobs(py, Path(args.data_root), runs, extra, args.extra_decoding,
                             args.extra_epochs), args.only)
    if args.dry_run:
        for j in jobs.values():
            print(f"{'DONE ' if is_done(j, runs) else '     '}{j.kind:5s} {j.name}  deps={j.deps}")
        print(f"{len(jobs)} jobs")
        return 0
    if not args.score_only:
        sys.path.insert(0, str(REPO))
        from common.run_registry import write_run_record
        env = dict(os.environ)
        env.setdefault("HF_HOME", "/data/code-switching/hf_cache")
        env.setdefault("HF_TOKEN_PATH", os.path.expanduser("~/.cache/huggingface/token"))
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_OFFLINE"] = "1"        # use the cached snapshots below, never a newer revision
        os.environ["HF_HOME"] = env["HF_HOME"]
        from huggingface_hub import snapshot_download
        snapshots = {m: Path(snapshot_download(m, local_files_only=True)).name for m in (WHISPER, LFM)}
        print(f"[ladder] model snapshots: {snapshots}")
        write_run_record(runs, vars(args), extra={"model_snapshots": snapshots, "seeds": SEEDS},
                         filename=f"ladder_{time.strftime('%Y%m%d_%H%M%S')}.json")
        gib = lambda x: int(x * 1024)
        run_jobs(jobs, runs, args.gpus.split(","), args.eval_slots, args.cpu_slots, env,
                 need_mib={"gpu": gib(args.eval_mem_gb), "train": gib(args.train_mem_gb), "cpu": 0},
                 headroom_mib=gib(args.headroom_gb), allow_train=not args.no_train)
    summarize(score_all(py, runs, args.bootstrap, tuple(extra)), runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
