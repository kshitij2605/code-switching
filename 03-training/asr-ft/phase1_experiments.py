#!/usr/bin/env python3
"""Phase 1 experiments v1 (pre-registered protocol), on the
shared memory-aware scheduler (common/jobs.py). Resumable.

  E1  Whisper-large-v3-turbo with the Phase 0 Whisper recipe, on {cs, csmono} x 3 seeds
      (product bake-off, step 1). Evaluated with auto language detection.
  E2  Whisper-large-v3 with labels without a language token (--label-language none),
      on {cs, csmono} x 3 seeds (Phase 4D). Evaluated with auto detection, plus forced
      ja on CS-FLEURS.
  E3  Qwen3-ASR-0.6B with its official full fine-tuning recipe (label masking fixed;
      train_qwen3_asr.py), on {cs, csmono} x 3 seeds (bake-off, step 2), in the
      qwen3-asr environment. Evaluated with automatic language identification.
  E4  Granite-4.0-1b-speech (train_granite_speech.py), on {cs, csmono} x 3 seeds
      (bake-off, step 3): a = projector only (IBM's recipe), b = projector + LLM LoRA r8.
  E5  LLM-only LoRA r8 (encoder and projector frozen) for Qwen3-ASR-0.6B and Granite,
      lr {1e-4, 2e-5}, csmono x 3 seeds (bake-off, step 4).

Every E1/E2 training command is the Phase 0 ladder's command for the same recipe and
seed, changed only by the experiment's factor (model id, or label format); E3 uses the
same data, dev sets and seeds with its own recipe. Test sets are the
ladder's: CS-FLEURS read/test, CS-YODAS jpn, FLEURS ja_jp / en_us test.

Usage (mactrn01, from the repo root or a pinned worktree, locked environment):
  PY=/data/code-switching/envs/liquid-ai-research/bin/python
  $PY 03-training/asr-ft/phase1_experiments.py --dry-run
  $PY 03-training/asr-ft/phase1_experiments.py --gpus 2,3
  $PY 03-training/asr-ft/phase1_experiments.py --score-only
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "04-evaluation" / "eval-ft"
TRAIN = REPO / "03-training" / "asr-ft"
sys.path.insert(0, str(REPO))
from common.jobs import Job, is_done, run_jobs, select  # noqa: E402

_spec = importlib.util.spec_from_file_location("phase0_ladder", TRAIN / "phase0_ladder.py")
P0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P0)

TURBO = "openai/whisper-large-v3-turbo"
QWEN06 = "Qwen/Qwen3-ASR-0.6B"
GRANITE = "ibm-granite/granite-4.0-1b-speech"
PY_QWEN = "/data/code-switching/envs/qwen3-asr/bin/python"   # pinned environment of the Qwen3-ASR runner
EXPERIMENTS = {          # name -> (model id, extra training args)
    "turbo": (TURBO, []),
    "nolang": (P0.WHISPER, ["--label-language", "none"]),
}
TESTS = ("csf_read", "csyodas", "mono_ja", "mono_en")


def build_jobs(py: str, data: Path, runs: Path) -> dict[str, Job]:
    tests = {
        "csf_read": (data / "eval-real/csfleurs/read_test/manifest.jsonl", data / "eval-real/csfleurs"),
        "csyodas": (data / "eval-real/csyodas/manifest.jsonl", data / "eval-real/csyodas"),
        "mono_ja": (data / "eval-mono/ja_jp_test/manifest.jsonl", data / "eval-mono"),
        "mono_en": (data / "eval-mono/en_us_test/manifest.jsonl", data / "eval-mono"),
    }
    dev = [data / "dev/csfleurs/xtts_test1/manifest.jsonl",
           data / "dev/fleurs/ja_jp_dev/manifest.jsonl",
           data / "dev/fleurs/en_us_dev/manifest.jsonl"]
    cs = [data / f"synth/{e}/manifest.jsonl" for e in ("edge_tts", "kokoro", "melo")] + \
         [data / "augmented_v2/manifest.jsonl"]
    mono = [data / "mono/mono_train.jsonl"]
    jobs: dict[str, Job] = {}

    def whisper_eval(name, model_id, test, extra, adapter, deps):
        m, root = tests[test]
        out = runs / name / "preds.jsonl"
        cmd = [py, "eval/run_whisper_hf.py", "--model-id", model_id, "--manifest", str(m),
               "--audio-root", str(root), "--out", str(out), *extra, "--lora-adapter", str(adapter)]
        jobs[name] = Job(name, cmd, EVAL_DIR, "gpu", deps=list(deps), outputs=[out])

    for seed in P0.SEEDS:
        for cfg, manifests in (("cs", cs), ("csmono", cs + mono)):
            for exp, (model_id, extra_train) in EXPERIMENTS.items():
                name = f"ft_{exp}_{cfg}_s{seed}"
                model = runs / name / "model"
                # the Phase 0 ladder's command (phase0_ladder.build_jobs), plus the factor
                cmd = [py, str(TRAIN / "train_whisper.py"), "--model-id", model_id,
                       "--manifests", *map(str, manifests), "--dev-manifests", *map(str, dev),
                       "--out", str(model), "--epochs", "3", "--batch-size", "16",
                       "--lr", "1e-4", "--lora-r", "8", "--seed", str(seed), *extra_train]
                jobs[name] = Job(name, cmd, REPO, "train",
                                 outputs=[model / "final/adapter_model.safetensors"])
                for test in TESTS:
                    whisper_eval(f"ev_{name}_auto__{test}", model_id, test, [], model / "final", [name])
                if exp == "nolang":
                    whisper_eval(f"ev_{name}_ja__csf_read", model_id, "csf_read",
                                 ["--language", "ja"], model / "final", [name])
            # E3: Qwen3-ASR-0.6B, its own (official, fixed) full fine-tuning recipe
            name = f"ft_qwen06_{cfg}_s{seed}"
            model = runs / name / "model"
            jobs[name] = Job(name, [PY_QWEN, str(TRAIN / "train_qwen3_asr.py"), "--model-id", QWEN06,
                                    "--manifests", *map(str, manifests), "--dev-manifests", *map(str, dev),
                                    "--out", str(model), "--epochs", "3", "--batch-size", "16",
                                    "--lr", "2e-5", "--seed", str(seed)],
                             REPO, "train", outputs=[model / "final/selection.json"])
            for test in TESTS:
                m, root = tests[test]
                out = runs / f"ev_{name}_auto__{test}" / "preds.jsonl"
                jobs[out.parent.name] = Job(out.parent.name, [
                    PY_QWEN, "eval/run_qwen3_asr.py", "--model-id", str(model / "final"),
                    "--manifest", str(m), "--audio-root", str(root), "--out", str(out)],
                    EVAL_DIR, "gpu", deps=[name], outputs=[out])
            # E4: Granite-4.0-1b-speech; a = projector only (IBM recipe), b = + LLM LoRA r8
            for var, extra_train in (("a", ["--lr", "3e-5"]), ("b", ["--lr", "1e-4", "--llm-lora-r", "8"])):
                name = f"ft_granite{var}_{cfg}_s{seed}"
                model = runs / name / "model"
                jobs[name] = Job(name, [py, str(TRAIN / "train_granite_speech.py"), "--model-id", GRANITE,
                                        "--manifests", *map(str, manifests), "--dev-manifests", *map(str, dev),
                                        "--out", str(model), "--epochs", "3", "--batch-size", "16",
                                        "--grad-accum", "2", "--seed", str(seed), *extra_train],
                                 REPO, "train", outputs=[model / "final/selection.json"])
                for test in TESTS:
                    m, root = tests[test]
                    out = runs / f"ev_{name}_auto__{test}" / "preds.jsonl"
                    jobs[out.parent.name] = Job(out.parent.name, [
                        py, "eval/run_audio_llm.py", "--family", "granite", "--model-id", str(model / "final"),
                        "--manifest", str(m), "--audio-root", str(root), "--out", str(out)],
                        EVAL_DIR, "gpu", deps=[name], outputs=[out])
            if cfg != "csmono":
                continue
            # E5: LoRA r8 on the LLM's q/v only (encoder and projector frozen), CS+mono only
            for tag, lr in (("1e4", "1e-4"), ("2e5", "2e-5")):
                for back, py_b, trainer, extra_train, runner in (
                        ("qwen06lora", PY_QWEN, "train_qwen3_asr.py",
                         ["--model-id", QWEN06, "--lora-r", "8"],
                         ["eval/run_qwen3_asr.py"]),
                        ("granitelora", py, "train_granite_speech.py",
                         ["--model-id", GRANITE, "--grad-accum", "2", "--llm-lora-r", "8", "--freeze-projector"],
                         ["eval/run_audio_llm.py", "--family", "granite"])):
                    name = f"ft_{back}{tag}_{cfg}_s{seed}"
                    model = runs / name / "model"
                    jobs[name] = Job(name, [py_b, str(TRAIN / trainer), *extra_train,
                                            "--manifests", *map(str, manifests), "--dev-manifests", *map(str, dev),
                                            "--out", str(model), "--epochs", "3", "--batch-size", "16",
                                            "--lr", lr, "--seed", str(seed)],
                                     REPO, "train", outputs=[model / "final/selection.json"])
                    for test in TESTS:
                        m, root = tests[test]
                        out = runs / f"ev_{name}_auto__{test}" / "preds.jsonl"
                        jobs[out.parent.name] = Job(out.parent.name, [
                            py_b, *runner, "--model-id", str(model / "final"),
                            "--manifest", str(m), "--audio-root", str(root), "--out", str(out)],
                            EVAL_DIR, "gpu", deps=[name], outputs=[out])
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/code-switching/data")
    ap.add_argument("--runs", default="/data/code-switching/runs/phase1_exp")
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--eval-slots", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--eval-mem-gb", type=float, default=10)
    ap.add_argument("--train-mem-gb", type=float, default=60)
    ap.add_argument("--headroom-gb", type=float, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    py = sys.executable
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    jobs = select(build_jobs(py, Path(args.data_root), runs), args.only)
    if args.dry_run:
        for j in jobs.values():
            print(f"{'DONE ' if is_done(j, runs) else '     '}{j.kind:5s} {j.name}")
        print(f"{len(jobs)} jobs")
        return 0
    if not args.score_only:
        from common.run_registry import write_run_record
        env = dict(os.environ)
        env.setdefault("HF_HOME", "/data/code-switching/hf_cache")
        env.setdefault("HF_TOKEN_PATH", os.path.expanduser("~/.cache/huggingface/token"))
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HOME"] = env["HF_HOME"]
        from huggingface_hub import snapshot_download
        snapshots = {m: Path(snapshot_download(m, local_files_only=True)).name for m in (P0.WHISPER, TURBO)}
        print(f"[exp] model snapshots: {snapshots}")
        write_run_record(runs, vars(args), extra={"model_snapshots": snapshots, "seeds": P0.SEEDS},
                         filename=f"experiments_{time.strftime('%Y%m%d_%H%M%S')}.json")
        gib = lambda x: int(x * 1024)
        run_jobs(jobs, runs, args.gpus.split(","), args.eval_slots, 1, env,
                 need_mib={"gpu": gib(args.eval_mem_gb), "train": gib(args.train_mem_gb), "cpu": 0},
                 headroom_mib=gib(args.headroom_gb))
    P0.summarize(P0.score_all(py, runs, args.bootstrap, ("csyodas",)), runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
