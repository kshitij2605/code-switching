"""Resumable, memory-aware job runner for GPU experiments on one host.

Extracted from 03-training/asr-ft/phase0_ladder.py so that every experiment
driver (the Phase 0 ladder, the Phase 1 leaderboard, later sweeps) shares one
scheduler:
  * a job is done once <runs>/<job>/DONE exists; reruns resume;
  * training jobs take a whole GPU, eval jobs take one of eval_slots per GPU,
    and waiting trainings drain GPUs;
  * a job starts only if total - (other processes' measured memory) - (our
    jobs, max of declared and measured) >= need + headroom, so the runner never
    pushes another user's job on a shared GPU into OOM.
"""
from __future__ import annotations

import fnmatch
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    name: str
    cmd: list[str]
    cwd: Path
    kind: str                       # cpu | gpu (one eval slot) | train (whole GPU)
    deps: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    mem_gb: float | None = None     # declared peak GPU memory; None = the kind's default


# ── scheduler ─────────────────────────────────────────────────────────────────

def is_done(job: Job, runs: Path) -> bool:
    return (runs / job.name / "DONE").exists()


def gpu_memory() -> tuple[dict[str, int], dict[int, tuple[str, int]]]:
    """Total MiB per GPU index, and {pid: (gpu index, used MiB)} for every compute
    process on the host, including other users' jobs, from nvidia-smi."""
    q = lambda args: subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True).stdout
    total, uuid2idx = {}, {}
    for line in q(["--query-gpu=index,uuid,memory.total"]).splitlines():
        idx, uuid, tot = (s.strip() for s in line.split(","))
        total[idx], uuid2idx[uuid] = int(tot), idx
    procs = {}
    for line in q(["--query-compute-apps=pid,gpu_uuid,used_memory"]).splitlines():
        if line.strip():
            pid, uuid, used = (s.strip() for s in line.split(","))
            if uuid in uuid2idx and used.isdigit():
                procs[int(pid)] = (uuid2idx[uuid], int(used))
    return total, procs


def select(jobs: dict[str, Job], patterns: list[str]) -> dict[str, Job]:
    """Jobs matching any pattern, plus everything they depend on."""
    if not patterns:
        return jobs
    keep: set[str] = set()
    stack = [n for n in jobs if any(fnmatch.fnmatch(n, p) for p in patterns)]
    while stack:
        n = stack.pop()
        if n not in keep:
            keep.add(n)
            stack += jobs[n].deps
    return {n: j for n, j in jobs.items() if n in keep}


def run_jobs(jobs: dict[str, Job], runs: Path, gpus: list[str], eval_slots: int,
             cpu_slots: int, env: dict, need_mib: dict[str, int], headroom_mib: int,
             allow_train: bool = True) -> set[str]:
    """need_mib: declared peak GPU memory per job kind ('gpu' eval/prep, 'train').
    A job starts on a GPU only if total - (other processes' measured use) -
    (this ladder's jobs, max of declared and measured) >= need + headroom, so the
    ladder never pushes another user's job on a shared GPU into OOM."""
    pending = [j for j in jobs.values() if not is_done(j, runs)]
    if not allow_train:
        held = {j.name for j in pending if j.kind == "train"}
        while True:     # also hold everything that (transitively) depends on a held job
            more = {j.name for j in pending if j.name not in held and any(d in held for d in j.deps)}
            if not more:
                break
            held |= more
        pending = [j for j in pending if j.name not in held]
        print(f"[ladder] --no-train: holding {len(held)} jobs (trainings and their evals)")
    print(f"[ladder] {len(jobs)} jobs, {len(jobs) - len(pending)} done or held, "
          f"{len(pending)} to run on GPUs {gpus}")
    free = {g: eval_slots for g in gpus}
    running: dict[str, tuple] = {}
    failed: set[str] = set()
    t0 = time.time()
    last_mem_note = 0.0

    def need_of(job: Job) -> int:
        return int(job.mem_gb * 1024) if job.mem_gb else need_mib[job.kind]

    def effective_free() -> dict[str, int]:
        total, procs = gpu_memory()
        mine = {r[0].pid: (r[2], need_of(r[1])) for r in running.values() if r[2] is not None}
        out = {}
        for g in gpus:
            other = sum(u for pid, (gi, u) in procs.items() if gi == g and pid not in mine)
            ours = sum(max(need, procs.get(pid, (g, 0))[1])
                       for pid, (gi, need) in mine.items() if gi == g)
            out[g] = total[g] - other - ours
        return out

    while pending or running:
        for name, (proc, job, gpu, units, logf, started) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            logf.close()
            del running[name]
            if gpu is not None:
                free[gpu] += units
            missing = [str(o) for o in job.outputs if not o.exists()]
            mins = (time.time() - started) / 60
            if rc == 0 and not missing:
                (runs / name / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
                print(f"[ladder] done   {name} ({mins:.1f} min)")
            else:
                failed.add(name)
                print(f"[ladder] FAILED {name} rc={rc} missing={missing[:2]} "
                      f"(see {runs / name / 'log.txt'})")
        for job in list(pending):
            if any(d in failed for d in job.deps):
                pending.remove(job)
                failed.add(job.name)
                print(f"[ladder] skip   {job.name}: a dependency failed")
        ready = [j for j in pending
                 if all(d not in jobs or is_done(jobs[d], runs) for d in j.deps)]
        # Training needs a whole GPU. While trainings wait, reserve (drain) that many
        # GPUs that are not training: no new evals go there until they are empty.
        training_gpus = {r[2] for r in running.values() if r[1].kind == "train"}
        n_wait = sum(1 for j in ready if j.kind == "train")
        others = sorted((g for g in gpus if g not in training_gpus), key=lambda g: -free[g])
        reserved = set(others[:n_wait])
        mem = effective_free() if any(j.kind != "cpu" for j in ready) else {}
        # long jobs first: cpu prep (gates training), training, then evals
        for job in sorted(ready, key=lambda j: {"cpu": 0, "train": 1, "gpu": 2}[j.kind]):
            gpu, units = None, 0
            if job.kind == "cpu":
                if sum(1 for r in running.values() if r[2] is None) >= cpu_slots:
                    continue
            elif job.kind == "train":
                units = eval_slots
                fits = [g for g in gpus if free[g] == eval_slots
                        and mem[g] >= need_of(job) + headroom_mib]
                if not fits:
                    continue
                gpu = fits[0]
                free[gpu] -= units
                reserved.discard(gpu)
            else:
                units = 1
                fits = [g for g in gpus if free[g] >= 1 and g not in reserved
                        and mem[g] >= need_of(job) + headroom_mib]
                if not fits:
                    if time.time() - last_mem_note > 600 and not running:
                        print(f"[ladder] waiting: effective free MiB {mem}")
                        last_mem_note = time.time()
                    continue
                gpu = min(fits, key=lambda g: free[g])   # pack evals onto busy GPUs
                free[gpu] -= units
            if gpu is not None:
                mem[gpu] -= need_of(job)
            jdir = runs / job.name
            jdir.mkdir(parents=True, exist_ok=True)
            (jdir / "cmd.txt").write_text(" ".join(job.cmd) + f"\n# cwd: {job.cwd}\n")
            logf = open(jdir / "log.txt", "w", encoding="utf-8")
            jenv = dict(env, CUDA_VISIBLE_DEVICES=gpu if gpu is not None else "")
            proc = subprocess.Popen(job.cmd, cwd=job.cwd, env=jenv, stdout=logf,
                                    stderr=subprocess.STDOUT)
            running[job.name] = (proc, job, gpu, units, logf, time.time())
            pending.remove(job)
            print(f"[ladder] start  {job.name} on {'cpu' if gpu is None else 'gpu ' + gpu}")
        time.sleep(5)
    print(f"[ladder] finished in {(time.time() - t0) / 3600:.2f} h; failed: {sorted(failed) or 'none'}")
    return failed
