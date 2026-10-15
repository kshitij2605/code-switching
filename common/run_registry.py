"""Reproducibility record written next to every training run.

    from common.run_registry import write_run_record
    write_run_record(out_dir, vars(args), inputs=args.manifests)

Writes ``<out_dir>/run_record.json`` with the git commit (and whether the tree
was dirty), the exact arguments, SHA-256 of every input file, the seed, host,
python / torch / transformers / peft / liquid-audio versions, and a timestamp.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version.split()[0]}
    for dist in ("torch", "torchaudio", "transformers", "peft", "accelerate", "datasets",
                 "liquid-audio", "numpy", "fugashi", "unidic-lite"):
        try:
            out[dist] = version(dist)       # distribution metadata: works without __version__
        except PackageNotFoundError:
            out[dist] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return out


def write_run_record(out_dir: str | os.PathLike, args: dict, inputs: list[str] | None = None,
                     extra: dict | None = None, filename: str = "run_record.json") -> Path:
    """Write ``<out_dir>/<filename>``. Data-building scripts that share an output
    directory pass their own filename (e.g. ``balanced_build.json``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = _git("status", "--porcelain")
    record = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(status) if status is not None else None},
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in args.items()
                 if not any(s in k.lower() for s in ("token", "key", "password", "secret"))},
        "inputs": {str(p): sha256_file(p) for p in (inputs or [])},
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": _versions(),
    }
    if extra:
        record["extra"] = extra
    path = out / filename
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run-registry] wrote {path} (commit {str(record['git']['commit'])[:10]}"
          f"{', DIRTY tree' if record['git']['dirty'] else ''})")
    return path


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA) RNGs."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
