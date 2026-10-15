#!/usr/bin/env python3
"""Convert FLEURS mono train manifests into the CS-corpus training format.

load_fleurs.py emits eval-contract rows ({id, reference, audio(rel), ...});
train_whisper.py expects {audio_path(abs), transcript}. This joins the ja_jp and
en_us train manifests into one mono replay manifest for the training mix.
Rows are licence-tagged (FLEURS, CC BY 4.0) and must pass the leak guard.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.leak_guard import LeakGuard  # noqa: E402
from common.licenses import tag_rows  # noqa: E402

ROOT = "data/mono"
out = []
for cfg in ("ja_jp_train", "en_us_train"):
    man = Path(ROOT) / cfg / "manifest.jsonl"
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append({
            "id": r["id"],
            "audio_path": os.path.abspath(os.path.join(ROOT, r["audio"])),
            "transcript": r["reference"],
            "lang": r.get("lang", cfg),
            "source": r.get("source", cfg),
        })
out = tag_rows(out, "google/fleurs")
LeakGuard().filter_rows(out, on_leak="raise", context="mono_train")
with open(f"{ROOT}/mono_train.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
print(f"mono_train: {len(out)} clips  by lang:",
      dict(Counter(r["lang"] for r in out)))
