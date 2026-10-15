#!/usr/bin/env python3
"""Leaderboard backend: metrics, checkpoint discovery, evaluation, persistence.

Pure-Python (no Streamlit) so it can be unit-run and reused. The Streamlit UI
(`eval/leaderboard_app.py`) is a thin layer over these functions.

The leaderboard scores models on the 3 frozen subsets (see data/eval/README.md):
CS (ScriptAcc/MER), JA-only (JA-CER), EN-only (EN-WER). New LFM LoRA checkpoints
are evaluated with the STANDARD prompt "Perform ASR." (default).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use the interpreter that's running this process (the active venv) for subprocess
# eval runs — portable, no hardcoded venv path.
PY = sys.executable
# Where fine-tuned LoRA checkpoints live. Override with the CKPT_ROOT env var:
#   CKPT_ROOT=/path/to/checkpoints streamlit run eval/leaderboard_app.py
CKPT_ROOT = os.environ.get("CKPT_ROOT", "checkpoints")
LEADERBOARD_PATH = os.path.join(REPO, "data", "eval", "leaderboard.json")
PRED_DIR = os.path.join(REPO, "artifacts", "preds", "leaderboard")
DEFAULT_PROMPT = "Perform ASR."

sys.path.insert(0, os.path.join(REPO, "eval"))
import score as S  # noqa: E402

# The 3 frozen benchmark subsets.
BENCH = {
    "cs": {"label": "Code-switching", "manifest": "data/eval/csfleurs_jaen_read196.jsonl",
           "audio_root": "data/csfleurs", "headline": "script_acc"},
    "ja": {"label": "Japanese-only", "manifest": "data/eval/fleurs_ja_jp_test200.jsonl",
           "audio_root": "data/fleurs", "headline": "ja_cer"},
    "en": {"label": "English-only", "manifest": "data/eval/fleurs_en_us_test200.jsonl",
           "audio_root": "data/fleurs", "headline": "en_wer"},
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _wer(ec) -> float | None:
    return round(100 * (ec.sub + ec.dele + ec.ins) / ec.ref_len, 1) if ec.ref_len else None


def _ec(ec) -> dict:
    return {"sub": ec.sub, "del": ec.dele, "ins": ec.ins, "ref": ec.ref_len}


def metrics_from_preds(path: str) -> dict:
    """Score one predictions JSONL -> full metric set incl. JA-WER + decomposition."""
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    m = S.score_records(records, "x")
    sc = m.script
    kc = m.kata
    return {
        "n": m.n,
        "mer": _wer(m.mer),
        "ja_cer": _wer(m.ja),
        "ja_wer": _wer(m.jaw),       # NEW: Japanese word-level WER (fugashi)
        "en_wer": _wer(m.en),
        "script_acc": round(100 * sc.latin / sc.en_ref, 1) if sc.en_ref else None,
        # scorer v2: exact-Latin ScriptAcc and katakana keep-rate (two-sided script metric)
        "script_acc_strict": round(100 * sc.latin_exact / sc.en_ref, 1) if sc.en_ref else None,
        "kata_keep": round(100 * kc.kept / kc.runs, 1) if kc.runs else None,
        # per-metric error decomposition (sub/del/ins/ref) for UI drill-down
        "edits": {"mer": _ec(m.mer), "ja_cer": _ec(m.ja), "ja_wer": _ec(m.jaw), "en_wer": _ec(m.en)},
        "script_bd": {"latin": sc.latin, "latin_exact": sc.latin_exact,
                      "latin_other": sc.latin_other, "katakana": sc.katakana,
                      "other_jp": sc.other_jp, "dropped": sc.dropped, "en_ref": sc.en_ref},
        "kata_bd": {"runs": kc.runs, "kept": kc.kept, "latinized": kc.latinized,
                    "other": kc.other},
        "flags": m.flags(),
        "scorer_version": S.SCORER_VERSION,
    }


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #
def list_lora_checkpoints(root: str = CKPT_ROOT) -> list[str]:
    """All LoRA-adapter dirs (containing adapter_config.json), as relative names."""
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "adapter_config.json" in files:
            found.append(dirpath)
    # sort by run name, then numeric step (step_1000 after step_200; 'final' last)
    def keyf(p):
        rel = os.path.relpath(p, root)
        run = os.path.dirname(rel)
        leaf = os.path.basename(rel)
        step = int(leaf.split("_")[1]) if leaf.startswith("step_") else 10**9
        return (run, step, leaf)
    return sorted(found, key=keyf)


def ckpt_display_name(path: str, root: str = CKPT_ROOT) -> str:
    return os.path.relpath(path, root)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_lfm_checkpoint(adapter_path: str, prompt: str = DEFAULT_PROMPT,
                            limit: int | None = None, gpu: str = "2",
                            progress=lambda *_: None) -> dict:
    """Run the LFM LoRA checkpoint over all 3 subsets and return per-subset metrics.

    `progress(stage, line)` is called with streamed log lines for UI display.
    Raises RuntimeError if any subset run fails.
    """
    os.makedirs(PRED_DIR, exist_ok=True)
    safe = ckpt_display_name(adapter_path).replace("/", "__")
    results = {}
    for key, sub in BENCH.items():
        out = os.path.join(PRED_DIR, f"{safe}__{key}.jsonl")
        cmd = [PY, "eval/run_baseline.py", "--model", "lfm",
               "--lora-adapter", adapter_path, "--system-prompt", prompt,
               "--manifest", sub["manifest"], "--audio-root", sub["audio_root"],
               "--out", out]
        if limit:
            cmd += ["--limit", str(limit)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        progress(key, f"$ {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in proc.stdout:  # stream
            tail.append(line.rstrip())
            tail = tail[-400:]
            progress(key, line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"[{key}] eval failed (exit {proc.returncode}):\n" + "\n".join(tail[-30:]))
        results[key] = metrics_from_preds(out)
        progress(key, f"  -> {results[key]}")
    return results


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_leaderboard() -> list[dict]:
    if os.path.exists(LEADERBOARD_PATH):
        return json.load(open(LEADERBOARD_PATH, encoding="utf-8"))
    return []


def save_leaderboard(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def upsert_entry(entry: dict) -> list[dict]:
    """Add/replace an entry keyed by name; persist; return the new leaderboard."""
    entries = [e for e in load_leaderboard() if e.get("name") != entry["name"]]
    entries.append(entry)
    save_leaderboard(entries)
    return entries


# Reference rows scored from the canonical prediction files (run once to seed).
_SEED = [
    ("LFM base", "lfm-base", "Perform ASR.",
     {"cs": "lfm_read196", "ja": "lfm_base_performasr_ja_jp", "en": "lfm_base_performasr_en_us"}),
    ("LFM + LoRA step_1000", "lfm-lora", "Transcribe the audio.",
     {"cs": "lfm_lora1000_correctprompt_read196", "ja": "lfm_lora1000_ja_jp", "en": "lfm_lora1000_en_us"}),
    ("Whisper large-v3 base", "whisper", "—",
     {"cs": "whisper_base_tf_read196", "ja": "whisper_base_tf_ja_jp", "en": "whisper_base_tf_en_us"}),
    ("Whisper large-v3 fine-tuned", "whisper", "—",
     {"cs": "whisper_ft_read196", "ja": "whisper_ft_ja_jp", "en": "whisper_ft_en_us"}),
]


def _pred_files_for(entry: dict) -> dict | None:
    """Map a leaderboard entry to its 3 prediction files (for re-scoring)."""
    if entry.get("checkpoint", "(reference)") == "(reference)":
        for name, _kind, _prompt, files in _SEED:
            if name == entry["name"]:
                base = os.path.join(REPO, "artifacts", "preds")
                return {sub: os.path.join(base, stem + ".jsonl") for sub, stem in files.items()}
        return None
    safe = ckpt_display_name(entry["checkpoint"]).replace("/", "__")
    return {sub: os.path.join(PRED_DIR, f"{safe}__{sub}.jsonl") for sub in ("cs", "ja", "en")}


def recompute_all() -> list[dict]:
    """Re-score every entry from its prediction files (adds new metrics like JA-WER
    to entries scored before those metrics existed). Skips entries whose preds are gone."""
    out = []
    for e in load_leaderboard():
        files = _pred_files_for(e)
        if files and all(os.path.exists(p) for p in files.values()):
            for sub, p in files.items():
                e[sub] = metrics_from_preds(p)
        out.append(e)
    save_leaderboard(out)
    return out


def needs_recompute(entries: list[dict]) -> bool:
    """True if any entry lacks the newer metrics (e.g. ja_wer / edits)."""
    return any("edits" not in e.get("cs", {}) or e.get("ja", {}).get("ja_wer") is None
               for e in entries)


def seed_leaderboard(force: bool = False) -> list[dict]:
    """Populate the leaderboard from existing canonical pred files (reference rows)."""
    if os.path.exists(LEADERBOARD_PATH) and not force:
        return load_leaderboard()
    preds = os.path.join(REPO, "artifacts", "preds")
    entries = []
    for name, kind, prompt, files in _SEED:
        per = {}
        ok = True
        for sub, stem in files.items():
            p = os.path.join(preds, stem + ".jsonl")
            if not os.path.exists(p):
                ok = False
                break
            per[sub] = metrics_from_preds(p)
        if ok:
            entries.append({"name": name, "kind": kind, "checkpoint": "(reference)",
                            "prompt": prompt, "legacy": kind == "lfm-lora", "ts": None, **per})
    save_leaderboard(entries)
    return entries


if __name__ == "__main__":
    # CLI: seed and print the leaderboard.
    ents = seed_leaderboard(force="--force" in sys.argv)
    print(f"leaderboard: {len(ents)} entries -> {LEADERBOARD_PATH}")
    for e in ents:
        print(f"  {e['name']:32s} CS-SA {e['cs']['script_acc']} | JA-CER {e['ja']['ja_cer']} | EN-WER {e['en']['en_wer']}")
