"""
Converts generated audio + metadata into a flat JSONL training file.

For each audio sample, creates 3 training examples (one per task):
  - "Transcribe."             → output in native/detected language
  - "Transcribe in Japanese." → all output in Japanese
  - "Transcribe in English."  → all output in English

Output: data/training/train.jsonl, data/training/eval.jsonl
Each line:
  {"audio_file": "...", "system": "...", "target": "...", "type": "...", "group": "...",
   "license_source": "tts/edge-tts", ...}

Notes:
  * The 90/10 split is group-aware: all examples of a clip, and all clips
    that share a sentence, stay on one side (group = category + native
    transcript), and check_disjoint verifies train/eval share no group or audio.
  * Missing audio is an error (unless --allow-missing-audio).
  * Examples with a null target (e.g. a failed translation) are skipped and counted.
  * Rows are licence-tagged (edge-tts audio: research only) and must pass the leak guard.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.leak_guard import (LeakGuard, check_disjoint, group_split,  # noqa: E402
                               row_group_key, text_hash)
from common.licenses import tag_rows  # noqa: E402
from common.run_registry import write_run_record  # noqa: E402

SYS_NATIVE = "Transcribe the audio."
SYS_JA     = "Transcribe in Japanese."
SYS_EN     = "Transcribe in English."

# category dir -> (type prefix, native text key, [(system, target key, type suffix)])
CATEGORIES = {
    "code_switching": ("cs", "full_transcription",
                       [(SYS_NATIVE, "full_transcription", "native"),
                        (SYS_JA, "all_japanese", "ja"), (SYS_EN, "all_english", "en")]),
    "english_only":   ("en", "english_transcription",
                       [(SYS_NATIVE, "english_transcription", "native"),
                        (SYS_EN, "english_transcription", "en"), (SYS_JA, "japanese_translation", "ja")]),
    "japanese_only":  ("ja", "japanese_transcription",
                       [(SYS_NATIVE, "japanese_transcription", "native"),
                        (SYS_JA, "japanese_transcription", "ja"), (SYS_EN, "english_translation", "en")]),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="Holds <category>/metadata/transcriptions.json")
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--allow-missing-audio", action="store_true")
    return p.parse_args()


def load_category(data_dir: Path, category: str, allow_missing: bool, stats: Counter) -> list[dict]:
    prefix, native_key, tasks = CATEGORIES[category]
    path = data_dir / category / "metadata" / "transcriptions.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found yet")
        return []
    samples = json.loads(path.read_text(encoding="utf-8"))
    examples, missing = [], []
    for s in samples:
        af = s["audio_file"]
        if not Path(af).exists():
            missing.append(af)
            continue
        group = f"{prefix}:{text_hash(s.get(native_key)) or af}"
        for system, key, suffix in tasks:
            if not s.get(key):
                stats[f"{prefix}_{suffix}: null target skipped"] += 1
                continue
            examples.append({"id": f"{s.get('id', Path(af).stem)}_{suffix}", "audio_file": af,
                             "system": system, "target": s[key], "type": f"{prefix}_{suffix}",
                             "group": group})
    if missing:
        msg = f"{category}: {len(missing)} audio files missing (e.g. {missing[:3]})"
        if not allow_missing:
            sys.exit(msg + "; regenerate them or pass --allow-missing-audio")
        print(f"  WARNING: {msg} -> skipped")
    print(f"  {prefix.upper()} : {len(samples) - len(missing)} audio → {len(examples)} examples")
    return examples


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = data_dir / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    stats: Counter = Counter()
    all_examples = []
    for cat in CATEGORIES:
        all_examples += load_category(data_dir, cat, args.allow_missing_audio, stats)
    if not all_examples:
        print("No examples found. Run data generation first.")
        return
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # generate_{cs,en,ja}_data.py synthesise with edge-tts -> research-only audio
    all_examples = tag_rows(all_examples, "tts/edge-tts")
    LeakGuard().filter_rows(all_examples, on_leak="raise", context="prepare_training_data")
    train, eval_ = group_split(all_examples, args.eval_ratio, rng, row_group_key)
    check_disjoint(train, eval_, context="prepare_training_data")
    rng.shuffle(train)

    # ── Write JSONL ──────────────────────────────────────────────────────────
    train_path = out_dir / "train.jsonl"
    eval_path  = out_dir / "eval.jsonl"
    for path, rows in ((train_path, train), (eval_path, eval_)):
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in rows),
                        encoding="utf-8")
    write_run_record(out_dir, vars(args), inputs=[str(train_path), str(eval_path)],
                     extra={"train": len(train), "eval": len(eval_), "skipped": dict(stats)},
                     filename="prepare_training_data.build.json")

    # ── Stats ────────────────────────────────────────────────────────────────
    print(f"\nTrain: {len(train)} examples → {train_path}")
    print(f"Eval : {len(eval_)}  examples → {eval_path}")
    print("\nTrain type breakdown:")
    for t, c in sorted(Counter(e["type"] for e in train).items()):
        print(f"  {t:<12}: {c}")

    # ── Sample preview ───────────────────────────────────────────────────────
    print("\nSample training examples:")
    for ex in rng.sample(train, min(3, len(train))):
        print(f"\n  audio : {ex['audio_file']}")
        print(f"  system: {ex['system']}")
        print(f"  target: {ex['target'][:80]}{'...' if len(ex['target'])>80 else ''}")
        print(f"  type  : {ex['type']}")


if __name__ == "__main__":
    main()
