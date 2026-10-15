"""
Dataset validation — checks audio files + text content before training.
Run from /data/lfm2.5/run-a with .venv active.
"""

import json, wave, struct, sys, io, re
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ISSUES = []
WARNINGS = []

def err(msg):   ISSUES.append(msg);   print(f"  ❌ {msg}")
def warn(msg):  WARNINGS.append(msg); print(f"  ⚠️  {msg}")
def ok(msg):    print(f"  ✅ {msg}")


# ── 1. Check WAV files ────────────────────────────────────────────────────────

def validate_wav(path: str, min_s=0.5, max_s=40.0) -> tuple[bool, float]:
    try:
        with wave.open(path, "rb") as wf:
            frames   = wf.getnframes()
            rate     = wf.getframerate()
            channels = wf.getnchannels()
            width    = wf.getsampwidth()
            duration = frames / rate
        if frames == 0:
            return False, 0.0
        if duration < min_s:
            return False, duration
        if duration > max_s:
            return False, duration
        return True, duration
    except Exception as e:
        return False, -1.0


def check_audio_files(folder: Path, prefix: str, expected: int):
    print(f"\n── {prefix} audio ({folder}) ──")
    files = sorted(folder.glob(f"{prefix}_*.wav"))
    print(f"  Files found: {len(files)} / expected {expected}")

    if len(files) != expected:
        err(f"{prefix}: expected {expected} files, found {len(files)}")

    bad, durations = [], []
    for f in files:
        ok_flag, dur = validate_wav(str(f))
        if not ok_flag:
            bad.append((f.name, dur))
            err(f"{f.name} — invalid/empty (duration={dur:.2f}s)")
        else:
            durations.append(dur)

    if durations:
        ok(f"Valid WAVs: {len(durations)}  |  "
           f"avg={sum(durations)/len(durations):.1f}s  "
           f"min={min(durations):.1f}s  max={max(durations):.1f}s")
    return len(bad)


# ── 2. Check text content ─────────────────────────────────────────────────────

# Characters we never want in model targets
BAD_CHARS = re.compile(r'[「」『』【】〔〕（）\(\)\[\]<>"​‌‍﻿]')
# Suspiciously short targets
MIN_TARGET_LEN = 3

def check_text(text: str, field: str, sample_id: str) -> list[str]:
    problems = []
    if not text or not text.strip():
        problems.append(f"{sample_id}.{field}: EMPTY")
    elif len(text.strip()) < MIN_TARGET_LEN:
        problems.append(f"{sample_id}.{field}: too short ({repr(text)})")
    m = BAD_CHARS.search(text)
    if m:
        problems.append(f"{sample_id}.{field}: bad char {repr(m.group())} in {repr(text[:60])}")
    if text != text.strip():
        problems.append(f"{sample_id}.{field}: leading/trailing whitespace")
    return problems


def check_metadata(meta_path: Path, dataset: str):
    print(f"\n── {dataset} metadata ({meta_path.name}) ──")
    if not meta_path.exists():
        err(f"{meta_path} not found"); return

    samples = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"  Samples: {len(samples)}")

    text_problems = []
    missing_audio = []
    for s in samples:
        sid = s.get("id", "?")
        af  = s.get("audio_file", "")
        if not Path(af).exists():
            missing_audio.append(af)

        # Check all text fields
        for field in ["full_transcription","all_japanese","all_english",
                      "japanese_transcription","english_transcription",
                      "japanese_translation","english_translation",
                      "japanese_parts","english_parts"]:
            if field in s:
                text_problems.extend(check_text(s[field], field, sid))

    if missing_audio:
        for f in missing_audio[:5]:
            err(f"Audio missing: {f}")
        if len(missing_audio) > 5:
            err(f"...and {len(missing_audio)-5} more missing audio files")
    else:
        ok("All audio files referenced in metadata exist")

    if text_problems:
        for p in text_problems[:10]:
            err(p)
        if len(text_problems) > 10:
            warn(f"...and {len(text_problems)-10} more text issues")
    else:
        ok("All text fields clean — no bad chars, no empty strings")


# ── 3. Check training JSONL ───────────────────────────────────────────────────

def check_jsonl(path: Path, label: str):
    print(f"\n── {label} ({path.name}) ──")
    if not path.exists():
        err(f"{path} not found"); return

    lines       = path.read_text(encoding="utf-8").strip().splitlines()
    total       = len(lines)
    parse_errs  = 0
    empty_target= 0
    missing_audio = 0
    bad_text    = 0
    type_counts = defaultdict(int)
    target_lens = []

    for i, line in enumerate(lines):
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            parse_errs += 1
            err(f"Line {i+1}: JSON parse error — {e}")
            continue

        # audio exists?
        if not Path(ex.get("audio_file","")).exists():
            missing_audio += 1

        # target ok?
        tgt = ex.get("target","")
        if not tgt or not tgt.strip():
            empty_target += 1
        else:
            target_lens.append(len(tgt))

        # bad chars in target?
        if BAD_CHARS.search(tgt):
            bad_text += 1

        type_counts[ex.get("type","?")] += 1

    print(f"  Total examples : {total}")
    if parse_errs:   err(f"JSON parse errors : {parse_errs}")
    else:            ok("All lines valid JSON")
    if missing_audio: err(f"Missing audio files: {missing_audio}")
    else:             ok("All audio files exist")
    if empty_target:  err(f"Empty targets      : {empty_target}")
    else:             ok("No empty targets")
    if bad_text:      err(f"Bad chars in target: {bad_text}")
    else:             ok("No bad characters in targets")

    if target_lens:
        ok(f"Target length — avg={sum(target_lens)/len(target_lens):.0f} "
           f"min={min(target_lens)} max={max(target_lens)} chars")

    print("  Task breakdown:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:<14}: {c}")


# ── 4. Run all checks ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Dataset Validation")
    print("=" * 60)

    # Audio files
    check_audio_files(Path("data/code_switching/audio"), "cs", 500)
    check_audio_files(Path("data/english_only/audio"),   "en", 300)
    check_audio_files(Path("data/japanese_only/audio"),  "ja", 300)

    # Metadata JSONs
    check_metadata(Path("data/code_switching/metadata/transcriptions.json"), "Code-Switching")
    check_metadata(Path("data/english_only/metadata/transcriptions.json"),   "English-Only")
    check_metadata(Path("data/japanese_only/metadata/transcriptions.json"),  "Japanese-Only")

    # Training JSONL
    check_jsonl(Path("data/training/train.jsonl"), "Train JSONL")
    check_jsonl(Path("data/training/eval.jsonl"),  "Eval JSONL")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    if ISSUES:
        print(f"\n❌ {len(ISSUES)} ERROR(S) found:")
        for i in ISSUES:
            print(f"   • {i}")
    else:
        print("\n✅ No errors found — dataset is clean and ready for training.")

    if WARNINGS:
        print(f"\n⚠️  {len(WARNINGS)} warning(s):")
        for w in WARNINGS:
            print(f"   • {w}")

if __name__ == "__main__":
    main()
