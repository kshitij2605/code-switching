"""
Evaluate LoRA adapter on the full synthetic training dataset, task-matched:
  - CS  audio  + "Transcribe the audio."     vs full_transcription
  - EN  audio  + "Transcribe in English."    vs english_transcription
  - JA  audio  + "Transcribe in Japanese."   vs japanese_transcription

Usage:
  python -u eval_synthetic.py \
    --adapter /data/lfm2.5/run-a/checkpoints/lfm_lora/step_1000 \
    --data-root /data/lfm2.5/run-a \
    --hf_token hf_xxx
"""
from __future__ import annotations

import argparse
import json
import os
import time
import unicodedata

import torch
import soundfile as sf
from huggingface_hub import login
from jiwer import wer

BASE_MODEL   = "LiquidAI/LFM2.5-Audio-1.5B-JP"
IM_END_TOKEN = 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",    required=True)
    p.add_argument("--data-root",  default="/data/lfm2.5/run-a")
    p.add_argument("--hf_token",   default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--task",       choices=["cs", "en", "ja", "all"], default="all")
    p.add_argument("--limit",      type=int, default=None, help="Cap per category (for quick runs)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    return p.parse_args()


def load_audio(path: str):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return torch.tensor(arr, dtype=torch.float).unsqueeze(0), sr


def infer(model, proc, wave, sr, prompt: str, max_new_tokens: int) -> str:
    from liquid_audio import ChatState
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(prompt); chat.end_turn()
    chat.new_turn("user");   chat.add_audio(wave, sr); chat.end_turn()
    chat.new_turn("assistant")
    toks = []
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=max_new_tokens):
            if t.numel() == 1:
                tid = int(t.reshape(-1)[0].item())
                if tid == IM_END_TOKEN:
                    break
                toks.append(tid)
    return proc.text.decode(toks, skip_special_tokens=True).strip() if toks else ""


def cer(ref: str, hyp: str) -> float:
    """Character error rate (works for Japanese/CJK)."""
    ref_chars = list(unicodedata.normalize("NFC", ref))
    hyp_chars = list(unicodedata.normalize("NFC", hyp))
    # simple edit distance
    r, h = len(ref_chars), len(hyp_chars)
    dp = list(range(h + 1))
    for i in range(1, r + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, h + 1):
            temp = dp[j]
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[h] / max(r, 1)


def evaluate_category(model, proc, items, audio_root, prompt, ref_key, label, args):
    print(f"\n{'='*70}")
    print(f"Task: {label}  |  prompt: \"{prompt}\"  |  N={len(items)}")
    print(f"{'='*70}")

    total_wer, total_cer, n = 0.0, 0.0, 0
    t0 = time.time()

    for i, item in enumerate(items):
        apath = os.path.join(audio_root, item["audio_file"])
        if not os.path.exists(apath):
            continue

        ref = item[ref_key].strip()
        wave, sr = load_audio(apath)
        hyp = infer(model, proc, wave, sr, prompt, args.max_new_tokens)

        w = wer(ref, hyp)
        c = cer(ref, hyp)
        total_wer += w
        total_cer += c
        n += 1

        if i < 3 or i % 50 == 0:
            print(f"\n  [{i+1:3d}/{len(items)}] {item['id']}")
            print(f"  REF: {ref[:100]}")
            print(f"  HYP: {hyp[:100]}")
            print(f"  WER={w:.1%}  CER={c:.1%}")

    dt = time.time() - t0
    avg_wer = total_wer / max(n, 1)
    avg_cer = total_cer / max(n, 1)
    print(f"\n  RESULT  N={n}  avg-WER={avg_wer:.1%}  avg-CER={avg_cer:.1%}  ({dt:.0f}s)")
    return avg_wer, avg_cer, n


def main():
    args = parse_args()
    if args.hf_token:
        login(token=args.hf_token)

    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    from peft import PeftModel

    print("Loading base model + LoRA adapter ...")
    proc  = LFM2AudioProcessor.from_pretrained(BASE_MODEL)
    base  = LFM2AudioModel.from_pretrained(BASE_MODEL, device="cuda", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    print(f"  ready.\n")

    root = args.data_root

    def load_meta(cat):
        with open(f"{root}/data/{cat}/metadata/transcriptions.json") as f:
            items = json.load(f)
        if args.limit:
            items = items[:args.limit]
        return items

    results = {}
    run_tasks = ["cs", "en", "ja"] if args.task == "all" else [args.task]

    if "cs" in run_tasks:
        cs_items = load_meta("code_switching")
        w, c, n = evaluate_category(
            model, proc, cs_items, root,
            prompt="Transcribe the audio.",
            ref_key="full_transcription",
            label="CS audio → native (code-switching preserved)",
            args=args,
        )
        results["CS-native"] = dict(wer=w, cer=c, n=n)

    if "en" in run_tasks:
        en_items = load_meta("english_only")
        w, c, n = evaluate_category(
            model, proc, en_items, root,
            prompt="Transcribe in English.",
            ref_key="english_transcription",
            label="EN audio → force English",
            args=args,
        )
        results["EN-force_en"] = dict(wer=w, cer=c, n=n)

    if "ja" in run_tasks:
        ja_items = load_meta("japanese_only")
        w, c, n = evaluate_category(
            model, proc, ja_items, root,
            prompt="Transcribe in Japanese.",
            ref_key="japanese_transcription",
            label="JA audio → force Japanese",
            args=args,
        )
        results["JA-force_jp"] = dict(wer=w, cer=c, n=n)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — LoRA adapter on synthetic dataset")
    print(f"{'='*70}")
    print(f"  {'Task':<35} {'N':>5}  {'WER':>8}  {'CER':>8}")
    print(f"  {'-'*35}  {'-'*5}  {'-'*8}  {'-'*8}")
    for tag, r in results.items():
        print(f"  {tag:<35} {r['n']:>5}  {r['wer']:>7.1%}  {r['cer']:>7.1%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
