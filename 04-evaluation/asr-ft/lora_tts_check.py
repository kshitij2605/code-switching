"""
Verify LoRA adapter task conditioning on the 7 synthetic training-domain TTS samples.
Loads base model + LoRA adapter and runs all 3 system prompts.

Usage:
  python lora_tts_check.py --adapter /data/lfm2.5/run-a/checkpoints/lfm_lora/step_1000
"""
from __future__ import annotations

import argparse
import os

import torch
import soundfile as sf
from huggingface_hub import login

BASE_MODEL   = "LiquidAI/LFM2.5-Audio-1.5B-JP"
ROOT         = "/data/lfm2.5/run-a"
IM_END_TOKEN = 7
MAX_NEW_TOKENS = 150

PROMPTS = [
    ("Native",   "Transcribe the audio."),
    ("Force-JP", "Transcribe in Japanese."),
    ("Force-EN", "Transcribe in English."),
]

SAMPLES = [
    dict(type="CS", id="cs_001", audio="data/code_switching/audio/cs_001.wav",
         native="絶対 この shirt かなり cool と思わない 値段は reasonable だよ よろしく",
         jp="絶対 この シャツ かなり かっこいい と思わない 値段は 手頃 だよ よろしく",
         en="You absolutely have to agree that this shirt is pretty cool, and the price is reasonable too."),
    dict(type="CS", id="cs_002", audio="data/code_switching/audio/cs_002.wav",
         native="明日の schedule を確認して right? それと the review も終わらせないと だね",
         jp="明日の スケジュール を確認して そうでしょ? それと レビュー も終わらせないと だね",
         en="Let's confirm tomorrow's schedule, right? And we also need to finish the review."),
    dict(type="CS", id="cs_003", audio="data/code_switching/audio/cs_003.wav",
         native="その 明日の presentation の準備できた for sure それと the budget も確認したほうがいい だよね",
         jp="その 明日の プレゼン の準備できた 確かに それと 予算 も確認したほうがいい だよね",
         en="You've definitely prepared for tomorrow's presentation, and you should also check the budget, right?"),
    dict(type="EN", id="en_001", audio="data/english_only/audio/en_001.wav",
         native="I need to schedule a checkup with my doctor.",
         jp="医者と検診の予約をしなければなりません。",
         en="I need to schedule a checkup with my doctor."),
    dict(type="EN", id="en_002", audio="data/english_only/audio/en_002.wav",
         native="Let's review the agenda before the meeting starts.",
         jp="会議が始まる前に議題を確認しましょう。",
         en="Let's review the agenda before the meeting starts."),
    dict(type="JA", id="ja_001", audio="data/japanese_only/audio/ja_001.wav",
         native="ピーク時のためにサーバーの容量を増やす必要があります。",
         jp="ピーク時のためにサーバーの容量を増やす必要があります。",
         en="We need to increase the server capacity for peak hours."),
    dict(type="JA", id="ja_002", audio="data/japanese_only/audio/ja_002.wav",
         native="このプロジェクトの締め切りは来週の金曜日です。",
         jp="このプロジェクトの締め切りは来週の金曜日です。",
         en="The deadline for this project is next Friday."),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",  required=True)
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    return p.parse_args()


def load_audio(path):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return torch.tensor(arr, dtype=torch.float).unsqueeze(0), sr


def infer(model, proc, wave, sr, prompt):
    from liquid_audio import ChatState
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(prompt); chat.end_turn()
    chat.new_turn("user");   chat.add_audio(wave, sr); chat.end_turn()
    chat.new_turn("assistant")
    toks = []
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=MAX_NEW_TOKENS):
            if t.numel() == 1:
                tid = int(t.reshape(-1)[0].item())
                if tid == IM_END_TOKEN:
                    break
                toks.append(tid)
    return proc.text.decode(toks, skip_special_tokens=True).strip() if toks else "(empty)"


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
    print(f"  adapter loaded from: {args.adapter}\n")

    SEP = "=" * 110
    hits = {label: 0 for label, _ in PROMPTS}
    total = 0

    for s in SAMPLES:
        apath = os.path.join(ROOT, s["audio"])
        if not os.path.exists(apath):
            print(f"SKIP {s['id']} — not found: {apath}")
            continue

        wave, sr = load_audio(apath)
        total += 1
        print(SEP)
        print(f"[{s['type']}] {s['id']}")

        for label, prompt in PROMPTS:
            ref = {"Native": s["native"], "Force-JP": s["jp"], "Force-EN": s["en"]}[label]
            hyp = infer(model, proc, wave, sr, prompt)
            match = "✓" if hyp.strip() == ref.strip() else "✗"
            if hyp.strip() == ref.strip():
                hits[label] += 1
            print(f"\n  [{label}] {match}  prompt: \"{prompt}\"")
            print(f"  REF: {ref}")
            print(f"  HYP: {hyp}")

    print(f"\n{SEP}")
    print("SUMMARY (exact match on training-domain TTS samples):")
    for label, _ in PROMPTS:
        print(f"  {label:10s}: {hits[label]}/{total}")
    print(SEP)


if __name__ == "__main__":
    main()
