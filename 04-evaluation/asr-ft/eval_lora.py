"""
Evaluate LoRA-fine-tuned LFM2.5-Audio-1.5B-JP on CS-FLEURS 196-utterance benchmark.

Loads base model + LoRA adapter (PEFT), runs all three task prompts.

Usage:
  python eval_lora.py \
    --adapter  /data/lfm2.5/run-a/checkpoints/lfm_lora/step_1000 \
    --manifest /data/lfm2.5/run-b/liquid-ai/data/csfleurs/read_test/manifest.jsonl \
    --audio-root /data/lfm2.5/run-b/liquid-ai/data/csfleurs \
    --out-dir  /data/lfm2.5/run-a/artifacts/lora_eval \
    --hf_token hf_xxx
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import soundfile as sf
from huggingface_hub import login

BASE_MODEL   = "LiquidAI/LFM2.5-Audio-1.5B-JP"
IM_END_TOKEN = 7

PROMPTS = [
    ("native",    "Transcribe the audio."),
    ("force_jp",  "Transcribe in Japanese."),
    ("force_en",  "Transcribe in English."),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",      required=True, help="Path to LoRA adapter dir")
    p.add_argument("--manifest",     required=True)
    p.add_argument("--audio-root",   default=".")
    p.add_argument("--out-dir",      required=True)
    p.add_argument("--hf_token",     default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--limit",        type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def load_audio_mono(path: str):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def infer(model, proc, wave, sr, system_prompt: str, max_new_tokens: int) -> str:
    from liquid_audio import ChatState
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(system_prompt); chat.end_turn()
    chat.new_turn("user");   chat.add_audio(wave, sr);     chat.end_turn()
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


def main():
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_audio_path"] = os.path.join(args.audio_root, r["audio"])
            rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"Loaded {len(rows)} manifest rows")

    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    from peft import PeftModel

    print(f"Loading base model: {BASE_MODEL}")
    proc  = LFM2AudioProcessor.from_pretrained(BASE_MODEL)
    base  = LFM2AudioModel.from_pretrained(BASE_MODEL, device="cuda", dtype=torch.bfloat16)

    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    print("  Model ready.")

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    for tag, prompt in PROMPTS:
        out_path = os.path.join(args.out_dir, f"lora_{tag}.jsonl")
        print(f"\n{'='*70}")
        print(f"Prompt [{tag}]: '{prompt}'")
        print(f"{'='*70}")
        preds = []

        for i, r in enumerate(rows):
            arr, sr = load_audio_mono(r["_audio_path"])
            wave    = torch.tensor(arr, dtype=torch.float).unsqueeze(0)
            hyp     = infer(model, proc, wave, sr, prompt, args.max_new_tokens)
            preds.append({"id": r["id"], "reference": r["reference"], "hypothesis": hyp})

            if i < 5 or i % 20 == 0:
                print(f"  [{i+1:3d}/{len(rows)}] {r['id']}")
                print(f"     REF: {r['reference'][:90]}")
                print(f"     HYP: {hyp[:90]}")

        with open(out_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\n  → saved {len(preds)} preds: {out_path}")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s  ({dt/max(len(rows),1)/len(PROMPTS):.2f}s/utt/prompt)")
    print(f"\nScore with:")
    for tag, _ in PROMPTS:
        out_path = os.path.join(args.out_dir, f"lora_{tag}.jsonl")
        print(f"  python /data/lfm2.5/run-b/liquid-ai/eval/score.py "
              f"\"LoRA-{tag}:{out_path}\" "
              f"\"Base LFM:/data/lfm2.5/run-b/liquid-ai/artifacts/preds/lfm_read196.jsonl\"")


if __name__ == "__main__":
    main()
