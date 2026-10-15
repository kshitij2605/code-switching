"""
Evaluate our fine-tuned LFM2.5-Audio-1.5B-JP model on CS-FLEURS 196-utterance benchmark.

Loads the base model from HF, overlays our fine-tuned weights, then runs
inference using the same manifest/scoring format as the team's eval harness.

Usage:
  python eval_finetuned.py \
    --checkpoint /data/lfm2.5/run-a/checkpoints/lfm_asr/final/model.safetensors \
    --manifest /data/lfm2.5/run-b/liquid-ai/data/csfleurs/read_test/manifest.jsonl \
    --audio-root /data/lfm2.5/run-b/liquid-ai/data/csfleurs \
    --out /data/lfm2.5/run-b/liquid-ai/artifacts/preds/finetuned_read196.jsonl \
    --hf_token hf_xxx
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import soundfile as sf
from safetensors.torch import load_file
from huggingface_hub import login

BASE_MODEL = "LiquidAI/LFM2.5-Audio-1.5B-JP"
ASR_PROMPT = "Transcribe the audio."   # the system prompt our model was trained on
IM_END_TOKEN = 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True, help="Path to model.safetensors")
    p.add_argument("--manifest",    required=True)
    p.add_argument("--audio-root",  default=".")
    p.add_argument("--out",         required=True)
    p.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--limit",       type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--system-prompt", default=ASR_PROMPT)
    return p.parse_args()


def load_audio_mono(path: str):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def main():
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    # Load manifest
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

    # Load model
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

    print(f"Loading base model: {BASE_MODEL}")
    proc  = LFM2AudioProcessor.from_pretrained(BASE_MODEL)
    model = LFM2AudioModel.from_pretrained(BASE_MODEL, device="cuda", dtype=torch.bfloat16)

    print(f"Loading fine-tuned weights: {args.checkpoint}")
    state_dict = load_file(args.checkpoint, device="cuda")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    print("  Weights loaded OK")

    model.eval()

    # Run inference
    print(f"\nSystem prompt: '{args.system_prompt}'")
    print(f"Running on {len(rows)} utterances...\n")
    preds = []
    t0 = time.time()

    for i, r in enumerate(rows):
        arr, sr = load_audio_mono(r["_audio_path"])
        wave = torch.tensor(arr, dtype=torch.float).unsqueeze(0)

        chat = ChatState(proc)
        chat.new_turn("system")
        chat.add_text(args.system_prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wave, sr)
        chat.end_turn()
        chat.new_turn("assistant")

        toks = []
        with torch.no_grad():
            for t in model.generate_sequential(**chat, max_new_tokens=args.max_new_tokens):
                if t.numel() == 1:
                    tid = int(t.reshape(-1)[0].item())
                    if tid == IM_END_TOKEN:
                        break
                    toks.append(tid)

        hyp = proc.text.decode(toks, skip_special_tokens=True).strip() if toks else ""
        preds.append({"id": r["id"], "reference": r["reference"], "hypothesis": hyp})

        if i < 5 or i % 20 == 0:
            print(f"  [{i+1:3d}/{len(rows)}] {r['id']}")
            print(f"     REF: {r['reference'][:90]}")
            print(f"     HYP: {hyp[:90]}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    dt = time.time() - t0
    print(f"\nWrote {len(preds)} predictions → {args.out}  ({dt:.1f}s, {dt/max(len(preds),1):.2f}s/utt)")
    print(f"\nScore with:")
    print(f"  python /data/lfm2.5/run-b/liquid-ai/eval/score.py "
          f"\"Fine-tuned:{args.out}\" "
          f"\"Base LFM:/data/lfm2.5/run-b/liquid-ai/artifacts/preds/lfm_read196.jsonl\" "
          f"\"Whisper:/data/lfm2.5/run-b/liquid-ai/artifacts/preds/whisper_read196.jsonl\"")


if __name__ == "__main__":
    main()
