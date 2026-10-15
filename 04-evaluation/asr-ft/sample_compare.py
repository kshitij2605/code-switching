"""
Qualitative sample comparison: Base LFM vs Fine-tuned LFM.
Tests 3 system prompts on 3 audio files (2 real CS-FLEURS, 1 synthetic TTS).

Prints a formatted table so you can visually compare model behaviour before
committing to LoRA fine-tuning.

Usage:
  python sample_compare.py --hf_token hf_xxx \
    --checkpoint /data/lfm2.5/run-a/checkpoints/lfm_asr/final/model.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap

import torch
import soundfile as sf
from safetensors.torch import load_file
from huggingface_hub import login

BASE_MODEL  = "LiquidAI/LFM2.5-Audio-1.5B-JP"
IM_END_TOKEN = 7
MAX_NEW_TOKENS = 200

PROMPTS = {
    "Native  (code-switch)": "Transcribe the audio.",
    "Force Japanese       ": "Transcribe in Japanese.",
    "Force English        ": "Transcribe in English.",
}

# 2 real CS-FLEURS utterances + 1 synthetic TTS sample
SAMPLES = [
    {
        "id": "jpn_1662_SS  (real CS-FLEURS)",
        "audio": "/data/lfm2.5/run-b/liquid-ai/data/csfleurs/read_test/audio/jpn_1662_SS.wav",
        "reference": "合金とは、basically a mixture of 2種類以上のmetalsです。Don't forget that there are たくさんのelements on the periodic table.",
    },
    {
        "id": "jpn_1835_SS  (real CS-FLEURS)",
        "audio": "/data/lfm2.5/run-b/liquid-ai/data/csfleurs/read_test/audio/jpn_1835_SS.wav",
        "reference": "津波警報は発令されませんでした。そして、Jakarta geophysics 機関によれば、quake がmagnitude 6.5の要件を満たさなかったので、tsunami warning の発令に至らなかったということです。",
    },
    {
        "id": "cs_001  (synthetic TTS)",
        "audio": "/data/lfm2.5/run-a/data/code_switching/audio/cs_001.wav",
        "reference": None,   # will load from metadata
    },
]

CSFLEURS_MANIFEST = "/data/lfm2.5/run-b/liquid-ai/data/csfleurs/read_test/manifest.jsonl"
CS_META = "/data/lfm2.5/run-a/data/code_switching/metadata/transcriptions.json"


def load_audio(path: str):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return torch.tensor(arr, dtype=torch.float).unsqueeze(0), sr


def infer(model, proc, wave, sr, system_prompt: str) -> str:
    from liquid_audio import ChatState
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(system_prompt); chat.end_turn()
    chat.new_turn("user");   chat.add_audio(wave, sr);    chat.end_turn()
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


def wrap(text: str, width: int = 90, prefix: str = "    ") -> str:
    return textwrap.fill(text or "(none)", width=width, initial_indent=prefix, subsequent_indent=prefix)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",   default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--checkpoint", default="/data/lfm2.5/run-a/checkpoints/lfm_asr/final/model.safetensors")
    return p.parse_args()


def main():
    args = parse_args()
    if args.hf_token:
        login(token=args.hf_token)

    # Patch reference for synthetic sample from metadata
    try:
        meta = json.loads(open(CS_META, encoding="utf-8").read())
        for s in meta:
            if s.get("id") == "cs_001" or s.get("audio_file", "").endswith("cs_001.wav"):
                SAMPLES[2]["reference"] = s.get("full_transcription", "(unknown)")
                break
    except Exception:
        pass

    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    print("\n" + "="*100)
    print("LOADING BASE MODEL")
    print("="*100)
    proc  = LFM2AudioProcessor.from_pretrained(BASE_MODEL)
    base_model = LFM2AudioModel.from_pretrained(BASE_MODEL, device="cuda", dtype=torch.bfloat16)
    base_model.eval()

    print("\n" + "="*100)
    print("LOADING FINE-TUNED MODEL")
    print("="*100)
    ft_model = LFM2AudioModel.from_pretrained(BASE_MODEL, device="cuda", dtype=torch.bfloat16)
    state_dict = load_file(args.checkpoint, device="cuda")
    missing, unexpected = ft_model.load_state_dict(state_dict, strict=False)
    print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    ft_model.eval()

    print("\n\n" + "="*100)
    print("SAMPLE COMPARISON — BASE vs FINE-TUNED")
    print("="*100)

    for sample in SAMPLES:
        sid   = sample["id"]
        apath = sample["audio"]
        ref   = sample["reference"]

        if not os.path.exists(apath):
            print(f"\n[SKIP] {sid} — audio not found: {apath}")
            continue

        wave, sr = load_audio(apath)

        print(f"\n{'─'*100}")
        print(f"SAMPLE: {sid}")
        print(f"REFERENCE:")
        print(wrap(ref))
        print()

        for prompt_label, prompt_text in PROMPTS.items():
            print(f"  PROMPT: \"{prompt_text}\"   [{prompt_label}]")

            base_hyp = infer(base_model, proc, wave, sr, prompt_text)
            ft_hyp   = infer(ft_model,   proc, wave, sr, prompt_text)

            print(f"  BASE  : {base_hyp[:110]}")
            print(f"  FT    : {ft_hyp[:110]}")
            print()

    print("="*100)
    print("DONE")
    print("="*100)


if __name__ == "__main__":
    main()
