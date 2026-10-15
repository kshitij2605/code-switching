#!/usr/bin/env python3
"""Run a Hugging Face transformers Whisper checkpoint over the eval manifest.

Use this for **fine-tuned Whisper** checkpoints stored in transformers format
(safetensors + processor configs) — e.g. <hf-user>/whisper-large-v3-ja-en-cs-merged.
Unlike eval/run_baseline.py's `whisper` path (faster-whisper / CTranslate2, whose
bundled CUDA kernels hit cudaErrorUnsupportedPtxVersion on this box's driver),
this runs through transformers + our working cu126 torch, so it uses the GPU.

Output JSONL matches the shared contract ({id, reference, hypothesis}) so the
predictions drop straight into eval/score.py.

Private repos: pass --hf-token-file pointing at a file containing only the token.

Example (pin GPU 2 on the shared box):
    CUDA_VISIBLE_DEVICES=2 python eval/run_whisper_hf.py \
        --model-id <hf-user>/whisper-large-v3-ja-en-cs-merged \
        --hf-token-file hf_token.txt \
        --manifest data/eval/csfleurs_jaen_read196.jsonl --audio-root data/csfleurs \
        --out artifacts/preds/whisper_ft_read196.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# reuse the manifest reader + audio loader from the baseline runner
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline import load_audio_mono, read_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run an HF transformers Whisper checkpoint")
    ap.add_argument("--model-id", required=True, help="HF repo id or local path (base model)")
    ap.add_argument("--lora-adapter", default=None,
                    help="Optional PEFT LoRA adapter dir to merge onto --model-id "
                         "(fine-tuned Whisper eval without a merged HF repo)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hf-token-file", default=None, help="File containing only the HF token")
    ap.add_argument("--language", default=None,
                    help="Force decoding language (default: auto-detect, matches base Whisper run)")
    ap.add_argument("--chunk-length-s", type=int, default=30, help="Long-form chunk size")
    ap.add_argument("--batch-size", type=int, default=1)
    # Decoding controls. The ja-en-cs fine-tune degenerates into single-char
    # repetition under plain greedy/beam decoding; repetition_penalty +
    # no_repeat_ngram_size are required for it to produce usable output. Apply
    # the SAME values to the base model for a fair head-to-head.
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--num-beams", type=int, default=1)
    args = ap.parse_args(argv)

    token = None
    if args.hf_token_file:
        with open(args.hf_token_file, "r", encoding="utf-8") as f:
            token = f.read().strip()

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    dev_name = torch.cuda.get_device_name(0) if device == 0 else "cpu"

    if args.lora_adapter:
        # Load base Whisper, merge the PEFT LoRA adapter, build the pipeline from
        # the merged model. Processor/tokenizer come from the adapter dir (it
        # carries the fine-tune's tokenizer + added tokens).
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from peft import PeftModel
        print(f"[whisper-hf] base {args.model_id} + LoRA {args.lora_adapter} on {dev_name} ({dtype})")
        base = WhisperForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=dtype, token=token)
        merged = PeftModel.from_pretrained(base, args.lora_adapter).merge_and_unload()
        if device == 0:
            merged = merged.to("cuda")
        proc = WhisperProcessor.from_pretrained(args.lora_adapter)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=merged, tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
            device=device, torch_dtype=dtype, chunk_length_s=args.chunk_length_s,
        )
    else:
        print(f"[whisper-hf] loading {args.model_id} on {dev_name} ({dtype})")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model_id, token=token, device=device,
            torch_dtype=dtype, chunk_length_s=args.chunk_length_s,
        )
    print("[whisper-hf] loaded.")

    gen_kwargs = {"task": "transcribe"}
    if args.language:
        gen_kwargs["language"] = args.language
    if args.num_beams > 1:
        gen_kwargs["num_beams"] = args.num_beams
    if args.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    print(f"[whisper-hf] generate kwargs: {gen_kwargs}")

    rows = read_manifest(args.manifest, args.audio_root, args.limit)
    print(f"Loaded {len(rows)} manifest rows from {args.manifest}")
    if not rows:
        raise SystemExit("Empty manifest — nothing to run.")

    t0 = time.time()
    out = []
    for i, r in enumerate(rows):
        arr, sr = load_audio_mono(r["_audio_path"])  # float32 mono, native sr
        res = pipe({"raw": arr, "sampling_rate": sr}, generate_kwargs=gen_kwargs)
        hyp = (res.get("text") or "").strip()
        out.append({"id": r["id"], "reference": r["reference"], "hypothesis": hyp})
        if i < 5 or (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}] {r['id']}\n     REF: {r['reference'][:90]}\n     HYP: {hyp[:90]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"\nWrote {len(out)} predictions -> {args.out}  ({dt:.1f}s, {dt/max(len(out),1):.2f}s/utt)")
    print(f"Score with:\n  python eval/score.py whisper-ft:{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
