#!/usr/bin/env python3
"""Zero-shot transcription with transformers-native speech LLMs (Phase 1 leaderboard).

Families (all run in the locked research environment):
  granite    ibm-granite/granite-4.0-1b-speech, granite-speech-4.1-2b (Apache-2.0;
             ASR languages include Japanese). Prompt from the model card:
             "<|audio|>can you transcribe the speech into a written format?".
             --keywords appends the card's keyword-biasing suffix.
  qwen3omni  Qwen/Qwen3-Omni-30B-A3B-Instruct (Apache-2.0), text output only.
             Prompt: "Transcribe the audio into text exactly as spoken. Do not
             translate." Needs a whole H100 (~65 GB bf16).

Decoding is greedy (num_beams=1, do_sample=False), batch 1, --max-new-tokens 256.
Output JSONL matches the shared contract ({id, reference, hypothesis}), so
predictions drop straight into eval/score.py.

Example:
  CUDA_VISIBLE_DEVICES=0 $PY eval/run_audio_llm.py --family granite \
      --model-id ibm-granite/granite-4.0-1b-speech --manifest M.jsonl --audio-root ROOT --out preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline import load_audio_mono, read_manifest  # noqa: E402

GRANITE_PROMPT = "<|audio|>can you transcribe the speech into a written format?"


class Granite:
    def __init__(self, model_id: str, keywords: str | None):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.tok = self.proc.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, device_map="cuda", dtype=torch.bfloat16).eval()
        user = GRANITE_PROMPT + (f" Keywords: {keywords}" if keywords else "")
        self.prompt = self.tok.apply_chat_template([{"role": "user", "content": user}],
                                                   tokenize=False, add_generation_prompt=True)

    def transcribe(self, path: str, max_new_tokens: int) -> str:
        # soundfile I/O: torchaudio 2.11's load() needs torchcodec, which the lock omits
        import torchaudio
        arr, sr = load_audio_mono(path)
        wav = self.torch.from_numpy(arr).float().unsqueeze(0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        inputs = self.proc(self.prompt, wav, device="cuda", return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False, num_beams=1)
        new = out[0, inputs["input_ids"].shape[-1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip()


QWEN3OMNI_PROMPT = "Transcribe the audio into text exactly as spoken. Do not translate."


class Qwen3Omni:
    """Qwen/Qwen3-Omni-30B-A3B-Instruct (Apache-2.0; 19 speech-input languages incl.
    Japanese). Text-only: the talker (speech output) is disabled. About 65 GB
    in bf16, so it needs a whole H100."""

    def __init__(self, model_id: str, keywords: str | None):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        self.torch = torch
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        self.model.disable_talker()
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(model_id)
        self.prompt = QWEN3OMNI_PROMPT + (f" Terms that may occur: {keywords}." if keywords else "")

    def transcribe(self, path: str, max_new_tokens: int) -> str:
        return self.transcribe_batch([path], max_new_tokens)[0]

    def transcribe_batch(self, paths: list[str], max_new_tokens: int) -> list[str]:
        """Batched greedy decoding (left padding). Transformers runs this MoE's
        experts in a Python loop, so batch 1 costs ~70 s/utt on an H100; batching
        amortises the per-token expert calls."""
        import torchaudio
        texts, audios = [], []
        for path in paths:
            arr, sr = load_audio_mono(path)
            if sr != 16000:
                arr = torchaudio.functional.resample(self.torch.from_numpy(arr).float(), sr, 16000).numpy()
            conv = [{"role": "user", "content": [{"type": "audio", "audio": path},
                                                 {"type": "text", "text": self.prompt}]}]
            texts.append(self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False))
            audios.append(arr)
        self.proc.tokenizer.padding_side = "left"
        inputs = self.proc(text=texts, audio=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with self.torch.inference_mode():
            ids, _ = self.model.generate(**inputs, return_audio=False, use_audio_in_video=False,
                                         thinker_max_new_tokens=max_new_tokens, thinker_do_sample=False)
        seq = ids.sequences if hasattr(ids, "sequences") else ids
        return [t.strip() for t in self.proc.batch_decode(seq[:, inputs["input_ids"].shape[1]:],
                                                          skip_special_tokens=True)]


FAMILIES = {"granite": Granite, "qwen3omni": Qwen3Omni}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keywords", default=None, help="granite: keyword-biasing list")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Utterances per generate() call (families with transcribe_batch only)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    rows = read_manifest(args.manifest, args.audio_root, args.limit)
    if not rows:
        raise SystemExit("Empty manifest — nothing to run.")
    print(f"[{args.family}] {args.model_id}: {len(rows)} utterances")
    runner = FAMILIES[args.family](args.model_id, args.keywords)

    t0, out = time.time(), []
    bs = args.batch_size if hasattr(runner, "transcribe_batch") else 1
    for b in range(0, len(rows), bs):
        batch = rows[b:b + bs]
        hyps = (runner.transcribe_batch([r["_audio_path"] for r in batch], args.max_new_tokens)
                if bs > 1 else [runner.transcribe(batch[0]["_audio_path"], args.max_new_tokens)])
        for r, hyp in zip(batch, hyps):
            out.append({"id": r["id"], "reference": r["reference"], "hypothesis": hyp})
            i = len(out) - 1
            if i < 3 or (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(rows)}] {r['id']}\n     REF: {r['reference'][:90]}\n     HYP: {hyp[:90]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"\nWrote {len(out)} predictions -> {args.out}  ({dt:.1f}s, {dt / len(out):.2f}s/utt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
