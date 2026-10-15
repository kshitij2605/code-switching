"""Option B: preprocess the translating-TTS JSONL into liquid-audio's format.

Mirrors the teammate's preprocess_asr_data.py, but the assistant turn carries BOTH the translated text
and the target audio: content=[TextSegment(target_text), AudioSegment(wav_bytes)]. The mapper encodes this
as `{target_text} <|audio_start|> {audio codes}` with supervision on text+audio — so the model learns to
emit the translation then speak it (sequential generation), matching the app's existing inference path.
"""
from __future__ import annotations

import argparse, json, os
from collections.abc import Iterator
from pathlib import Path

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment


class TransTTSIterator:
    def __init__(self, jsonl_path: str) -> None:
        self.examples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    if Path(ex["audio_file"]).exists():
                        self.examples.append(ex)
        print(f"  Loaded {len(self.examples)} examples from {jsonl_path}")

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        for ex in self.examples:
            audio_bytes = Path(ex["audio_file"]).read_bytes()
            yield [
                ChatMessage(role="system", content=[TextSegment(text=ex["system"])]),
                ChatMessage(role="user", content=[TextSegment(text=ex["input_text"])]),
                ChatMessage(role="assistant", content=[
                    TextSegment(text=ex["target_text"]),
                    AudioSegment(audio=audio_bytes),
                ]),
            ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="LiquidAI/LFM2.5-Audio-1.5B-JP")
    p.add_argument("--train_jsonl", default="/data/lfm2.5/run-c/optionB/data/training/train.jsonl")
    p.add_argument("--eval_jsonl", default="/data/lfm2.5/run-c/optionB/data/training/eval.jsonl")
    p.add_argument("--output_dir", default="/data/lfm2.5/run-c/optionB/data/preprocessed")
    p.add_argument("--max_context", type=int, default=1024)
    args = p.parse_args()

    print(f"Loading processor: {args.model_id}")
    processor = LFM2AudioProcessor.from_pretrained(args.model_id, device="cuda").eval()
    mapper = LFM2AudioChatMapper(processor)
    out_root = Path(args.output_dir)

    for split, jsonl in [("train", args.train_jsonl), ("eval", args.eval_jsonl)]:
        out_path = out_root / split
        if out_path.exists():
            print(f"[{split}] exists at {out_path}, skipping.")
            continue
        print(f"\n[{split}] Preprocessing {jsonl} -> {out_path}")
        preprocess_dataset(data=TransTTSIterator(jsonl), output_path=out_path,
                           mapper=mapper, max_context_length=args.max_context)
        print(f"[{split}] Done.")
    print("\nDone. Dataset at:", args.output_dir)


if __name__ == "__main__":
    main()
