"""Option C: preprocess (transcript -> minutes) text->text pairs for the minutes LoRA.

Assistant turn is the minutes TEXT (TextSegment, supervised). system = the minutes SYSTEM_PROMPT (no few-shot —
the LoRA learns the format), user = transcript. Mirrors optionB_preprocess but pure text in / text out.
"""
from __future__ import annotations
import argparse, json, sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, "/data/lfm2.5/run-c/appsrc")
from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import ChatMessage, TextSegment
from csmeeting.minutes.prompt import SYSTEM_PROMPT


class MinutesIterator:
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        print(f"  Loaded {len(self.rows)} from {path}")

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        for r in self.rows:
            yield [
                ChatMessage(role="system", content=[TextSegment(text=SYSTEM_PROMPT)]),
                ChatMessage(role="user", content=[TextSegment(text=r["transcript"])]),
                ChatMessage(role="assistant", content=[TextSegment(text=r["minutes"])]),
            ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="LiquidAI/LFM2.5-Audio-1.5B-JP")
    ap.add_argument("--train_jsonl", default="/data/lfm2.5/run-c/optionC/data/training/train.jsonl")
    ap.add_argument("--eval_jsonl", default="/data/lfm2.5/run-c/optionC/data/training/eval.jsonl")
    ap.add_argument("--output_dir", default="/data/lfm2.5/run-c/optionC/data/preprocessed")
    ap.add_argument("--max_context", type=int, default=2048)
    args = ap.parse_args()
    proc = LFM2AudioProcessor.from_pretrained(args.model_id, device="cuda").eval()
    mapper = LFM2AudioChatMapper(proc)
    for split, jsonl in [("train", args.train_jsonl), ("eval", args.eval_jsonl)]:
        out = Path(args.output_dir) / split
        if out.exists():
            print(f"[{split}] exists, skip"); continue
        print(f"[{split}] {jsonl} -> {out}")
        preprocess_dataset(data=MinutesIterator(jsonl), output_path=out, mapper=mapper, max_context_length=args.max_context)
    print("done")


if __name__ == "__main__":
    main()
