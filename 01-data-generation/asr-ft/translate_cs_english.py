"""
Translates all CS samples' all_english field into proper fluent English
using an OpenAI-compatible Qwen vLLM endpoint (--base-url / --model).

Notes:
  * A failed or invalid translation is never replaced by the source CS text: the sample's all_english is set
    to null, and prepare_training_data.py skips its "Transcribe in English."
    example. The run exits non-zero if more than --max-fail-frac of the samples
    fail.
  * Outputs are validated: reasoning blocks (<think>...</think>) are stripped,
    and the result must be non-empty and mostly Latin script.
  * The script does not build train/eval JSONL itself. Run 02-preprocessing/asr-ft/prepare_training_data.py
    afterwards.
"""

import argparse, io, json, re, sys, time
from pathlib import Path
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CS_META = Path("data/code_switching/metadata/transcriptions.json")

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "You will receive a short Japanese-English code-switched sentence. "
    "Translate it into one fluent, natural English sentence. "
    "Do not add explanations, notes, or punctuation beyond the sentence. "
    "Output only the English translation."
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_JA = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:9000/v1")
    p.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-fail-frac", type=float, default=0.02)
    return p.parse_args()


def valid_english(text: str) -> bool:
    """Non-empty, no leftover reasoning, and >= 90% of letters are Latin."""
    if not text or "<think>" in text:
        return False
    latin, ja = len(_LATIN.findall(text)), len(_JA.findall(text))
    return latin > 0 and latin / (latin + ja) >= 0.9


def translate(client: OpenAI, model: str, text: str, retries: int = 3) -> str | None:
    """English translation, or None if every attempt fails or is invalid."""
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                max_tokens=128,
                temperature=0.0,
            )
            out = _THINK.sub("", resp.choices[0].message.content or "").strip()
            if valid_english(out):
                return out
            print(f"  [attempt {attempt}/{retries}] invalid output: {out[:80]!r}")
        except Exception as e:
            print(f"  [attempt {attempt}/{retries}] error: {e}")
        if attempt < retries:
            time.sleep(2 * attempt)
    return None


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key="none")
    samples = json.loads(CS_META.read_text(encoding="utf-8"))
    total   = len(samples)
    print(f"Translating {total} CS samples via {args.model} at {args.base_url}...\n")

    failed = []
    for i, s in enumerate(samples, 1):
        full = s.get("full_transcription", "")
        print(f"[{i:03d}/{total}] {s['id']}  |  {full[:60]}")
        translation = translate(client, args.model, full, args.retries)
        s["all_english"] = translation
        s["all_english_model"] = args.model
        if translation is None:
            failed.append(s["id"])
            print("         → FAILED (all_english = null)")
        else:
            print(f"         → {translation[:80]}")

        # Save every 50 samples as checkpoint
        if i % 50 == 0:
            CS_META.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  💾 Checkpoint saved at {i}/{total}")

    # Final save of metadata
    CS_META.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{total - len(failed)}/{total} translations done; {len(failed)} failed "
          f"(e.g. {failed[:5]}). Metadata saved.")
    print("Next: python 02-preprocessing/asr-ft/prepare_training_data.py")
    if len(failed) > args.max_fail_frac * total:
        sys.exit(f"too many failed translations ({len(failed)}/{total}); fix the endpoint and rerun")


if __name__ == "__main__":
    main()
