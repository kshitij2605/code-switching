#!/usr/bin/env python3
"""
Step 5 – Verify the produced GGUF files are valid and print metadata.

Usage:
  python 05_verify_gguf.py work/gguf_q4/lfm25-audio-jp-backbone-q4_k_m.gguf
  python 05_verify_gguf.py work/gguf_f16/mmproj-lfm25-audio-jp-encoder-bf16.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def verify_gguf(gguf_path: str) -> bool:
    p = Path(gguf_path)
    if not p.exists():
        print(f"ERROR: file not found: {gguf_path}")
        return False

    try:
        # Use llama.cpp's gguf_reader
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "llama.cpp" / "gguf-py"))
        from gguf import GGUFReader
    except ImportError:
        print("gguf package not found; run setup_gguf_env.sh first")
        return False

    print(f"\nVerifying: {p}")
    print(f"  File size: {p.stat().st_size / 1e9:.3f} GB")

    reader = GGUFReader(str(p))

    print("\n  Key-value metadata:")
    kv_print_keys = [
        "general.architecture", "general.name", "llm.context_length",
        "llm.embedding_length", "llm.block_count", "llm.feed_forward_length",
        "tokenizer.ggml.model",
        "clip.audio.encoder_type",
    ]
    for field in reader.fields.values():
        name = field.name
        if any(k in name for k in ["architecture", "name", "context", "embedding",
                                    "block_count", "feed_forward", "vocab", "audio",
                                    "shortconv", "layer_type"]):
            try:
                val = field.parts[field.data[0]] if field.data else "?"
                val_str = bytes(val).decode("utf-8", errors="replace") if hasattr(val, '__iter__') and not isinstance(val, (int, float)) else str(val)
            except Exception:
                val_str = str(field.parts[-1] if field.parts else "?")
            print(f"    {name:<45s} = {val_str[:80]}")

    print(f"\n  Tensors: {len(reader.tensors)}")
    print("  First 10 tensors:")
    for t in reader.tensors[:10]:
        print(f"    {t.name:<50s}  shape={str(list(t.shape)):<20s}  dtype={t.tensor_type.name}")

    print(f"\n  GGUF file is valid.")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Verify GGUF files")
    p.add_argument("files", nargs="+", help="GGUF files to verify")
    args = p.parse_args()

    ok = True
    for f in args.files:
        ok = verify_gguf(f) and ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
