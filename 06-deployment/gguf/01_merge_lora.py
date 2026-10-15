#!/usr/bin/env python3
"""
Step 1 – Merge LoRA adapter weights into the base LFM2.5-Audio model.

Usage:
  python 01_merge_lora.py \
      --base-model LiquidAI/LFM2.5-Audio-1.5B-JP \
      --lora-path  /data/lfm2.5/run-a/checkpoints/lfm_lora/final \
      --output-dir /data/lfm2.5/run-a/gguf/work/merged

The script produces a directory with:
  model.safetensors  – merged float32 weights
  config.json        – original audio model config
  tokenizer*.json    – tokenizer files
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file as sf_load, save_file as sf_save


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into LFM2.5-Audio base model")
    p.add_argument("--base-model", required=True,
                   help="HuggingFace repo ID or local path to base audio model")
    p.add_argument("--lora-path", required=True,
                   help="Path to LoRA adapter checkpoint directory")
    p.add_argument("--output-dir", default="work/merged",
                   help="Where to save the merged model (default: work/merged)")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="Dtype for merged weights (default: float32)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    from liquid_audio import LFM2AudioModel

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = Path(args.lora_path)

    print(f"Loading base model from: {args.base_model}")
    # Pass Path for local dirs so liquid_audio skips snapshot_download
    import os
    base_arg = Path(args.base_model) if os.path.isdir(args.base_model) else args.base_model
    model = LFM2AudioModel.from_pretrained(
        base_arg, dtype=dtype, device=torch.device("cpu")
    )
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Apply LoRA adapter ────────────────────────────────────────────────────
    print(f"Loading LoRA adapter from: {lora_dir}")
    adapter_file = lora_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter safetensors not found: {adapter_file}")

    from peft import get_peft_model, LoraConfig
    adapter_cfg_path = lora_dir / "adapter_config.json"
    with adapter_cfg_path.open() as f:
        acfg = json.load(f)

    lora_config = LoraConfig(
        r=acfg["r"],
        lora_alpha=acfg["lora_alpha"],
        lora_dropout=acfg.get("lora_dropout", 0.0),
        target_modules=acfg["target_modules"],
        bias=acfg.get("bias", "none"),
    )
    model = get_peft_model(model, lora_config)

    adapter_sd = sf_load(str(adapter_file), device="cpu")
    result = set_peft_model_state_dict(model, adapter_sd)
    if result.unexpected_keys or result.missing_keys:
        print(f"  WARNING unexpected={result.unexpected_keys[:3]}  missing={result.missing_keys[:3]}")
    else:
        print("  Adapter weights loaded cleanly.")

    # ── Merge LoRA into base weights ──────────────────────────────────────────
    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()
    print("  Merge complete.")

    # ── Save merged model ─────────────────────────────────────────────────────
    print(f"Saving merged model to: {out_dir}")
    state_dict = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
    sf_save(state_dict, str(out_dir / "model.safetensors"))

    # Copy config and tokenizer files
    import os
    base_dir = Path(args.base_model) if os.path.isdir(args.base_model) else Path(
        __import__('huggingface_hub').snapshot_download(args.base_model)
    )
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "chat_template.jinja"]:
        src = base_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    print(f"\nMerged model saved to: {out_dir}")
    print("  Files:", [f.name for f in sorted(out_dir.iterdir())])


if __name__ == "__main__":
    main()
