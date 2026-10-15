#!/usr/bin/env python3
"""
Step 2 – Extract the LFM2 text backbone from a LFM2.5-Audio model and save it
         as a standalone HuggingFace model directory suitable for llama.cpp's
         convert_hf_to_gguf.py.

Usage:
  # From base HF model (no LoRA):
  python 02_prepare_backbone_hf.py \
      --model-path LiquidAI/LFM2.5-Audio-1.5B-JP \
      --output-dir work/backbone_hf

  # From already-merged local model:
  python 02_prepare_backbone_hf.py \
      --model-path work/merged \
      --output-dir work/backbone_hf

Produces a directory with:
  model.safetensors  – only the LFM2 backbone weights (lfm.* → model.*)
  config.json        – Lfm2ForCausalLM config extracted from lfm sub-config
  tokenizer*.json    – tokenizer files
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file as sf_load, save_file as sf_save
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract LFM2 backbone as HF model")
    p.add_argument("--model-path", required=True,
                   help="HuggingFace repo ID or local path to the full audio model")
    p.add_argument("--output-dir", default="work/backbone_hf",
                   help="Where to save the backbone HF model (default: work/backbone_hf)")
    return p.parse_args()


def resolve_model_dir(model_path: str) -> Path:
    p = Path(model_path)
    if p.exists():
        return p
    # HuggingFace repo ID
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_path))


def main() -> None:
    args = parse_args()

    model_dir = resolve_model_dir(args.model_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load full audio model config ──────────────────────────────────────────
    config_path = model_dir / "config.json"
    with config_path.open() as f:
        full_config = json.load(f)

    print(f"Model architecture : {full_config.get('architectures')}")
    print(f"Source             : {model_dir}")
    print(f"Output             : {out_dir}")

    # ── Build Lfm2ForCausalLM config from the lfm sub-config ─────────────────
    lfm_config = full_config["lfm"].copy()
    # Ensure correct architectures field
    lfm_config["architectures"] = ["Lfm2ForCausalLM"]
    lfm_config.setdefault("model_type", "lfm2")

    with (out_dir / "config.json").open("w") as f:
        json.dump(lfm_config, f, indent=2)
    print("Wrote backbone config.json")

    # ── Extract backbone weights (lfm.* → model.*) ───────────────────────────
    weights_file = model_dir / "model.safetensors"
    print(f"Loading weights from: {weights_file}")
    all_weights = sf_load(str(weights_file), device="cpu")

    backbone_weights: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, tensor in tqdm(all_weights.items(), desc="Filtering tensors"):
        if key.startswith("lfm."):
            new_key = key.replace("lfm.", "model.", 1)
            backbone_weights[new_key] = tensor.contiguous()
        else:
            skipped += 1

    print(f"  Backbone tensors : {len(backbone_weights)}")
    print(f"  Skipped tensors  : {skipped} (audio encoder, depthformer, etc.)")

    sf_save(backbone_weights, str(out_dir / "model.safetensors"))
    print("Wrote backbone model.safetensors")

    # ── Copy tokenizer files ─────────────────────────────────────────────────
    for fname in ["tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "chat_template.jinja",
                  "tokenizer-e351c8d8-checkpoint125.safetensors"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            print(f"Copied {fname}")

    print(f"\nBackbone HF model ready at: {out_dir}")
    total_params = sum(t.numel() for t in backbone_weights.values())
    print(f"Total backbone parameters: {total_params:,} ({total_params/1e9:.2f}B)")


if __name__ == "__main__":
    main()
