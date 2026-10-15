# 06 · Deployment (on-device)

## gguf/ — llama.cpp GGUF pipeline (asr-ft)
Merge the LoRA into the backbone and quantize for CPU/laptop inference:
1. `01_merge_lora.py` — merge adapter into base weights.
2. `02_prepare_backbone_hf.py` — export merged HF backbone.
3. `03_convert_gguf.sh` — convert to GGUF (llama.cpp).
4. `04_quantize.sh` — quantize (e.g. Q4).
5. `05_verify_gguf.py` — verify the quantized model.
`run_pipeline.sh` runs the full chain; `setup_gguf_env.sh` provisions the toolchain.
(The `llama.cpp` checkout and compiled runner binaries are excluded — clone/build upstream.)

## onnx/ — ONNX export path
See `onnx/README.md`.
