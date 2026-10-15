# ONNX export path

The CPU/ONNX deployment path uses **LiquidONNX**, the upstream exporter at
[`Liquid4All/onnx-export`](https://github.com/Liquid4All/onnx-export), which already supports
LFM2.5-Audio (ASR / TTS / interleaved) export to fp16/q4/q8.

In this project the repo was used **unmodified** (a clean upstream checkout, no local changes),
so no fork is vendored here. To reproduce the on-device ONNX path:

```bash
git clone https://github.com/Liquid4All/onnx-export.git
cd onnx-export && uv sync
# export + run LFM2.5-Audio (see that repo README, section 4.4 Audio)
```

The Kaigi app consumes ONNX via `kaigi-app/src/csmeeting/live/onnx_engine.py`
(`--backend onnx --device cpu`, optional DirectML EP for Ryzen/Radeon iGPU).
