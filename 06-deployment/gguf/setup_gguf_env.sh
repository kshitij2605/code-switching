#!/usr/bin/env bash
# Setup uv venv and build llama.cpp (CPU-only) for GGUF conversion.
# Run once: bash setup_gguf_env.sh
set -euo pipefail

GGUF_DIR="/data/lfm2.5/run-a/gguf"
VENV="$GGUF_DIR/.venv_gguf"
LLAMA_DIR="$GGUF_DIR/llama.cpp"

echo "======================================================"
echo " LFM2.5 Audio -> GGUF  |  Environment Setup"
echo "======================================================"

# ---- 1. Python venv ----------------------------------------------------------
echo "[1/3] Creating uv venv at $VENV ..."
uv venv "$VENV" --python 3.12
source "$VENV/bin/activate"

echo "[1/3] Installing Python dependencies ..."
uv pip install --upgrade pip

# CPU-only torch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core conversion deps
uv pip install \
    safetensors \
    transformers \
    peft \
    accelerate \
    huggingface_hub \
    numpy \
    tqdm \
    "liquid-audio>=0.1"

# Install llama.cpp gguf-py for writing GGUF files
uv pip install "$LLAMA_DIR/gguf-py"

echo "[1/3] Python deps installed."

# ---- 2. Build llama-quantize (CPU, no AVX-VNNI for old binutils) ------------
echo "[2/3] Building llama-quantize (CPU only) ..."
cd "$LLAMA_DIR"
rm -rf build
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_AVX_VNNI=OFF \
    -DGGML_AVX512_VNNI=OFF \
    -DGGML_AMX_INT8=OFF \
    "-DCMAKE_CXX_FLAGS=-mno-avxvnni -mno-avx512vnni" \
    "-DCMAKE_C_FLAGS=-mno-avxvnni -mno-avx512vnni" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    2>&1 | tail -5
cmake --build build --target llama-quantize -j"$(nproc)" 2>&1 | tail -5
echo "[2/3] llama-quantize built: $LLAMA_DIR/build/bin/llama-quantize"

# ---- 3. Verify ---------------------------------------------------------------
echo "[3/3] Verifying environment ..."
source "$VENV/bin/activate"
python - <<'PY'
import torch, safetensors, transformers, peft, gguf
print(f"  torch        : {torch.__version__}")
print(f"  transformers : {transformers.__version__}")
print(f"  peft         : {peft.__version__}")
print(f"  safetensors  : {safetensors.__version__}")
print(f"  gguf         : ok")
PY

echo ""
echo "Done. Activate with: source $VENV/bin/activate"
