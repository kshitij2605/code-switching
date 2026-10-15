#!/usr/bin/env bash
# Step 3 – Convert backbone + audio encoder to GGUF using llama.cpp.
#
# Usage:
#   bash 03_convert_gguf.sh \
#       --backbone-hf   work/backbone_hf \
#       --audio-model   LiquidAI/LFM2.5-Audio-1.5B-JP \   # or local merged path
#       --output-dir    work/gguf_f16
#
# Produces:
#   work/gguf_f16/lfm25-audio-jp-backbone-f16.gguf
#   work/gguf_f16/mmproj-lfm25-audio-jp-encoder-f16.gguf
set -euo pipefail

GGUF_DIR="/data/lfm2.5/run-a/gguf"
LLAMA_DIR="$GGUF_DIR/llama.cpp"
VENV="$GGUF_DIR/.venv_gguf"

# ── Argument parsing ──────────────────────────────────────────────────────────
BACKBONE_HF="$GGUF_DIR/work/backbone_hf"
AUDIO_MODEL=""
OUTPUT_DIR="$GGUF_DIR/work/gguf_f16"
OUTTYPE="bf16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbone-hf)   BACKBONE_HF="$2"; shift 2 ;;
        --audio-model)   AUDIO_MODEL="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";  shift 2 ;;
        --outtype)       OUTTYPE="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
source "$VENV/bin/activate"

CONVERT="$LLAMA_DIR/convert_hf_to_gguf.py"

# ── Resolve audio model path (HF ID → local cache) ───────────────────────────
if [[ -z "$AUDIO_MODEL" ]]; then
    echo "ERROR: --audio-model is required"
    exit 1
fi

AUDIO_MODEL_DIR="$AUDIO_MODEL"
if [[ ! -d "$AUDIO_MODEL" ]]; then
    echo "Resolving HuggingFace model: $AUDIO_MODEL"
    AUDIO_MODEL_DIR=$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$AUDIO_MODEL'))
")
fi
echo "Audio model dir: $AUDIO_MODEL_DIR"

# ── Step 3a: Convert LFM2 backbone (text model) ──────────────────────────────
BACKBONE_OUT="$OUTPUT_DIR/lfm25-audio-jp-backbone-${OUTTYPE}.gguf"
echo ""
echo "============================="
echo " Converting LFM2 backbone"
echo "============================="
echo "  Input  : $BACKBONE_HF"
echo "  Output : $BACKBONE_OUT"
echo "  Type   : $OUTTYPE"

python "$CONVERT" \
    "$BACKBONE_HF" \
    --outfile "$BACKBONE_OUT" \
    --outtype "$OUTTYPE"

echo "Backbone GGUF: $BACKBONE_OUT"

# ── Step 3b: Convert Conformer audio encoder (mmproj) ────────────────────────
MMPROJ_OUT="$OUTPUT_DIR/mmproj-lfm25-audio-jp-encoder-${OUTTYPE}.gguf"
echo ""
echo "====================================="
echo " Converting Conformer audio encoder"
echo "====================================="
echo "  Input  : $AUDIO_MODEL_DIR"
echo "  Output : $MMPROJ_OUT"
echo "  Type   : $OUTTYPE"

python "$CONVERT" \
    "$AUDIO_MODEL_DIR" \
    --mmproj \
    --outfile "$MMPROJ_OUT" \
    --outtype "$OUTTYPE"

echo "Audio encoder mmproj GGUF: $MMPROJ_OUT"

echo ""
echo "=============================="
echo " GGUF conversion complete"
echo "=============================="
ls -lh "$OUTPUT_DIR"
