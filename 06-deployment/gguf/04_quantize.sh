#!/usr/bin/env bash
# Step 4 – Quantize the backbone GGUF to a smaller, CPU-friendly format.
#
# Supported quant types (trade-off: quality vs. size/speed):
#   Q4_K_M  – recommended for most use cases (~4 bit, good quality)
#   Q5_K_M  – higher quality, ~5 bit
#   Q8_0    – near-lossless 8 bit
#   Q2_K    – smallest, most aggressive compression
#
# Usage:
#   bash 04_quantize.sh \
#       --input   work/gguf_f16/lfm25-audio-jp-backbone-bf16.gguf \
#       --output  work/gguf_q4/lfm25-audio-jp-backbone-q4_k_m.gguf \
#       --type    Q4_K_M
set -euo pipefail

GGUF_DIR="/data/lfm2.5/run-a/gguf"
LLAMA_DIR="$GGUF_DIR/llama.cpp"
QUANTIZE_BIN="$LLAMA_DIR/build/bin/llama-quantize"

INPUT=""
OUTPUT=""
QTYPE="Q4_K_M"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)   INPUT="$2";  shift 2 ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        --type)    QTYPE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$INPUT" ]]; then
    echo "ERROR: --input is required"
    echo "Usage: bash 04_quantize.sh --input <backbone.gguf> [--output <out.gguf>] [--type Q4_K_M]"
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    QTYPE_LOWER="${QTYPE,,}"
    OUTPUT="${INPUT%.gguf}-${QTYPE_LOWER}.gguf"
fi

if [[ ! -f "$QUANTIZE_BIN" ]]; then
    echo "ERROR: llama-quantize not found at $QUANTIZE_BIN"
    echo "Run setup_gguf_env.sh first to build llama.cpp."
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "=============================="
echo " Quantizing GGUF"
echo "=============================="
echo "  Input  : $INPUT"
echo "  Output : $OUTPUT"
echo "  Type   : $QTYPE"
echo ""

"$QUANTIZE_BIN" "$INPUT" "$OUTPUT" "$QTYPE"

echo ""
echo "Input  size: $(du -sh "$INPUT"  | cut -f1)"
echo "Output size: $(du -sh "$OUTPUT" | cut -f1)"
echo "Done: $OUTPUT"
