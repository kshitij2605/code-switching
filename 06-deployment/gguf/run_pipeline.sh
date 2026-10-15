#!/usr/bin/env bash
# LFM2.5-Audio JP -> GGUF Master Pipeline
# See --help for full usage.
set -euo pipefail

GGUF_DIR="/data/lfm2.5/run-a/gguf"
VENV="$GGUF_DIR/.venv_gguf"
SCRIPTS_DIR="$GGUF_DIR"

MODEL=""
LORA_PATH=""
QUANT_TYPE="Q4_K_M"
NO_QUANT=0
OUTTYPE="bf16"
OUT_DIR=""
SKIP_SETUP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";      shift 2 ;;
        --lora)        LORA_PATH="$2";  shift 2 ;;
        --quant)       QUANT_TYPE="$2"; shift 2 ;;
        --no-quant)    NO_QUANT=1;      shift   ;;
        --out-dir)     OUT_DIR="$2";    shift 2 ;;
        --outtype)     OUTTYPE="$2";    shift 2 ;;
        --skip-setup)  SKIP_SETUP=1;    shift   ;;
        -h|--help)
            echo "Usage: bash run_pipeline.sh --model <HF_ID_or_path> [--lora <path>] [--quant Q4_K_M|Q5_K_M|Q8_0] [--no-quant] [--out-dir <dir>] [--outtype bf16|f16|f32]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model is required"; exit 1; }

MODEL_SLUG="${MODEL##*/}"; MODEL_SLUG="${MODEL_SLUG,,}"
[[ -z "$OUT_DIR" ]] && OUT_DIR="$GGUF_DIR/output/$MODEL_SLUG"

WORK_DIR="$OUT_DIR/work"
MERGED_DIR="$WORK_DIR/merged"
BACKBONE_HF_DIR="$WORK_DIR/backbone_hf"
GGUF_F16_DIR="$OUT_DIR/gguf_${OUTTYPE}"
GGUF_QUANT_DIR="$OUT_DIR/gguf_${QUANT_TYPE,,}"
mkdir -p "$WORK_DIR" "$GGUF_F16_DIR"

echo "========================================================"
echo " LFM2.5 Audio JP -> GGUF Pipeline"
echo " Model    : $MODEL"
echo " LoRA     : ${LORA_PATH:-(none)}"
echo " Quant    : $([ $NO_QUANT -eq 0 ] && echo $QUANT_TYPE || echo disabled)"
echo " Out dir  : $OUT_DIR"
echo "========================================================"

if [[ $SKIP_SETUP -eq 0 && ! -d "$VENV" ]]; then
    echo "Setting up environment first ..."
    bash "$SCRIPTS_DIR/setup_gguf_env.sh"
fi
source "$VENV/bin/activate"

AUDIO_SOURCE="$MODEL"

# Step 1: LoRA merge
if [[ -n "$LORA_PATH" ]]; then
    echo ""; echo "--- Step 1/4: Merging LoRA ---"
    python "$SCRIPTS_DIR/01_merge_lora.py" \
        --base-model "$MODEL" --lora-path "$LORA_PATH" \
        --output-dir "$MERGED_DIR" --dtype float32
    AUDIO_SOURCE="$MERGED_DIR"
else
    echo ""; echo "--- Step 1/4: No LoRA, using base model ---"
fi

# Step 2: Extract backbone
echo ""; echo "--- Step 2/4: Preparing backbone HF model ---"
python "$SCRIPTS_DIR/02_prepare_backbone_hf.py" \
    --model-path "$AUDIO_SOURCE" --output-dir "$BACKBONE_HF_DIR"

# Step 3: GGUF conversion
echo ""; echo "--- Step 3/4: Converting to GGUF ---"
bash "$SCRIPTS_DIR/03_convert_gguf.sh" \
    --backbone-hf "$BACKBONE_HF_DIR" \
    --audio-model "$AUDIO_SOURCE" \
    --output-dir  "$GGUF_F16_DIR" \
    --outtype     "$OUTTYPE"

BACKBONE_GGUF=$(ls "$GGUF_F16_DIR"/*backbone*.gguf | head -1)
MMPROJ_GGUF=$(ls  "$GGUF_F16_DIR"/mmproj*.gguf      | head -1)

# Step 4: Quantize
if [[ $NO_QUANT -eq 0 ]]; then
    echo ""; echo "--- Step 4/4: Quantizing -> $QUANT_TYPE ---"
    mkdir -p "$GGUF_QUANT_DIR"
    QTYPE_LOWER="${QUANT_TYPE,,}"
    QUANT_OUT="$GGUF_QUANT_DIR/$(basename "${BACKBONE_GGUF%.gguf}")-${QTYPE_LOWER}.gguf"
    bash "$SCRIPTS_DIR/04_quantize.sh" --input "$BACKBONE_GGUF" --output "$QUANT_OUT" --type "$QUANT_TYPE"
    cp "$MMPROJ_GGUF" "$GGUF_QUANT_DIR/"
    FINAL_BACKBONE="$QUANT_OUT"
    FINAL_MMPROJ="$GGUF_QUANT_DIR/$(basename "$MMPROJ_GGUF")"
else
    FINAL_BACKBONE="$BACKBONE_GGUF"
    FINAL_MMPROJ="$MMPROJ_GGUF"
fi

# Verify
echo ""; echo "--- Verifying ---"
python "$SCRIPTS_DIR/05_verify_gguf.py" "$FINAL_BACKBONE" "$FINAL_MMPROJ"

echo ""
echo "========================================================"
echo " DONE"
echo " Backbone : $FINAL_BACKBONE  ($(du -sh "$FINAL_BACKBONE" | cut -f1))"
echo " mmproj   : $FINAL_MMPROJ  ($(du -sh "$FINAL_MMPROJ" | cut -f1))"
echo ""
echo " CPU inference:"
echo "   llama-cli -m $(basename "$FINAL_BACKBONE") --mmproj $(basename "$FINAL_MMPROJ") --audio-file audio.wav"
echo "========================================================"
