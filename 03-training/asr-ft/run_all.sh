#!/usr/bin/env bash
set -e

BASE="/data/lfm2.5/run-a"
cd "$BASE"

echo "========================================"
echo " LFM2.5 Dataset Generator"
echo " Code-switching: 500 | English: 300 | Japanese: 300"
echo "========================================"

# ── 1. Set up virtual environment with uv ────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating virtual environment with uv..."
    uv venv .venv --python 3.11
else
    echo "[1/5] Virtual environment already exists, skipping."
fi

source .venv/bin/activate

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo "[2/5] Installing dependencies..."
uv pip install edge-tts imageio-ffmpeg

# ── 3. Create output directories ──────────────────────────────────────────────
echo "[3/5] Creating output directories..."
mkdir -p data/code_switching/{audio,metadata}
mkdir -p data/english_only/{audio,metadata}
mkdir -p data/japanese_only/{audio,metadata}

# ── 4. Generate datasets ───────────────────────────────────────────────────────
echo "[4/5] Generating datasets..."

echo ""
echo "--- Code-switching (500 samples) ---"
python generate_cs_data.py

echo ""
echo "--- English-only (300 samples) ---"
python generate_en_data.py

echo ""
echo "--- Japanese-only (300 samples) ---"
python generate_ja_data.py

# ── 5. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Done! Summary:"
echo "  Code-switching audio : $(ls data/code_switching/audio/*.wav 2>/dev/null | wc -l) files"
echo "  English-only audio   : $(ls data/english_only/audio/*.wav   2>/dev/null | wc -l) files"
echo "  Japanese-only audio  : $(ls data/japanese_only/audio/*.wav  2>/dev/null | wc -l) files"
echo ""
echo "  Metadata:"
echo "    data/code_switching/metadata/transcriptions.json"
echo "    data/english_only/metadata/transcriptions.json"
echo "    data/japanese_only/metadata/transcriptions.json"
echo ""
echo "Total disk usage:"
du -sh data/
