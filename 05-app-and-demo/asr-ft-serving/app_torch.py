"""
LFM2.5-Audio-1.5B-JP -- Local ASR Web App (PyTorch / liquid_audio backend)
"""
from __future__ import annotations
import io, tempfile, os
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torchaudio
import soundfile as sf
import numpy as np
from peft import PeftModel
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_MODEL = "LiquidAI/LFM2.5-Audio-1.5B-JP"
LORA_CKPT  = "/data/lfm2.5/run-a/checkpoints/lfm_lora_fleurs/step_1200"
DEVICE     = "cuda:0"
IM_END     = 7
MAX_TOK    = 300
TARGET_SR  = 16000
STATIC_DIR = Path(__file__).parent / "static"
# ffmpeg found in cotomi's vllm overlay
FFMPEG     = "/home/cotomi/vllm_cotomi/0cb98957e176fe94734442885fa8bd4c4bb00345701a73291d4e0dd3d0611ed9/usr/bin/ffmpeg"

model = None
proc  = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, proc
    print("Loading processor ...")
    proc = LFM2AudioProcessor.from_pretrained(BASE_MODEL)
    print("Loading base model ...")
    base = LFM2AudioModel.from_pretrained(BASE_MODEL, device=DEVICE, dtype=torch.bfloat16)
    print(f"Applying LoRA from {LORA_CKPT} ...")
    model = PeftModel.from_pretrained(base, LORA_CKPT)
    model.eval()
    print("Model ready.")
    yield


app = FastAPI(title="LFM2.5 ASR (torch)", lifespan=lifespan)


def load_audio(data: bytes) -> tuple[torch.Tensor, int]:
    """Load audio bytes -> 16 kHz mono tensor: soundfile first, ffmpeg as fallback."""
    try:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        waveform = torch.from_numpy(arr).unsqueeze(0)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        return waveform, TARGET_SR
    except Exception:
        pass  # not a format soundfile reads (e.g. webm from the browser) -> ffmpeg

    # Fallback: write to temp file, convert via ffmpeg
    import subprocess
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
        f.write(data); in_path = f.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", in_path, "-ar", str(TARGET_SR), "-ac", "1", "-f", "wav", out_path],
            capture_output=True, check=True,
        )
        arr, sr = sf.read(out_path, dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return torch.from_numpy(arr).unsqueeze(0), TARGET_SR
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), task: str = "Perform ASR."):
    try:
        wave, sr = load_audio(await audio.read())

        chat = ChatState(proc)
        chat.new_turn("system"); chat.add_text(task);        chat.end_turn()
        chat.new_turn("user");   chat.add_audio(wave, sr);   chat.end_turn()
        chat.new_turn("assistant")

        toks = []
        with torch.no_grad():
            for t in model.generate_sequential(**chat, max_new_tokens=MAX_TOK):
                if t.numel() == 1:
                    tid = int(t.reshape(-1)[0].item())
                    if tid == IM_END:
                        break
                    toks.append(tid)

        text = proc.text.decode(toks, skip_special_tokens=True).strip() or "(empty)"
        return JSONResponse({"text": text, "task": task})
    except Exception as exc:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
