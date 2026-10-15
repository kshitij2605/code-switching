"""
LFM2.5-Audio-1.5B-JP — live ASR demo (LoRA fine-tune, step 2500).
Loads base model + PEFT LoRA adapter and serves a Gradio UI.

Usage:
  CUDA_VISIBLE_DEVICES=0 python serve_gradio_lora.py
"""
import os, numpy as np, torch
import soundfile as sf
import gradio as gr
from peft import PeftModel

BASE_MODEL  = "LiquidAI/LFM2.5-Audio-1.5B-JP"
LORA_CKPT   = "/data/lfm2.5/run-a/checkpoints/lfm_lora/final"
DEVICE      = "cuda:0"
IM_END      = 7
MAX_TOK     = 300

TASKS = {
    "Native  (preserve code-switching)": "Transcribe the audio.",
    "Force Japanese":                     "Transcribe in Japanese.",
    "Force English":                      "Transcribe in English.",
}

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading processor ...")
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
proc  = LFM2AudioProcessor.from_pretrained(BASE_MODEL)

print("Loading base model ...")
base  = LFM2AudioModel.from_pretrained(BASE_MODEL, device=DEVICE, dtype=torch.bfloat16)

print(f"Applying LoRA adapter from {LORA_CKPT} ...")
model = PeftModel.from_pretrained(base, LORA_CKPT)
model.eval()
print("Model ready.\n")


def transcribe(audio, task_label: str) -> str:
    if audio is None:
        return "No audio received — please record something first."

    sr, arr = audio
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if np.abs(arr).max() > 1.0:
        arr = arr / 32768.0

    wave   = torch.tensor(arr).unsqueeze(0)
    prompt = TASKS[task_label]

    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(prompt);    chat.end_turn()
    chat.new_turn("user");   chat.add_audio(wave, sr); chat.end_turn()
    chat.new_turn("assistant")

    toks = []
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=MAX_TOK):
            if t.numel() == 1:
                tid = int(t.reshape(-1)[0].item())
                if tid == IM_END:
                    break
                toks.append(tid)

    return proc.text.decode(toks, skip_special_tokens=True).strip() or "(empty)"


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="LFM2.5 ASR — LoRA step 2500") as demo:
    gr.Markdown(
        "## LFM2.5-Audio-1.5B-JP  |  LoRA fine-tune (step 2500)\n"
        "GPU: H100 NVL  |  No GGUF — full bfloat16 weights"
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
            task_sel = gr.Radio(choices=list(TASKS.keys()), value=list(TASKS.keys())[0], label="Task")
            btn      = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Transcription", lines=6)

    btn.click(fn=transcribe, inputs=[audio_in, task_sel], outputs=out)
    audio_in.stop_recording(fn=transcribe, inputs=[audio_in, task_sel], outputs=out)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssl_certfile="/data/lfm2.5/run-a/server.crt",
    ssl_keyfile="/data/lfm2.5/run-a/server.key",
    ssl_verify=False,
    show_error=True,
)
