"""
LFM2.5-Audio-1.5B-JP — live ASR demo (checkpoint_6, step 1400).
Serves a Gradio UI with microphone recording on a single GPU.

Usage:
  CUDA_VISIBLE_DEVICES=0 python serve_gradio.py
"""
import os, numpy as np, torch, soundfile as sf
from safetensors.torch import load_file
import gradio as gr

BASE_MODEL = "LiquidAI/LFM2.5-Audio-1.5B-JP"
CKPT = "/data/lfm2.5/run-a/checkpoints/lfm_asr/checkpoints/checkpoint_6/model.safetensors"
DEVICE = "cuda:0"
IM_END = 7
MAX_TOK = 300

TASKS = {
    "🌐  Native  (preserve code-switching)": "Transcribe the audio.",
    "🇯🇵  Force Japanese                  ": "Transcribe in Japanese.",
    "🇬🇧  Force English                   ": "Transcribe in English.",
}

# ── load model once at startup ──────────────────────────────────────────────
print("Loading processor …")
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
proc  = LFM2AudioProcessor.from_pretrained(BASE_MODEL)

print("Loading model weights …")
model = LFM2AudioModel.from_pretrained(BASE_MODEL, device=DEVICE, dtype=torch.bfloat16)
sd    = load_file(CKPT, device=DEVICE)
miss, unex = model.load_state_dict(sd, strict=False)
print(f"  checkpoint_6 loaded  missing={len(miss)}  unexpected={len(unex)}")
model.eval()
print("Model ready.\n")


def transcribe(audio, task_label: str) -> str:
    if audio is None:
        return "No audio received — please upload an audio file first."

    sr, arr = audio                         # Gradio returns (sample_rate, ndarray)
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if np.abs(arr).max() > 1.0:            # int16 PCM → float32 [-1, 1]
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
CSS = """
#title  { text-align: center; }
#output { font-size: 1.25em; }
"""

with gr.Blocks(title="LFM2.5 ASR Demo") as demo:
    gr.Markdown(
        "# LFM2.5-Audio-1.5B-JP — Live ASR Demo\n"
        "**Model:** checkpoint_6 (step 1400, best val loss)  |  "
        "**GPU:** single H100 NVL",
        elem_id="title",
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["upload"],
                type="numpy",
                label="Upload Audio File",
            )
            task_sel = gr.Radio(
                choices=list(TASKS.keys()),
                value=list(TASKS.keys())[0],
                label="Task",
            )
            btn = gr.Button("▶  Transcribe", variant="primary", size="lg")

        with gr.Column(scale=1):
            out = gr.Textbox(
                label="📝  Transcription",
                lines=6,
                elem_id="output",
            )
            gr.Markdown(
                "**Tips:**\n"
                "- *Native*: keeps English words in Latin script (code-switching)\n"
                "- *Force Japanese*: rewrites everything in Japanese (katakana/kanji)\n"
                "- *Force English*: translates/transcribes everything in English"
            )

    btn.click(fn=transcribe, inputs=[audio_in, task_sel], outputs=out)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssl_certfile="/data/lfm2.5/run-a/server.crt",
    ssl_keyfile="/data/lfm2.5/run-a/server.key",
    ssl_verify=False,
    show_error=True,
)
