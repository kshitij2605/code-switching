"""Verify the integration mechanism before editing the app:
  1. PeftModel(JP-Audio, v3) can call generate_sequential (translate-speak) with adapter ON.
  2. JA ASR still works with adapter DISABLED (disable_adapter context) — so ONE model serves both roles.
  3. Compare JA ASR adapter-on vs off, to decide whether toggling is needed.
"""
import os, re
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
import torch, numpy as np, soundfile as sf
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from peft import PeftModel

DEV = "cuda"; SR = 24000
SPECIAL = re.compile(r"<\|[^|]*\|>")
JP = "LiquidAI/LFM2.5-Audio-1.5B-JP"
V3 = "/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora_v3/final"
JA_CLIP = "/data/lfm2.5/run-c/runs/ja_clip.wav"

proc = LFM2AudioProcessor.from_pretrained(JP, device=DEV).eval()
base = LFM2AudioModel.from_pretrained(JP).eval().to(DEV)
model = PeftModel.from_pretrained(base, V3).eval()
print("PeftModel wrapped. has generate_sequential:", hasattr(model, "generate_sequential"), flush=True)


def translate_speak(text, tgt):
    sysmsg = "Translate to English and speak." if tgt == "en" else "Translate to Japanese and speak."
    c = ChatState(proc); c.new_turn("system"); c.add_text(sysmsg); c.end_turn()
    c.new_turn("user"); c.add_text(text); c.end_turn(); c.new_turn("assistant")
    txt, frames, in_audio = [], [], False
    with torch.no_grad():
        for t in model.generate_sequential(**c, max_new_tokens=1024, audio_temperature=0.8, audio_top_k=64):
            if t.numel() == 1:
                tid = int(t.view(-1)[0])
                if tid == 128: in_audio = True; continue
                if tid == 7: break
                if not in_audio: txt.append(proc.text.decode(t))
            elif t.numel() > 1 and bool((t < 2048).all()):
                frames.append(t)
    audio = None
    if frames:
        w = proc.decode(torch.stack(frames, 1).unsqueeze(0))[0].cpu().float().numpy(); audio = w[0] if w.ndim > 1 else w
    return SPECIAL.sub("", "".join(txt)).strip(), audio


def asr(wav, adapter_on):
    c = ChatState(proc); c.new_turn("system"); c.add_text("Perform ASR in japanese."); c.end_turn()
    wt = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    c.new_turn("user"); c.add_audio(wt, SR); c.end_turn(); c.new_turn("assistant")
    out = []
    ctx = (lambda: __import__("contextlib").nullcontext()) if adapter_on else model.disable_adapter
    with torch.no_grad(), ctx():
        for t in model.generate_sequential(**c, max_new_tokens=256):
            if t.numel() == 1: out.append(proc.text.decode(t))
    return SPECIAL.sub("", "".join(out)).strip()


# 1+2: translate-speak (adapter on)
txt, audio = translate_speak("来週のミーティングのアジェンダを共有します。", "en")
print(f"\n[translate-speak ja->en] text='{txt}'  audio={'%.1fs'%(len(audio)/SR) if audio is not None else None}", flush=True)
txt2, audio2 = translate_speak("Let's confirm the budget before Friday.", "ja")
print(f"[translate-speak en->ja] text='{txt2}'  audio={'%.1fs'%(len(audio2)/SR) if audio2 is not None else None}", flush=True)

# 3: JA ASR adapter off vs on
wav, sr = sf.read(JA_CLIP, dtype="float32")
if sr != SR: import torchaudio; wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
print(f"\n[JA ASR adapter OFF] {asr(wav, False)}", flush=True)
print(f"[JA ASR adapter ON ] {asr(wav, True)}", flush=True)
print("\nOK")
