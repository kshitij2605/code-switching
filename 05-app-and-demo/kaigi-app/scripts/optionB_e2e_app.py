"""End-to-end app integration test: Assistant with the translating-TTS adapter (v3).

Verifies the wired path: ASR (adapter disabled) -> translate_speak_stream (adapter on) -> translated text + audio,
both directions, plus that ASR still works and minutes still run. Uses pre-made JA/EN clips in runs/.
"""
import os, sys
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
sys.path.insert(0, "/data/lfm2.5/run-c/appsrc")
import numpy as np, soundfile as sf
from csmeeting.live.config import LiveConfig
from csmeeting.app.assistant import Assistant

V3 = "/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora_v3/final"
RUNS = "/data/lfm2.5/run-c/runs"

cfg = LiveConfig(device="cuda", translating_tts_adapter=V3, load_text_model=True)
a = Assistant(cfg=cfg)
print("Assistant up. use_audio_translator =", a.use_audio_translator, flush=True)


def load(p):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    return w, sr


for clip, direction in [(f"{RUNS}/ja_clip.wav", "ja2en"), (f"{RUNS}/en_clip.wav", "en2ja")]:
    wav, sr = load(clip)
    src_text, translation, audio, out_sr, src, tgt = a.process(wav, sr, direction)
    dur = len(audio) / out_sr if audio is not None else 0
    print(f"\n[{direction}] {src}->{tgt}")
    print(f"  ASR src_text : {src_text}")
    print(f"  translation  : {translation}")
    print(f"  audio        : {dur:.1f}s @ {out_sr}")
    sf.write(f"{RUNS}/e2e_{direction}.wav", audio, out_sr)

# auto-direction on the JA clip
wav, sr = load(f"{RUNS}/ja_clip.wav")
st, tr, au, osr, s, t = a.process(wav, sr, "auto")
print(f"\n[auto] detected {s}->{t}  src='{st[:40]}'  tr='{tr[:50]}'")

print("\n[minutes]\n" + a.minutes()[:300])
print("\nE2E OK")
