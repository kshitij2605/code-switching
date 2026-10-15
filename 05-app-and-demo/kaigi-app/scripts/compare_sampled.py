"""A/B test of the PRODUCTION sampling path (translate+speak audio: temperature 0.8, top_k 64).

Sampled audio is non-deterministic, so we seed the RNG identically before each call. If shared (3 adapters on one
base) and separate (dedicated tt model) then produce the SAME sampled audio, the shared base doesn't alter the
sampling path. Also runs 2 different seeds to show the audio genuinely varies (i.e. sampling is active).
"""
import os, re, sys, torch
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
sys.path.insert(0, "/data/lfm2.5/run-c/appsrc")
import numpy as np
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from peft import PeftModel
JP="LiquidAI/LFM2.5-Audio-1.5B-JP"
ASR="/data/lfm2.5/run-a/checkpoints/lfm_lora_fleurs/step_1200"
TT ="/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora_v3/final"
MIN="/data/lfm2.5/run-c/optionC/checkpoints/minutes_lora_v5/final"
SP=re.compile(r"<\|[^|]*\|>")
proc=LFM2AudioProcessor.from_pretrained(JP, device="cuda").eval()
def base(): return LFM2AudioModel.from_pretrained(JP).eval().to("cuda")
print("load shared (3 adapters) + separate tt ...", flush=True)
A=PeftModel.from_pretrained(base(), TT, adapter_name="default").eval(); A.load_adapter(ASR,"asr"); A.load_adapter(MIN,"minutes")
B=PeftModel.from_pretrained(base(), TT).eval()

def seed(s): torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def tspeak(model, text, tgt, s, set_to=None):
    if set_to: model.set_adapter(set_to)
    sysp="Translate to English and speak." if tgt=="en" else "Translate to Japanese and speak."
    c=ChatState(proc); c.new_turn("system"); c.add_text(sysp); c.end_turn()
    c.new_turn("user"); c.add_text(text); c.end_turn(); c.new_turn("assistant")
    tids=[]; frames=[]; in_audio=False
    seed(s)                       # identical RNG state right before generation
    with torch.no_grad():
        for x in model.generate_sequential(**c, max_new_tokens=512, audio_temperature=0.8, audio_top_k=64):  # PRODUCTION
            if x.numel()==1:
                v=int(x.view(-1)[0])
                if v==128: in_audio=True; continue
                if v==7: break
                if not in_audio: tids.append(v)
            elif x.numel()>1 and bool((x<2048).all()):
                frames.append(x)
    txt=SP.sub("",proc.text.decode(torch.tensor(tids))).strip()
    aud=proc.decode(torch.stack(frames,1).unsqueeze(0))[0].cpu().float().numpy() if frames else np.zeros(0)
    return txt, (aud[0] if aud.ndim>1 else aud)

for text,tgt in [("来週のミーティングのアジェンダを共有します。","en"),("Let's confirm the budget before Friday's release.","ja")]:
    ta,aa=tspeak(A,text,tgt,1234,"default"); tb,ab=tspeak(B,text,tgt,1234)
    n=min(len(aa),len(ab)); d=float(np.max(np.abs(aa[:n]-ab[:n]))) if n else -1
    # second seed on shared to show audio actually varies with sampling
    ta2,aa2=tspeak(A,text,tgt,9999,"default"); n2=min(len(aa),len(aa2)); dvar=float(np.max(np.abs(aa[:n2]-aa2[:n2]))) if n2 else -1
    print(f"[{tgt}] same-seed A-vs-B: TEXT_MATCH={ta==tb} len A={len(aa)} B={len(ab)} max|Δ|={d:.2e} | diff-seed varies: max|Δ|={dvar:.2e}")
    if ta!=tb: print(f"   A:{ta}\n   B:{tb}")
