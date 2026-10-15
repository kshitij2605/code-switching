"""A/B test: 3 LoRA adapters on ONE shared base (set_adapter switching) vs 3 SEPARATE model instances.

Greedy/deterministic decoding everywhere (text: argmax; audio: top_k=1) so outputs are reproducible and any
A-vs-B difference is attributable to the shared-base multi-adapter setup, not sampling noise. Compares ASR text,
translation text + audio waveform, and minutes text — plus latency.
"""
import os, re, sys, time, torch
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
sys.path.insert(0, "/data/lfm2.5/run-c/appsrc")
import numpy as np, soundfile as sf
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from peft import PeftModel
from csmeeting.minutes.prompt import SYSTEM_PROMPT

JP="LiquidAI/LFM2.5-Audio-1.5B-JP"
ASR="/data/lfm2.5/run-a/checkpoints/lfm_lora_fleurs/step_1200"
TT ="/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora_v3/final"
MIN="/data/lfm2.5/run-c/optionC/checkpoints/minutes_lora_v5/final"
SP=re.compile(r"<\|[^|]*\|>")
proc=LFM2AudioProcessor.from_pretrained(JP, device="cuda").eval()

def load(path, name="default"):
    b=LFM2AudioModel.from_pretrained(JP).eval().to("cuda")
    return PeftModel.from_pretrained(b, path, adapter_name=name).eval()

print("loading shared (A) ...", flush=True)
A=load(TT, "default"); A.load_adapter(ASR,"asr"); A.load_adapter(MIN,"minutes")
print("loading separate (B) ...", flush=True)
B_asr=load(ASR); B_tt=load(TT); B_min=load(MIN)

def asr(model, wav, sr, set_to=None):
    if set_to: model.set_adapter(set_to)
    wt=torch.from_numpy(np.ascontiguousarray(wav,dtype=np.float32)).unsqueeze(0)
    c=ChatState(proc); c.new_turn("system"); c.add_text("Perform ASR."); c.end_turn()
    c.new_turn("user"); c.add_audio(wt,sr); c.end_turn(); c.new_turn("assistant")
    ids=[]; t=time.time()
    with torch.no_grad():
        for x in model.generate_sequential(**c,max_new_tokens=256):
            if x.numel()==1:
                v=int(x.view(-1)[0])
                if v in (7,128): break
                ids.append(v)
    return SP.sub("",proc.text.decode(torch.tensor(ids))).strip(), time.time()-t

def translate(model, text, tgt, set_to=None):
    if set_to: model.set_adapter(set_to)
    sysp="Translate to English and speak." if tgt=="en" else "Translate to Japanese and speak."
    c=ChatState(proc); c.new_turn("system"); c.add_text(sysp); c.end_turn()
    c.new_turn("user"); c.add_text(text); c.end_turn(); c.new_turn("assistant")
    tids=[]; frames=[]; in_audio=False; t=time.time()
    with torch.no_grad():
        for x in model.generate_sequential(**c,max_new_tokens=512,audio_temperature=1.0,audio_top_k=1):  # greedy audio
            if x.numel()==1:
                v=int(x.view(-1)[0])
                if v==128: in_audio=True; continue
                if v==7: break
                if not in_audio: tids.append(v)
            elif x.numel()>1 and bool((x<2048).all()):
                frames.append(x)
    txt=SP.sub("",proc.text.decode(torch.tensor(tids))).strip()
    aud=proc.decode(torch.stack(frames,1).unsqueeze(0))[0].cpu().float().numpy() if frames else np.zeros(0)
    if aud.ndim>1: aud=aud[0]
    return txt, aud, time.time()-t

def minutes(model, tr, set_to=None):
    if set_to: model.set_adapter(set_to)
    c=ChatState(proc); c.new_turn("system"); c.add_text(SYSTEM_PROMPT); c.end_turn()
    c.new_turn("user"); c.add_text(tr); c.end_turn(); c.new_turn("assistant")
    ids=[]; t=time.time()
    with torch.no_grad():
        for x in model.generate_sequential(**c,max_new_tokens=768):
            if x.numel()==1:
                v=int(x.view(-1)[0])
                if v in (7,128): break
                ids.append(v)
    return SP.sub("",proc.text.decode(torch.tensor(ids))).strip(), time.time()-t

def load_wav(p):
    w,sr=sf.read(p,dtype="float32"); return (w.mean(1) if w.ndim>1 else w), sr

TR="\n".join(["今日のmeetingのagendaをshareします。","田中さん、データの確認をお願いできますか。","はい、明日までに確認します。","鈴木さんにAPIの修正をお願いします。"])
print("\n================= ASR =================")
for p in ["runs/ja_clip.wav","runs/en_clip.wav"]:
    w,sr=load_wav("/data/lfm2.5/run-c/"+p)
    a,ta=asr(A,w,sr,"asr"); b,tb=asr(B_asr,w,sr)
    print(f"{p}: MATCH={a==b}  ({ta*1000:.0f}ms shared / {tb*1000:.0f}ms sep)")
    if a!=b: print(f"  A: {a}\n  B: {b}")
    else: print(f"  = {a}")

print("\n============= TRANSLATE+SPEAK =============")
for text,tgt in [("来週のミーティングのアジェンダを共有します。","en"),("Let's confirm the budget before Friday's release.","ja")]:
    a,aa,ta=translate(A,text,tgt,"default"); b,ab,tb=translate(B_tt,text,tgt)
    n=min(len(aa),len(ab)); dif=float(np.max(np.abs(aa[:n]-ab[:n]))) if n else -1
    print(f"[{tgt}] TEXT_MATCH={a==b}  audio_len A={len(aa)} B={len(ab)} max|Δ|={dif:.2e}  ({ta*1000:.0f}/{tb*1000:.0f}ms)")
    if a!=b: print(f"  A: {a}\n  B: {b}")
    else: print(f"  = {a}")

print("\n================ MINUTES ================")
a,ta=minutes(A,TR,"minutes"); b,tb=minutes(B_min,TR)
print(f"MATCH={a==b}  ({ta*1000:.0f}ms shared / {tb*1000:.0f}ms sep)")
if a!=b:
    print("--- A (shared) ---\n"+a+"\n--- B (separate) ---\n"+b)
else:
    print(a)
