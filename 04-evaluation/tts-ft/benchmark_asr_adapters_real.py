"""ASR adapter comparison on REAL speech: lfm_lora_fleurs/step_1200 vs lfm_lora/final vs clean base.

Datasets: FLEURS test (news) and Common Voice 17 (conversational, varied speakers) — English (WER) + Japanese (CER).
Robust audio decode (wav/flac via soundfile, mp3 via torchaudio/temp). Greedy, whole-seq decode.
"""
import os, re, sys, io, tempfile, torch
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
sys.path.insert(0, "/data/lfm2.5/run-c/appsrc")
import numpy as np, soundfile as sf, torchaudio
from datasets import load_dataset, Audio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from peft import PeftModel
JP="LiquidAI/LFM2.5-Audio-1.5B-JP"
A_NEW="/data/lfm2.5/run-a/checkpoints/lfm_lora_fleurs/step_1200"
A_OLD="/data/lfm2.5/run-a/checkpoints/lfm_lora/final"
SP=re.compile(r"<\|[^|]*\|>")
def lev(a,b):
    dp=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        prev=dp[0];dp[0]=i
        for j,cb in enumerate(b,1):
            cur=dp[j];dp[j]=min(dp[j]+1,dp[j-1]+1,prev+(ca!=cb));prev=cur
    return dp[-1]
def cer(r,h): r=r.replace(" ","");h=h.replace(" ",""); return lev(r,h)/max(len(r),1)
def wer(r,h): r=r.lower().split();h=h.lower().split(); return lev(r,h)/max(len(r),1)
norm=lambda s: re.sub(r"[、。,.!?！？\"']","",s).strip()

proc=LFM2AudioProcessor.from_pretrained(JP,device="cuda").eval()
m=PeftModel.from_pretrained(LFM2AudioModel.from_pretrained(JP).eval().to("cuda"),A_NEW,adapter_name="new").eval()
m.load_adapter(A_OLD,"old")

def asr(wav,sr,which):
    import contextlib
    ctx=m.disable_adapter if which=="base" else (lambda: contextlib.nullcontext())
    if which!="base": m.set_adapter(which)
    wt=torch.from_numpy(np.ascontiguousarray(wav,dtype=np.float32)).unsqueeze(0)
    c=ChatState(proc);c.new_turn("system");c.add_text("Perform ASR.");c.end_turn()
    c.new_turn("user");c.add_audio(wt,sr);c.end_turn();c.new_turn("assistant")
    ids=[]
    with torch.no_grad(), ctx():
        for x in m.generate_sequential(**c,max_new_tokens=200):
            if x.numel()==1:
                v=int(x.view(-1)[0])
                if v in (7,128):break
                ids.append(v)
    return SP.sub("",proc.text.decode(torch.tensor(ids))).strip() if ids else ""

def decode_any(cell):
    b=cell.get("bytes"); p=cell.get("path")
    try:
        wav,sr=sf.read(io.BytesIO(b) if b else p,dtype="float32"); return (wav.mean(1) if wav.ndim>1 else wav),sr
    except Exception:
        try:
            wav,sr=torchaudio.load(io.BytesIO(b)); return wav.mean(0).numpy(),sr
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".mp3",delete=False) as f: f.write(b); tp=f.name
            wav,sr=torchaudio.load(tp); os.unlink(tp); return wav.mean(0).numpy(),sr

def bench(name, loader, ref_field, lang, n=15):
    try:
        ds=loader()
    except Exception as e:
        print(f"\n[{name}] SKIP ({type(e).__name__}: {str(e)[:80]})"); return
    metric=wer if lang=="en" else cer
    tot={"new":0.0,"old":0.0,"base":0.0}; cnt=0; ex=[]
    for r in ds:
        ref=norm(r[ref_field]);
        if not ref: continue
        wav,sr=decode_any(r["audio"])
        if len(wav)<sr*0.5: continue
        h={w:norm(asr(wav,sr,w)) for w in ["new","old","base"]}
        for w in h: tot[w]+=metric(ref,h[w])
        if len(ex)<2: ex.append((ref,h))
        cnt+=1
        if cnt>=n: break
    if not cnt: print(f"\n[{name}] no clips"); return
    M="WER" if lang=="en" else "CER"
    print(f"\n===== {name} ({lang}, n={cnt}) — {M} %, lower=better =====")
    print(f"  fleurs-adapter(new): {tot['new']/cnt*100:5.1f}   old-adapter: {tot['old']/cnt*100:5.1f}   base: {tot['base']/cnt*100:5.1f}")
    for ref,h in ex:
        print(f"   REF: {ref[:62]}")
        print(f"   new: {h['new'][:62]}  | old: {h['old'][:62]}")

# FLEURS (news)
bench("FLEURS-en", lambda: load_dataset("google/fleurs","en_us",split="test").cast_column("audio",Audio(decode=False)).select(range(40)), "transcription","en")
bench("FLEURS-ja", lambda: load_dataset("google/fleurs","ja_jp",split="test").cast_column("audio",Audio(decode=False)).select(range(40)), "transcription","ja")
# Common Voice (conversational, varied speakers)
bench("CommonVoice-en", lambda: load_dataset("fsicoli/common_voice_17_0","en",split="test",streaming=True).cast_column("audio",Audio(decode=False)), "sentence","en")
bench("CommonVoice-ja", lambda: load_dataset("fsicoli/common_voice_17_0","ja",split="test",streaming=True).cast_column("audio",Audio(decode=False)), "sentence","ja")
