"""ASR quality: WITH the asr LoRA adapter vs WITHOUT (clean base), on English / Japanese / mixed (code-switch).

Uses the labeled data (ground-truth references) and reports CER (chars, lower=better), WER (words, EN),
and chrF (higher=better). Greedy decoding, whole-sequence decode. Both variants on the SAME clips.
"""
import os, re, sys, json, random, torch
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
sys.path.insert(0, "/data/lfm2.5/run-c/appsrc")
import numpy as np, soundfile as sf, sacrebleu
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from peft import PeftModel
random.seed(0)
JP="LiquidAI/LFM2.5-Audio-1.5B-JP"
ASR="/data/lfm2.5/run-a/checkpoints/lfm_lora_fleurs/step_1200"
BASE="/data/lfm2.5/run-a/data"
SP=re.compile(r"<\|[^|]*\|>")
proc=LFM2AudioProcessor.from_pretrained(JP, device="cuda").eval()
m=PeftModel.from_pretrained(LFM2AudioModel.from_pretrained(JP).eval().to("cuda"), ASR).eval()

def lev(a, b):
    dp=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        prev=dp[0]; dp[0]=i
        for j,cb in enumerate(b,1):
            cur=dp[j]; dp[j]=min(dp[j]+1, dp[j-1]+1, prev+(ca!=cb)); prev=cur
    return dp[-1]
def cer(ref,hyp): r=ref.replace(" ",""); h=hyp.replace(" ",""); return lev(r,h)/max(len(r),1)
def wer(ref,hyp): r=ref.split(); h=hyp.split(); return lev(r,h)/max(len(r),1)

def transcribe(wav, sr, use_adapter):
    wt=torch.from_numpy(np.ascontiguousarray(wav,dtype=np.float32)).unsqueeze(0)
    c=ChatState(proc); c.new_turn("system"); c.add_text("Perform ASR."); c.end_turn()
    c.new_turn("user"); c.add_audio(wt,sr); c.end_turn(); c.new_turn("assistant")
    ids=[]
    ctx=(lambda: __import__("contextlib").nullcontext()) if use_adapter else m.disable_adapter
    with torch.no_grad(), ctx():
        for x in m.generate_sequential(**c, max_new_tokens=256):
            if x.numel()==1:
                v=int(x.view(-1)[0])
                if v in (7,128): break
                ids.append(v)
    return SP.sub("", proc.text.decode(torch.tensor(ids))).strip() if ids else ""

def run(meta_rel, ref_field, label, n=20, metric="cer"):
    rows=json.loads((open(f"{BASE}/{meta_rel}", encoding="utf-8")).read())
    rows=[r for r in rows if (BASE.rsplit('/',1)[0] and os.path.exists(f"{os.path.dirname(BASE)}/{r['audio_file']}"))]
    random.shuffle(rows); rows=rows[:n]
    er_a=er_b=0.0; cf_a=[]; cf_b=[]; hyps_a=[]; refs=[]
    examples=[]
    for r in rows:
        wav,sr=sf.read(f"{os.path.dirname(BASE)}/{r['audio_file']}", dtype="float32"); wav=wav.mean(1) if wav.ndim>1 else wav
        ref=r[ref_field].strip()
        ha=transcribe(wav,sr,True); hb=transcribe(wav,sr,False)
        ea=(wer if metric=="wer" else cer)(ref,ha); eb=(wer if metric=="wer" else cer)(ref,hb)
        er_a+=ea; er_b+=eb
        cf_a.append(sacrebleu.corpus_chrf([ha],[[ref]]).score); cf_b.append(sacrebleu.corpus_chrf([hb],[[ref]]).score)
        if len(examples)<2: examples.append((ref,ha,hb))
    nn=len(rows)
    print(f"\n=== {label} (n={nn}) — {metric.upper()} lower=better, chrF higher=better ===")
    print(f"  WITH adapter : {metric.upper()}={er_a/nn*100:5.1f}%   chrF={sum(cf_a)/nn:5.1f}")
    print(f"  WITHOUT (base): {metric.upper()}={er_b/nn*100:5.1f}%   chrF={sum(cf_b)/nn:5.1f}")
    for ref,ha,hb in examples:
        print(f"   REF : {ref[:70]}")
        print(f"   ADPT: {ha[:70]}")
        print(f"   BASE: {hb[:70]}")

run("english_only/metadata/transcriptions.json","english_transcription","ENGLISH", metric="wer")
run("japanese_only/metadata/transcriptions.json","japanese_transcription","JAPANESE", metric="cer")
run("code_switching/metadata/transcriptions.json","full_transcription","MIXED (code-switch)", metric="cer")
