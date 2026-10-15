"""Full-pipeline benchmark on REAL speech (FLEURS test): transcription -> translation -> generated audio.

Per clip and direction: (1) ASR text vs human transcript (CER/WER), (2) translation vs FLEURS reference
translation (chrF), (3) generated speech transcribed by an INDEPENDENT judge (Whisper) and compared to the
translation text (chrF) — i.e. did the TTS actually say the translation. Uses the live consolidated adapters.
"""
import os, re, sys, io, argparse, torch
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
sys.path.insert(0, "/data/lfm2.5/run-c/appsrc")
import numpy as np, soundfile as sf, sacrebleu, torchaudio
from datasets import load_dataset, Audio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from peft import PeftModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

JP="LiquidAI/LFM2.5-Audio-1.5B-JP"
ASR="/data/lfm2.5/run-a/checkpoints/lfm_lora_fleurs/step_1200"
TT ="/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora_v3/final"
SP=re.compile(r"<\|[^|]*\|>")

def lev(a,b):
    dp=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        prev=dp[0]; dp[0]=i
        for j,cb in enumerate(b,1):
            cur=dp[j]; dp[j]=min(dp[j]+1,dp[j-1]+1,prev+(ca!=cb)); prev=cur
    return dp[-1]
def cer(r,h): r=r.replace(" ","");h=h.replace(" ",""); return lev(r,h)/max(len(r),1)
def wer(r,h): r=r.lower().split();h=h.lower().split(); return lev(r,h)/max(len(r),1)
def chrf(h,r): return sacrebleu.corpus_chrf([h],[[r]]).score

proc=LFM2AudioProcessor.from_pretrained(JP,device="cuda").eval()
m=PeftModel.from_pretrained(LFM2AudioModel.from_pretrained(JP).eval().to("cuda"),TT,adapter_name="default").eval()
m.load_adapter(ASR,"asr")
wp=WhisperProcessor.from_pretrained("openai/whisper-small")
wm=WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cuda").eval()

def asr(wav,sr):
    m.set_adapter("asr")
    wt=torch.from_numpy(np.ascontiguousarray(wav,dtype=np.float32)).unsqueeze(0)
    c=ChatState(proc); c.new_turn("system");c.add_text("Perform ASR.");c.end_turn()
    c.new_turn("user");c.add_audio(wt,sr);c.end_turn();c.new_turn("assistant")
    ids=[]
    with torch.no_grad():
        for x in m.generate_sequential(**c,max_new_tokens=256):
            if x.numel()==1:
                v=int(x.view(-1)[0])
                if v in (7,128):break
                ids.append(v)
    return SP.sub("",proc.text.decode(torch.tensor(ids))).strip() if ids else ""

def translate_speak(text,tgt):
    m.set_adapter("default")
    sysp="Translate to English and speak." if tgt=="en" else "Translate to Japanese and speak."
    c=ChatState(proc); c.new_turn("system");c.add_text(sysp);c.end_turn()
    c.new_turn("user");c.add_text(text);c.end_turn();c.new_turn("assistant")
    tids=[];frames=[];in_audio=False
    with torch.no_grad():
        for x in m.generate_sequential(**c,max_new_tokens=512,audio_temperature=0.8,audio_top_k=64):
            if x.numel()==1:
                v=int(x.view(-1)[0])
                if v==128:in_audio=True;continue
                if v==7:break
                if not in_audio:tids.append(v)
            elif x.numel()>1 and bool((x<2048).all()):frames.append(x)
    txt=SP.sub("",proc.text.decode(torch.tensor(tids))).strip() if tids else ""
    aud=proc.decode(torch.stack(frames,1).unsqueeze(0))[0].cpu().float().numpy() if frames else np.zeros(0)
    return txt,(aud[0] if aud.ndim>1 else aud)

def whisper(wav24,lang):
    w16=torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(wav24,dtype=np.float32)),24000,16000).numpy()
    f=wp(w16,sampling_rate=16000,return_tensors="pt").input_features.to("cuda")
    with torch.no_grad():
        ids=wm.generate(f,language=lang,task="transcribe",max_new_tokens=128)
    return wp.batch_decode(ids,skip_special_tokens=True)[0].strip()

def decode_audio(cell):
    wav,sr=sf.read(io.BytesIO(cell["bytes"]),dtype="float32") if cell.get("bytes") else sf.read(cell["path"],dtype="float32")
    return (wav.mean(1) if wav.ndim>1 else wav),sr

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--n",type=int,default=20);args=ap.parse_args()
    ja=load_dataset("google/fleurs","ja_jp",split="test").cast_column("audio",Audio(decode=False))
    en=load_dataset("google/fleurs","en_us",split="test").cast_column("audio",Audio(decode=False))
    en_by={r["id"]:r for r in en}; ja_by={r["id"]:r for r in ja}
    ids=[i for i in ja_by if i in en_by][:args.n]
    print(f"{len(ids)} real FLEURS clip pairs/direction\n",flush=True)
    for direction in ["ja2en","en2ja"]:
        st_asr=st_tr=st_au=0.0; ex=[]
        for k,i in enumerate(ids):
            if direction=="ja2en":
                aud_in,sr=decode_audio(ja_by[i]["audio"]); src_ref=ja_by[i]["transcription"]; tgt_ref=en_by[i]["transcription"]; tgt="en"
            else:
                aud_in,sr=decode_audio(en_by[i]["audio"]); src_ref=en_by[i]["transcription"]; tgt_ref=ja_by[i]["transcription"]; tgt="ja"
            t=asr(aud_in,sr)
            tr,gen=translate_speak(t,tgt)
            back=whisper(gen,tgt) if len(gen)>2400 else ""
            e_asr=(cer if direction=="ja2en" else wer)(src_ref,t)
            c_tr=chrf(tr,tgt_ref); c_au=chrf(back,tr) if tr else 0.0
            st_asr+=e_asr; st_tr+=c_tr; st_au+=c_au
            if k<3: ex.append((src_ref,t,tr,tgt_ref,back))
        n=len(ids); em="CER" if direction=="ja2en" else "WER"
        print(f"===== {direction} (real speech, n={n}) =====")
        print(f"  1) Transcription {em}: {st_asr/n*100:.1f}%   (lower=better)")
        print(f"  2) Translation chrF vs FLEURS ref: {st_tr/n:.1f}")
        print(f"  3) Generated-audio faithfulness (Whisper-back vs translation) chrF: {st_au/n:.1f}")
        for sr_,t_,tr_,ref_,bk_ in ex[:2]:
            print(f"   src_ref : {sr_[:65]}")
            print(f"   ASR     : {t_[:65]}")
            print(f"   transl. : {tr_[:65]}")
            print(f"   ref_tr  : {ref_[:65]}")
            print(f"   audio→  : {bk_[:65]}")
        print()

if __name__=="__main__":
    main()
