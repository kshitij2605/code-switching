"""Compare base vs v1 (600-pair) vs v2 (2200-pair + Kokoro) translating-TTS on NEUTRAL FLEURS text.

FLEURS is never in training, so this is a fair test of generalization. Metrics per direction:
  - translate chrF vs FLEURS reference
  - audio intelligibility: round-trip ASR (EN-base judges English audio, clean JP judges Japanese audio).
"""
import os, re, json, argparse
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
import torch, sacrebleu, numpy as np
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

DEV = "cuda"; SR = 24000
SPECIAL = re.compile(r"<\|[^|]*\|>")
JP_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
EN_ID = "LiquidAI/LFM2.5-Audio-1.5B"
FLEURS = "/data/lfm2.5/run-c/data/mt/fleurs_ja_en_test.jsonl"
V1 = "/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora/final"
V2 = "/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora_v2/final"
V3 = "/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora_v3/final"
V4 = "/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora_v4/final"
OUTDIR = "/data/lfm2.5/run-c/runs/optionB_eval_v4"
os.makedirs(OUTDIR, exist_ok=True)
SYS = {"ja2en": "Translate to English and speak.", "en2ja": "Translate to Japanese and speak."}


def load(mid):
    p = LFM2AudioProcessor.from_pretrained(mid, device=DEV).eval()
    m = LFM2AudioModel.from_pretrained(mid).eval().to(DEV)
    return p, m


def merged(adapter):
    from peft import PeftModel
    m = LFM2AudioModel.from_pretrained(JP_ID).eval()
    m = PeftModel.from_pretrained(m, adapter)
    return m.merge_and_unload().to(DEV).eval()


def translate_and_speak(proc, model, system, text):
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(system); chat.end_turn()
    chat.new_turn("user"); chat.add_text(text); chat.end_turn()
    chat.new_turn("assistant")
    txt, frames, in_audio = [], [], False
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=1024, audio_temperature=0.8, audio_top_k=64):
            if t.numel() == 1:
                tid = int(t.view(-1)[0])
                if tid == 128: in_audio = True; continue
                if tid == 7: break
                if not in_audio: txt.append(proc.text.decode(t))
            elif t.numel() > 1 and bool((t < 2048).all()):
                frames.append(t)
    audio = None
    if frames:
        wav = proc.decode(torch.stack(frames, 1).unsqueeze(0))[0].detach().cpu().float().numpy()
        audio = wav[0] if wav.ndim > 1 else wav
    return SPECIAL.sub("", "".join(txt)).strip(), audio


def asr(proc, model, system, wav):
    wt = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(system); chat.end_turn()
    chat.new_turn("user"); chat.add_audio(wt, SR); chat.end_turn()
    chat.new_turn("assistant")
    out = []
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=256):
            if t.numel() == 1: out.append(proc.text.decode(t))
    return SPECIAL.sub("", "".join(out)).strip()


def chrf(h, r): return round(sacrebleu.corpus_chrf([h], [[r]]).score, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); args = ap.parse_args()
    pairs = []
    seen = set()
    for line in open(FLEURS, encoding="utf-8"):
        r = json.loads(line)
        if r["ja"] in seen: continue
        seen.add(r["ja"]); pairs.append((r["ja"], r["en"]))
        if len(pairs) >= args.n: break

    jp_p, jp_base = load(JP_ID)      # base eval + JA-ASR judge
    en_p, en_judge = load(EN_ID)     # EN-ASR judge
    v3 = merged(V3); v4 = merged(V4)
    print("loaded base/en-judge/v3/v4", flush=True)
    variants = {"base": jp_base, "v3": v3, "v4_real": v4}

    report = {}
    for d in ("ja2en", "en2ja"):
        tlang = "en" if d == "ja2en" else "ja"
        jp_, jm_, js_ = (en_p, en_judge, "Perform ASR.") if tlang == "en" else (jp_p, jp_base, "Perform ASR in japanese.")
        agg = {k: {"tr": [], "rt": []} for k in variants}
        examples = []
        for ja, en in pairs:
            src, ref = (ja, en) if d == "ja2en" else (en, ja)
            row = {"in": src[:60], "ref": ref[:60]}
            for name, mdl in variants.items():
                txt, audio = translate_and_speak(jp_p, mdl, SYS[d], src)
                tr = chrf(txt, ref); agg[name]["tr"].append(tr)
                if audio is not None and len(audio) > SR // 4:
                    back = asr(jp_, jm_, js_, audio); rt = chrf(back, txt) if txt else 0.0
                else:
                    rt = 0.0
                agg[name]["rt"].append(rt)
                row[f"{name}_txt"] = txt[:55]; row[f"{name}_tr"] = tr; row[f"{name}_rt"] = rt
            examples.append(row)
        def a(x): return round(sum(x)/max(len(x),1), 1)
        report[d] = {k: {"translate_chrF": a(v["tr"]), "audio_rt_chrF": a(v["rt"])} for k, v in agg.items()}
        report[d]["examples"] = examples[:4]
        print(f"\n=== {d} (target audio={tlang}, n={len(pairs)}) ===")
        for k in variants:
            print(f"  {k:10s}: translate {report[d][k]['translate_chrF']:5.1f}   audio-RT {report[d][k]['audio_rt_chrF']:5.1f}")

    with open(f"{OUTDIR}/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {OUTDIR}/report.json")


if __name__ == "__main__":
    main()
