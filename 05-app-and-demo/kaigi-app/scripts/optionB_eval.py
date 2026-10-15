"""Option B eval: did the translating-TTS LoRA improve JP-Audio on translate + cross-lingual TTS?

For each eval sentence and each variant (base JP-Audio vs LoRA-merged), run the translating-TTS task:
  generate_sequential -> collect translated TEXT (until <|audio_start|>), then AUDIO frames.
Metrics:
  - translation chrF vs the reference target_text
  - audio intelligibility: round-trip ASR (EN-base judges English audio, clean JP judges Japanese audio),
    chrF(asr_back, produced_text). Higher = clearer speech in the target language.
"""
import os, re, json, argparse
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
import torch, sacrebleu, numpy as np
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

DEV = "cuda"; SR = 24000
SPECIAL = re.compile(r"<\|[^|]*\|>")
JP_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
EN_ID = "LiquidAI/LFM2.5-Audio-1.5B"
EVAL = "/data/lfm2.5/run-c/optionB/data/training/eval.jsonl"
ADAPTER = "/data/lfm2.5/run-c/optionB/checkpoints/transtts_lora/final"
OUTDIR = "/data/lfm2.5/run-c/runs/optionB_eval"
os.makedirs(OUTDIR, exist_ok=True)


def load(mid):
    p = LFM2AudioProcessor.from_pretrained(mid, device=DEV).eval()
    m = LFM2AudioModel.from_pretrained(mid).eval().to(DEV)
    return p, m


def translate_and_speak(proc, model, system, text):
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(system); chat.end_turn()
    chat.new_turn("user"); chat.add_text(text); chat.end_turn()
    chat.new_turn("assistant")
    txt, frames, in_audio = [], [], False
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=1024,
                                            audio_temperature=0.8, audio_top_k=64):
            if t.numel() == 1:
                tid = int(t.view(-1)[0])
                if tid == 128:   # <|audio_start|> -> switch to audio
                    in_audio = True; continue
                if tid == 7:     # <|im_end|>
                    break
                if not in_audio:
                    txt.append(proc.text.decode(t))
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
            if t.numel() == 1:
                out.append(proc.text.decode(t))
    return SPECIAL.sub("", "".join(out)).strip()


def chrf(h, r):
    return round(sacrebleu.corpus_chrf([h], [[r]]).score, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_dir", type=int, default=15)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    by_dir = {"ja2en": [], "en2ja": []}
    for r in rows:
        by_dir[r["direction"]].append(r)
    for d in by_dir:
        by_dir[d] = by_dir[d][:args.n_per_dir]

    jp_p, jp_judge = load(JP_ID)         # clean JP: base translate eval + JA-ASR judge
    en_p, en_judge = load(EN_ID)         # EN-ASR judge
    # tuned = base JP + merged adapter
    from peft import PeftModel
    tuned = LFM2AudioModel.from_pretrained(JP_ID).eval()
    tuned = PeftModel.from_pretrained(tuned, ADAPTER)
    tuned = tuned.merge_and_unload().to(DEV).eval()
    print("models loaded (jp clean, en judge, jp tuned)", flush=True)

    ASR_EN = "Perform ASR."
    ASR_JA = "Perform ASR in japanese."
    report = {}
    for d, items in by_dir.items():
        tgt_audio_lang = "en" if d == "ja2en" else "ja"
        judge_p, judge_m, judge_sys = (en_p, en_judge, ASR_EN) if tgt_audio_lang == "en" else (jp_p, jp_judge, ASR_JA)
        agg = {"base": {"tr": [], "rt": []}, "tuned": {"tr": [], "rt": []}}
        examples = []
        for it in items:
            sysmsg, src, ref = it["system"], it["input_text"], it["target_text"]
            row = {"in": src, "ref": ref}
            for name, (pp, mm) in {"base": (jp_p, jp_judge), "tuned": (jp_p, tuned)}.items():
                txt, audio = translate_and_speak(pp, mm, sysmsg, src)
                tr = chrf(txt, ref)
                agg[name]["tr"].append(tr)
                rt = None
                if audio is not None and len(audio) > SR // 4:
                    back = asr(judge_p, judge_m, judge_sys, audio)
                    rt = chrf(back, txt) if txt else 0.0
                    agg[name]["rt"].append(rt)
                    row[f"{name}_asrback"] = back
                else:
                    agg[name]["rt"].append(0.0)
                row[f"{name}_txt"] = txt
                row[f"{name}_tr_chrf"] = tr
                row[f"{name}_rt_chrf"] = rt
                row[f"{name}_dur"] = round(len(audio) / SR, 1) if audio is not None else 0.0
            examples.append(row)
        def avg(x): return round(sum(x) / max(len(x), 1), 1)
        report[d] = {
            "n": len(items),
            "base":  {"translate_chrF": avg(agg["base"]["tr"]),  "audio_roundtrip_chrF": avg(agg["base"]["rt"])},
            "tuned": {"translate_chrF": avg(agg["tuned"]["tr"]), "audio_roundtrip_chrF": avg(agg["tuned"]["rt"])},
            "examples": examples[:4],
        }
        b, t = report[d]["base"], report[d]["tuned"]
        print(f"\n=== {d} (n={len(items)}, target audio={tgt_audio_lang}) ===")
        print(f"  translate chrF :  base {b['translate_chrF']:5.1f}  ->  tuned {t['translate_chrF']:5.1f}")
        print(f"  audio RT  chrF :  base {b['audio_roundtrip_chrF']:5.1f}  ->  tuned {t['audio_roundtrip_chrF']:5.1f}")
        for ex in examples[:3]:
            print(f"   in : {ex['in'][:60]}")
            print(f"   ref: {ex['ref'][:60]}")
            print(f"   base : '{ex['base_txt'][:55]}'  tr={ex['base_tr_chrf']} rt={ex['base_rt_chrf']}")
            print(f"   tuned: '{ex['tuned_txt'][:55]}'  tr={ex['tuned_tr_chrf']} rt={ex['tuned_rt_chrf']}")

    with open(f"{OUTDIR}/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {OUTDIR}/report.json")
    print("\n=== SUMMARY ===")
    print(json.dumps({d: {"base": report[d]["base"], "tuned": report[d]["tuned"]} for d in report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
