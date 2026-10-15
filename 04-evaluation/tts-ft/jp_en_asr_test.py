"""Decisive test for the single-model (JP-Audio only) plan: can LFM2.5-Audio-1.5B-JP do ENGLISH ASR?

Synthesize a few English clips with the EN base model (clean reference text), then transcribe them with
JP-Audio under several ASR prompts. Score CER vs the reference. If JP-Audio retained English ASR from its
base, this single-model app is viable for the EN-speaker input step.
"""
import os, re, json
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
import torch, sacrebleu
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

DEV = "cuda"; SR = 24000
SPECIAL = re.compile(r"<\|[^|]*\|>")
JP_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
EN_ID = "LiquidAI/LFM2.5-Audio-1.5B"

REFS = [
    "Let's confirm the budget before Friday's release.",
    "Can you share the agenda for tomorrow's meeting?",
    "We need to increase the server capacity for peak hours.",
]

def load(mid):
    p = LFM2AudioProcessor.from_pretrained(mid, device=DEV).eval()
    m = LFM2AudioModel.from_pretrained(mid).eval().to(DEV)
    return p, m

def tts(p, m, sysmsg, text):
    c = ChatState(p); c.new_turn("system"); c.add_text(sysmsg); c.end_turn()
    c.new_turn("user"); c.add_text(text); c.end_turn(); c.new_turn("assistant")
    fr = []
    with torch.no_grad():
        for t in m.generate_sequential(**c, max_new_tokens=1024, audio_temperature=0.8, audio_top_k=64):
            if t.numel() > 1 and bool((t < 2048).all()):
                fr.append(t)
    w = p.decode(torch.stack(fr, 1).unsqueeze(0))[0].cpu().float().numpy()
    return w[0] if w.ndim > 1 else w

def asr(p, m, sysmsg, wav):
    wt = torch.from_numpy(wav).float().unsqueeze(0)
    c = ChatState(p); c.new_turn("system"); c.add_text(sysmsg); c.end_turn()
    c.new_turn("user"); c.add_audio(wt, SR); c.end_turn(); c.new_turn("assistant")
    out = []
    with torch.no_grad():
        for t in m.generate_sequential(**c, max_new_tokens=256):
            if t.numel() == 1:
                out.append(p.text.decode(t))
    return SPECIAL.sub("", "".join(out)).strip()

def cer(hyp, ref):
    # char error rate via sacrebleu chrf is not CER; compute simple normalized edit distance
    import difflib
    h, r = hyp.lower().strip(), ref.lower().strip()
    sm = difflib.SequenceMatcher(None, r, h)
    return round(100 * (1 - sm.ratio()), 1)

jp_p, jp_m = load(JP_ID)
en_p, en_m = load(EN_ID)
print("loaded", flush=True)

clips = [tts(en_p, en_m, "Perform TTS. Use the US female voice.", r) for r in REFS]
print("synthesized EN clips", flush=True)

prompts = ["Perform ASR.", "Perform ASR in english.", "Transcribe in English.", "Transcribe the audio."]
report = {"en_asr_by_jp_audio": {}}
for sp in prompts:
    rows = []
    for ref, w in zip(REFS, clips):
        h = asr(jp_p, jp_m, sp, w)
        rows.append({"ref": ref, "hyp": h, "err%": cer(h, ref)})
    avg = round(sum(x["err%"] for x in rows) / len(rows), 1)
    report["en_asr_by_jp_audio"][sp] = {"avg_err_pct": avg, "rows": rows}
    print(f"\n[JP-Audio EN-ASR] prompt='{sp}'  avg_err={avg}%")
    for x in rows:
        print(f"   ref: {x['ref']}\n   hyp: {x['hyp']}   (err {x['err%']}%)")

# sanity: EN base transcribing same clips (ceiling)
base_rows = [{"ref": r, "hyp": asr(en_p, en_m, "Perform ASR.", w), } for r, w in zip(REFS, clips)]
for x in base_rows:
    x["err%"] = cer(x["hyp"], x["ref"])
report["en_asr_by_en_base_ceiling"] = {"avg_err_pct": round(sum(x["err%"] for x in base_rows)/len(base_rows),1), "rows": base_rows}
print(f"\n[EN-base ceiling] avg_err={report['en_asr_by_en_base_ceiling']['avg_err_pct']}%")

print("\n=== JSON ===")
print(json.dumps(report, ensure_ascii=False, indent=2))
