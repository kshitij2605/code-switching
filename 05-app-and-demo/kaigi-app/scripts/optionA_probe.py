"""Option A feasibility probe: can the LFM2.5-Audio backbone replace the separate LFM2.5-1.2B-JP translator?

Two questions, both measured (not assumed):
  (1) MT quality — translate text->text through each audio model's own LM backbone (via generate_sequential,
      which starts in TEXT modality) and compare chrF/BLEU to the dedicated translator on FLEURS, both directions.
  (2) Cross-lingual TTS — does JP-Audio actually speak English (and EN-Audio Japanese)? Measured by round-trip:
      TTS the text, then ASR the audio back with the matching-language audio model, and score chrF vs the input.

Run on the GPU host with the 3.12 audio-venv. GPU picked via CUDA_VISIBLE_DEVICES (default: a free one).
"""
import os, re, json, time, argparse
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
import torch, sacrebleu
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
OUT = "/data/lfm2.5/run-c/runs/optionA"
os.makedirs(OUT, exist_ok=True)
SR = 24000
SPECIAL = re.compile(r"<\|[^|]*\|>")

JP_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
EN_ID = "LiquidAI/LFM2.5-Audio-1.5B"
TXT_ID = "LiquidAI/LFM2.5-1.2B-JP"
FLEURS = "/data/lfm2.5/run-c/data/mt/fleurs_ja_en_test.jsonl"

SYS_JA2EN = ("You are a professional Japanese-to-English interpreter for business meetings. "
             "Translate the user's Japanese into natural, fluent English. The Japanese may contain English words "
             "(code-switching) and katakana loanwords - render the meaning naturally. Preserve names, numbers, and "
             "technical/product terms. Output ONLY the English translation, with no preamble, notes, or quotation marks.")
SYS_EN2JA = ("You are a professional English-to-Japanese interpreter for business meetings. "
             "Translate the user's English into natural, fluent Japanese (use katakana for established loanwords). "
             "Preserve names, numbers, and technical/product terms. Output ONLY the Japanese translation, with no "
             "preamble, notes, or quotation marks.")
FEW_JA2EN = [("来週のミーティングまでにKPIをfinalizeしておきます。", "I'll finalize the KPIs before next week's meeting."),
             ("そのissueはengineeringチームにassignしました。", "I've assigned that issue to the engineering team."),
             ("今日のアジェンダは、スケジュールの確認とリソースの調整です。", "Today's agenda is to review the schedule and adjust resources.")]
FEW_EN2JA = [("Let's sync on the deliverables before Friday's release.", "金曜日のリリース前に、成果物についてすり合わせましょう。"),
             ("I'll assign that issue to the engineering team.", "その課題はエンジニアリングチームにアサインします。")]


def load_audio(mid):
    p = LFM2AudioProcessor.from_pretrained(mid, device=DEV).eval()
    m = LFM2AudioModel.from_pretrained(mid).eval().to(DEV)
    return p, m


# ---------- translation through the AUDIO model's text backbone ----------
def audio_translate(proc, model, text, direction):
    sysmsg = SYS_JA2EN if direction == "ja2en" else SYS_EN2JA
    shots = FEW_JA2EN if direction == "ja2en" else FEW_EN2JA
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(sysmsg); chat.end_turn()
    for a, b in shots:
        chat.new_turn("user"); chat.add_text(a); chat.end_turn()
        chat.new_turn("assistant"); chat.add_text(b); chat.end_turn()
    chat.new_turn("user"); chat.add_text(text); chat.end_turn()
    chat.new_turn("assistant")
    pieces = []
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=160, text_temperature=0.2, text_top_k=50):
            if t.numel() == 1:
                if int(t.view(-1)[0]) == 128:  # would switch to audio; not expected for translation
                    break
                pieces.append(proc.text.decode(t))
            else:
                break  # audio frame leaked -> stop, this isn't a clean text translation
    return SPECIAL.sub("", "".join(pieces)).strip()


# ---------- dedicated text translator baseline ----------
def make_text_translator():
    tok = AutoTokenizer.from_pretrained(TXT_ID)
    lm = AutoModelForCausalLM.from_pretrained(TXT_ID, dtype=torch.bfloat16, device_map="cuda").eval()

    def tr(text, direction):
        sysmsg = SYS_JA2EN if direction == "ja2en" else SYS_EN2JA
        shots = FEW_JA2EN if direction == "ja2en" else FEW_EN2JA
        msgs = [{"role": "system", "content": sysmsg}]
        for a, b in shots:
            msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": text})
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(lm.device)
        plen = ids["input_ids"].shape[1]
        with torch.no_grad():
            out = lm.generate(**ids, do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.05, max_new_tokens=160)
        return tok.decode(out[0][plen:], skip_special_tokens=True).strip()
    return tr


# ---------- audio model ASR + TTS (for the round-trip TTS test) ----------
def asr(proc, model, sysmsg, wav):
    wt = torch.from_numpy(wav).float().unsqueeze(0) if wav.ndim == 1 else torch.from_numpy(wav).float()
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(sysmsg); chat.end_turn()
    chat.new_turn("user"); chat.add_audio(wt, SR); chat.end_turn()
    chat.new_turn("assistant")
    out = []
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=256):
            if t.numel() == 1:
                out.append(proc.text.decode(t))
    return SPECIAL.sub("", "".join(out)).strip()


def tts(proc, model, sysmsg, text):
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(sysmsg); chat.end_turn()
    chat.new_turn("user"); chat.add_text(text); chat.end_turn()
    chat.new_turn("assistant")
    frames = []
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=1024, audio_temperature=0.8, audio_top_k=64):
            if t.numel() > 1 and bool((t < 2048).all()):
                frames.append(t)
    if not frames:
        return None
    wav = proc.decode(torch.stack(frames, 1).unsqueeze(0))[0].detach().cpu().float().numpy()
    return wav[0] if wav.ndim > 1 else wav


def load_pairs(n):
    seen, pairs = set(), []
    for line in open(FLEURS, encoding="utf-8"):
        r = json.loads(line)
        if r["ja"] in seen:
            continue
        seen.add(r["ja"]); pairs.append((r["ja"], r["en"]))
        if len(pairs) >= n:
            break
    return pairs


def score(hyps, refs):
    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--skip-mt", action="store_true")
    ap.add_argument("--skip-tts", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    jp_p, jp_m = load_audio(JP_ID)
    en_p, en_m = load_audio(EN_ID)
    print(f"audio models loaded {time.time()-t0:.0f}s", flush=True)

    report = {}

    if not args.skip_mt:
        pairs = load_pairs(args.n)
        print(f"\n=== MT: {len(pairs)} FLEURS pairs, both directions ===", flush=True)
        baseline = make_text_translator()
        engines = {"text-LFM2.5-1.2B-JP(baseline)": baseline,
                   "JP-Audio-backbone": (lambda txt, d: audio_translate(jp_p, jp_m, txt, d)),
                   "EN-Audio-backbone": (lambda txt, d: audio_translate(en_p, en_m, txt, d))}
        report["mt"] = {}
        for name, fn in engines.items():
            for d in ("ja2en", "en2ja"):
                t = time.time()
                if d == "ja2en":
                    hyps = [fn(ja, d) for ja, _ in pairs]; refs = [en for _, en in pairs]
                else:
                    hyps = [fn(en, d) for _, en in pairs]; refs = [ja for ja, _ in pairs]
                s = score(hyps, refs); s["sec"] = round(time.time() - t, 1)
                report["mt"][f"{name}|{d}"] = s
                print(f"  {name:34s} {d}: chrF={s['chrf']:5.1f} BLEU={s['bleu']:5.1f} ({s['sec']}s)", flush=True)
        # a couple of qualitative samples from each audio backbone
        report["mt_samples"] = []
        for ja, en in pairs[:3]:
            report["mt_samples"].append({"ja": ja, "ref_en": en,
                "jpA_ja2en": audio_translate(jp_p, jp_m, ja, "ja2en"),
                "enA_ja2en": audio_translate(en_p, en_m, ja, "ja2en"),
                "jpA_en2ja": audio_translate(jp_p, jp_m, en, "en2ja"),
                "enA_en2ja": audio_translate(en_p, en_m, en, "en2ja")})

    if not args.skip_tts:
        print("\n=== Cross-lingual TTS: does each model speak the OTHER language? (round-trip ASR) ===", flush=True)
        en_text = "Let's sync on the deliverables before Friday's product release and confirm the budget."
        ja_text = "金曜日のリリース前に、成果物と予算について確認させてください。"
        # try several system prompts to find what (if anything) makes JP-Audio speak English
        trials = [
            ("JP-Audio", jp_p, jp_m, "Perform TTS.", en_text, "en", en_p, en_m, "Perform ASR."),
            ("JP-Audio", jp_p, jp_m, "Perform TTS in english.", en_text, "en", en_p, en_m, "Perform ASR."),
            ("EN-Audio", en_p, en_m, "Perform TTS in japanese.", ja_text, "ja", jp_p, jp_m, "Perform ASR in japanese."),
            ("EN-Audio", en_p, en_m, "Perform TTS.", ja_text, "ja", jp_p, jp_m, "Perform ASR in japanese."),
        ]
        report["tts"] = []
        for i, (who, tp, tm, tsys, text, lang, ap_, am, asys) in enumerate(trials):
            wav = tts(tp, tm, tsys, text)
            rec = {"speaker": who, "tts_prompt": tsys, "target_lang": lang, "in": text}
            if wav is None or len(wav) < SR // 4:
                rec["result"] = "NO/EMPTY AUDIO"
            else:
                path = f"{OUT}/tts_{i}_{who}_{lang}.wav"
                import soundfile as sf
                sf.write(path, wav, SR)
                back = asr(ap_, am, asys, wav)
                rec["dur_s"] = round(len(wav) / SR, 1)
                rec["asr_back"] = back
                rec["chrf_roundtrip"] = round(sacrebleu.corpus_chrf([back], [[text]]).score, 1)
                rec["wav"] = path
            report["tts"].append(rec)
            print(f"  [{who}] '{tsys}' -> {lang}: {rec.get('chrf_roundtrip','-')} | back='{rec.get('asr_back', rec.get('result'))[:70]}'", flush=True)

    print("\n=== REPORT JSON ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    with open(f"{OUT}/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {OUT}/report.json  | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
