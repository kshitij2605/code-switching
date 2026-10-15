"""Option B v3: push English audio further — wider Kokoro voice pool, 2 EN voices per JA->EN sentence.

Validates a broad voice pool first (drops any that fail to load), then synthesizes:
  - ja2en (EN audio): each JA->EN text pair spoken by 2 DISTINCT English voices  (~2x English coverage)
  - en2ja (JA audio): CS pairs, 1 rotating Japanese voice
Combined with the original 600 edge-tts pairs. Output: optionB/data/training_v3/{train,eval}.jsonl
"""
import os, json, random
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
from pathlib import Path
import numpy as np, soundfile as sf
from kokoro import KPipeline

random.seed(77)
BASE = Path("/data/lfm2.5/run-a/data")
ROOT = Path("/data/lfm2.5/run-c/optionB/data")
KDIR = ROOT / "kokoro_v3"; (KDIR / "en").mkdir(parents=True, exist_ok=True); (KDIR / "ja").mkdir(parents=True, exist_ok=True)
OUT = ROOT / "training_v3"; OUT.mkdir(parents=True, exist_ok=True)
SR = 24000
SYS_JA2EN = "Translate to English and speak."
SYS_EN2JA = "Translate to Japanese and speak."

EN_POOL = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_aoede", "af_kore",
           "am_michael", "am_adam", "am_eric", "am_liam", "am_fenrir",
           "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
JA_POOL = ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jm_kumo"]


def meta(rel): return json.loads((BASE / rel).read_text(encoding="utf-8"))


def original_rows():
    rows = []
    for s in meta("english_only/metadata/transcriptions.json"):
        af = BASE.parent / s["audio_file"]
        if af.exists() and s.get("japanese_translation") and s.get("english_transcription"):
            rows.append({"audio_file": str(af), "system": SYS_JA2EN, "input_text": s["japanese_translation"].strip(),
                         "target_text": s["english_transcription"].strip(), "direction": "ja2en", "src": "edge"})
    for s in meta("japanese_only/metadata/transcriptions.json"):
        af = BASE.parent / s["audio_file"]
        if af.exists() and s.get("english_translation") and s.get("japanese_transcription"):
            rows.append({"audio_file": str(af), "system": SYS_EN2JA, "input_text": s["english_translation"].strip(),
                         "target_text": s["japanese_transcription"].strip(), "direction": "en2ja", "src": "edge"})
    return rows


def main():
    pipes = {}
    def pipe(lc):
        if lc not in pipes: pipes[lc] = KPipeline(lang_code=lc)
        return pipes[lc]
    def synth(text, voice):
        gen = pipe(voice[0])(text, voice=voice)
        return np.concatenate([g.audio.numpy() for g in gen]).astype(np.float32)

    # validate voice pools
    def validate(pool, probe):
        ok = []
        for v in pool:
            try:
                a = synth(probe, v)
                if len(a) > SR // 5: ok.append(v)
            except Exception as e:
                print(f"  drop voice {v}: {type(e).__name__}", flush=True)
        return ok
    en_ok = validate(EN_POOL, "This is a short test sentence for the meeting.")
    ja_ok = validate(JA_POOL, "これはテスト用の短い文です。")
    print(f"EN voices OK ({len(en_ok)}): {en_ok}", flush=True)
    print(f"JA voices OK ({len(ja_ok)}): {ja_ok}", flush=True)

    # gather text specs
    ja2en_txt = []
    for s in meta("english_only/metadata/transcriptions.json"):
        if s.get("japanese_translation") and s.get("english_transcription"):
            ja2en_txt.append((s["japanese_translation"].strip(), s["english_transcription"].strip()))
    for s in meta("japanese_only/metadata/transcriptions.json"):
        if s.get("japanese_transcription") and s.get("english_translation"):
            ja2en_txt.append((s["japanese_transcription"].strip(), s["english_translation"].strip()))
    for s in meta("code_switching/metadata/transcriptions.json"):
        if s.get("all_japanese") and s.get("all_english"):
            ja2en_txt.append((s["all_japanese"].strip(), s["all_english"].strip()))
    en2ja_txt = [(s["all_english"].strip(), s["all_japanese"].strip())
                 for s in meta("code_switching/metadata/transcriptions.json")
                 if s.get("all_english") and s.get("all_japanese")]

    rows, ei, ji, fail = [], 0, 0, 0
    # ja2en: 2 distinct EN voices per sentence
    for k, (inp, tgt) in enumerate(ja2en_txt):
        v1 = en_ok[k % len(en_ok)]
        v2 = en_ok[(k + 1 + k // len(en_ok)) % len(en_ok)]
        if v2 == v1: v2 = en_ok[(k + 2) % len(en_ok)]
        for voice in (v1, v2):
            try:
                a = synth(tgt, voice)
                if len(a) < SR // 4: fail += 1; continue
            except Exception:
                fail += 1; continue
            p = KDIR / "en" / f"ken_{ei:05d}_{voice}.wav"; ei += 1; sf.write(p, a, SR)
            rows.append({"audio_file": str(p), "system": SYS_JA2EN, "input_text": inp,
                         "target_text": tgt, "direction": "ja2en", "src": f"kokoro:{voice}"})
        if (k + 1) % 200 == 0: print(f"  ja2en {k+1}/{len(ja2en_txt)} (en clips={ei} fail={fail})", flush=True)
    # en2ja: 1 JA voice per sentence
    for k, (inp, tgt) in enumerate(en2ja_txt):
        voice = ja_ok[k % len(ja_ok)]
        try:
            a = synth(tgt, voice)
            if len(a) < SR // 4: fail += 1; continue
        except Exception:
            fail += 1; continue
        p = KDIR / "ja" / f"kja_{ji:05d}_{voice}.wav"; ji += 1; sf.write(p, a, SR)
        rows.append({"audio_file": str(p), "system": SYS_EN2JA, "input_text": inp,
                     "target_text": tgt, "direction": "en2ja", "src": f"kokoro:{voice}"})

    allrows = original_rows() + rows
    random.shuffle(allrows)
    n_eval = max(1, int(len(allrows) * 0.07))
    (OUT / "eval.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in allrows[:n_eval]), encoding="utf-8")
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in allrows[n_eval:]), encoding="utf-8")
    from collections import Counter
    tr = allrows[n_eval:]
    print(f"\nKokoro v3: en={ei} ja={ji} fail={fail}; total {len(allrows)}")
    print("train dir:", dict(Counter(r["direction"] for r in tr)))
    print("train src:", dict(Counter(r["src"].split(':')[0] for r in tr)))
    print(f"Train {len(tr)} | Eval {n_eval} -> {OUT}")


if __name__ == "__main__":
    main()
