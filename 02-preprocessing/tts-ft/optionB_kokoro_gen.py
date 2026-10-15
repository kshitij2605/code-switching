"""Option B augmentation: grow the translating-TTS set with fresh Kokoro-82M target audio.

Focus = English audio (the weak side, round-trip 61). Pull JA<->EN parallel text from ALL THREE metadata
sources and synthesize target-language audio with ROTATING Kokoro voices (voice diversity per the research),
then combine with the original edge-tts pairs (two synthetic sources = less overfit to one synthesizer).

Output: optionB/data/training_v2/{train,eval}.jsonl  + wavs under optionB/data/kokoro/{en,ja}/
"""
import os, json, random
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
from pathlib import Path
import numpy as np, soundfile as sf
from kokoro import KPipeline

random.seed(77)
BASE = Path("/data/lfm2.5/run-a/data")
ROOT = Path("/data/lfm2.5/run-c/optionB/data")
KDIR = ROOT / "kokoro"
(KDIR / "en").mkdir(parents=True, exist_ok=True)
(KDIR / "ja").mkdir(parents=True, exist_ok=True)
OUT = ROOT / "training_v2"
OUT.mkdir(parents=True, exist_ok=True)
SR = 24000
SYS_JA2EN = "Translate to English and speak."
SYS_EN2JA = "Translate to Japanese and speak."

EN_VOICES = ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bm_george"]
JA_VOICES = ["jf_alpha", "jf_gongitsune", "jm_kumo"]


def meta(rel):
    return json.loads((BASE / rel).read_text(encoding="utf-8"))


def original_rows():
    """The first-round edge-tts pairs (kept for source diversity)."""
    rows = []
    for s in meta("english_only/metadata/transcriptions.json"):
        af = BASE.parent / s["audio_file"]
        if af.exists() and s.get("japanese_translation") and s.get("english_transcription"):
            rows.append({"audio_file": str(af), "system": SYS_JA2EN,
                         "input_text": s["japanese_translation"].strip(),
                         "target_text": s["english_transcription"].strip(), "direction": "ja2en", "src": "edge"})
    for s in meta("japanese_only/metadata/transcriptions.json"):
        af = BASE.parent / s["audio_file"]
        if af.exists() and s.get("english_translation") and s.get("japanese_transcription"):
            rows.append({"audio_file": str(af), "system": SYS_EN2JA,
                         "input_text": s["english_translation"].strip(),
                         "target_text": s["japanese_transcription"].strip(), "direction": "en2ja", "src": "edge"})
    return rows


def kokoro_specs():
    """List of (input_text, target_text, target_lang, direction) to synthesize with Kokoro."""
    specs = []
    # ja2en (EN audio target): input JA, speak EN
    for s in meta("english_only/metadata/transcriptions.json"):
        if s.get("japanese_translation") and s.get("english_transcription"):
            specs.append((s["japanese_translation"].strip(), s["english_transcription"].strip(), "en", "ja2en"))
    for s in meta("japanese_only/metadata/transcriptions.json"):
        if s.get("japanese_transcription") and s.get("english_translation"):
            specs.append((s["japanese_transcription"].strip(), s["english_translation"].strip(), "en", "ja2en"))
    for s in meta("code_switching/metadata/transcriptions.json"):
        if s.get("all_japanese") and s.get("all_english"):
            specs.append((s["all_japanese"].strip(), s["all_english"].strip(), "en", "ja2en"))
    # en2ja (JA audio target): input EN, speak JA — CS only, to keep JA fresh + balanced
    for s in meta("code_switching/metadata/transcriptions.json"):
        if s.get("all_english") and s.get("all_japanese"):
            specs.append((s["all_english"].strip(), s["all_japanese"].strip(), "ja", "en2ja"))
    return specs


def main():
    pipes = {}
    def pipe(lang_code):
        if lang_code not in pipes:
            pipes[lang_code] = KPipeline(lang_code=lang_code)
        return pipes[lang_code]

    def synth(text, voice):
        lc = voice[0]  # a=US, b=GB, j=JA
        gen = pipe(lc)(text, voice=voice)
        return np.concatenate([g.audio.numpy() for g in gen]).astype(np.float32)

    specs = kokoro_specs()
    print(f"Kokoro specs: {len(specs)}  (ja2en + en2ja CS)", flush=True)
    rows, ei, ji, fail = [], 0, 0, 0
    for k, (inp, tgt, tlang, direction) in enumerate(specs):
        voices = EN_VOICES if tlang == "en" else JA_VOICES
        voice = voices[k % len(voices)]
        try:
            audio = synth(tgt, voice)
            if len(audio) < SR // 4:
                fail += 1; continue
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  synth fail [{voice}] {type(e).__name__}: {str(e)[:70]}", flush=True)
            continue
        if tlang == "en":
            path = KDIR / "en" / f"ken_{ei:05d}_{voice}.wav"; ei += 1
        else:
            path = KDIR / "ja" / f"kja_{ji:05d}_{voice}.wav"; ji += 1
        sf.write(path, audio, SR)
        rows.append({"audio_file": str(path), "system": SYS_JA2EN if direction == "ja2en" else SYS_EN2JA,
                     "input_text": inp, "target_text": tgt, "direction": direction, "src": f"kokoro:{voice}"})
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(specs)} synthesized (en={ei} ja={ji} fail={fail})", flush=True)

    orig = original_rows()
    allrows = orig + rows
    random.shuffle(allrows)
    n_eval = max(1, int(len(allrows) * 0.08))
    eval_, train = allrows[:n_eval], allrows[n_eval:]
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    (OUT / "eval.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in eval_), encoding="utf-8")

    from collections import Counter
    print(f"\nKokoro made {len(rows)} (en={ei} ja={ji} fail={fail}); + original {len(orig)} = {len(allrows)}")
    print("train dir:", dict(Counter(r["direction"] for r in train)))
    print("train src:", dict(Counter(r["src"].split(':')[0] for r in train)))
    print(f"Train {len(train)} | Eval {len(eval_)} -> {OUT}")


if __name__ == "__main__":
    main()
