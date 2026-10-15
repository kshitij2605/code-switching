"""Option B v4: inject REAL human speech (FLEURS validation) into the translating-TTS set.

Synthetic voices are plateauing; real audio adds genuine prosody/voice diversity. FLEURS validation is
disjoint from the FLEURS test split we evaluate on, so eval stays clean.

  en_us validation: real EN audio + EN text -> input JA = translate(EN->JA);  target=(EN text, real EN audio)  [ja2en]
  ja_jp validation: real JA audio + JA text -> input EN = translate(JA->EN);  target=(JA text, real JA audio)  [en2ja]

Back-translation uses the dedicated LFM2.5-1.2B-JP. Output merged with v3 -> optionB/data/training_v4/.
"""
import os, io, json, random
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
from pathlib import Path
import numpy as np, soundfile as sf, torch
from datasets import load_dataset, Audio
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(77)
ROOT = Path("/data/lfm2.5/run-c/optionB/data")
RDIR = ROOT / "real"; (RDIR / "en").mkdir(parents=True, exist_ok=True); (RDIR / "ja").mkdir(parents=True, exist_ok=True)
V3 = ROOT / "training_v3"
OUT = ROOT / "training_v4"; OUT.mkdir(parents=True, exist_ok=True)
SR = 24000
SYS_JA2EN = "Translate to English and speak."
SYS_EN2JA = "Translate to Japanese and speak."
TXT_ID = "LiquidAI/LFM2.5-1.2B-JP"
MT_J2E = ("You are a professional Japanese-to-English interpreter for business meetings. Translate the user's "
          "Japanese into natural, fluent English. Output ONLY the English translation, no preamble or quotes.")
MT_E2J = ("You are a professional English-to-Japanese interpreter for business meetings. Translate the user's "
          "English into natural, fluent Japanese. Output ONLY the Japanese translation, no preamble or quotes.")


def main():
    tok = AutoTokenizer.from_pretrained(TXT_ID)
    lm = AutoModelForCausalLM.from_pretrained(TXT_ID, dtype=torch.bfloat16, device_map="cuda").eval()

    def translate(text, sysmsg):
        ids = tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True).to(lm.device)
        pl = ids["input_ids"].shape[1]
        with torch.no_grad():
            o = lm.generate(**ids, do_sample=False, repetition_penalty=1.05, max_new_tokens=128)
        return tok.decode(o[0][pl:], skip_special_tokens=True).strip()

    def decode_audio(cell):
        if cell.get("bytes"):
            wav, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
        else:
            wav, sr = sf.read(cell["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), sr

    rows = []
    plan = [("en_us", "en", "ja2en", SYS_JA2EN, MT_E2J),   # EN audio; input=translate(EN->JA)
            ("ja_jp", "ja", "en2ja", SYS_EN2JA, MT_J2E)]   # JA audio; input=translate(JA->EN)
    for cfg, alang, direction, sysprompt, mt_sys in plan:
        ds = load_dataset("google/fleurs", cfg, split="validation").cast_column("audio", Audio(decode=False))
        print(f"{cfg} validation: {len(ds)} rows", flush=True)
        n = 0
        for i, r in enumerate(ds):
            tgt_text = r["transcription"].strip()
            if not tgt_text:
                continue
            try:
                wav, sr = decode_audio(r["audio"])
            except Exception as e:
                continue
            if len(wav) < sr // 2:   # skip <0.5s
                continue
            inp = translate(tgt_text, mt_sys)
            if not inp:
                continue
            p = RDIR / alang / f"real_{cfg}_{i:05d}.wav"
            sf.write(p, wav, sr)   # native SR; mapper resamples to 24k for Mimi
            rows.append({"audio_file": str(p), "system": sysprompt, "input_text": inp,
                         "target_text": tgt_text, "direction": direction, "src": "real_fleurs"})
            n += 1
            if n % 100 == 0:
                print(f"  {cfg}: {n} pairs (e.g. in='{inp[:40]}' tgt='{tgt_text[:40]}')", flush=True)
        print(f"{cfg}: kept {n}", flush=True)

    # merge with v3
    v3rows = []
    for f in ("train.jsonl", "eval.jsonl"):
        for line in open(V3 / f, encoding="utf-8"):
            if line.strip():
                v3rows.append(json.loads(line))
    allrows = v3rows + rows
    random.shuffle(allrows)
    n_eval = max(1, int(len(allrows) * 0.06))
    (OUT / "eval.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in allrows[:n_eval]), encoding="utf-8")
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in allrows[n_eval:]), encoding="utf-8")
    from collections import Counter
    tr = allrows[n_eval:]
    print(f"\nreal pairs added: {len(rows)}; v3 rows: {len(v3rows)}; total {len(allrows)}")
    print("train dir:", dict(Counter(r['direction'] for r in tr)))
    print("train src:", dict(Counter(r['src'].split(':')[0] for r in tr)))
    print(f"Train {len(tr)} | Eval {n_eval} -> {OUT}")


if __name__ == "__main__":
    main()
