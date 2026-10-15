"""Option B: build the translating-TTS dataset for fine-tuning LFM2.5-Audio-1.5B-JP.

Task = text in language A -> (translated text in B) + (speech in B), via sequential generation:
the assistant turn is [translated_text, target_audio], which the mapper encodes as
`{text} <|audio_start|> {audio codes}`. So one fine-tune teaches translate + cross-lingual TTS.

Sources (translations are pre-computed in the metadata):
  english_only/  : EN audio + japanese_translation  -> input=JA text, target=(EN text, EN audio)   [ja2en]
  japanese_only/ : JA audio + english_translation    -> input=EN text, target=(JA text, JA audio)   [en2ja]

Each row: {audio_file (abs), system, input_text, target_text, direction}
"""
import json, random
from pathlib import Path

random.seed(77)
BASE = Path("/data/lfm2.5/run-a/data")
OUT = Path("/data/lfm2.5/run-c/optionB/data/training")
OUT.mkdir(parents=True, exist_ok=True)
EVAL_RATIO = 0.1

SYS_JA2EN = "Translate to English and speak."
SYS_EN2JA = "Translate to Japanese and speak."


def load(meta_rel, audio_in_field, text_out_field, audio_out_is, system, direction):
    """audio_in_field: text we feed as input; text_out_field: translated text target; the wav is the target audio."""
    path = BASE / meta_rel
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for s in rows:
        af = BASE.parent / s["audio_file"]  # audio_file already includes the "data/" prefix
        if not af.exists():
            continue
        inp = (s.get(audio_in_field) or "").strip()
        tgt = (s.get(text_out_field) or "").strip()
        if not inp or not tgt:
            continue
        out.append({"audio_file": str(af), "system": system, "input_text": inp,
                    "target_text": tgt, "direction": direction})
    print(f"  {meta_rel}: {len(out)} usable rows")
    return out


def main():
    # english_only: audio=EN. input = japanese_translation (JA text). target text = english_transcription. dir ja2en
    ja2en = load("english_only/metadata/transcriptions.json",
                 audio_in_field="japanese_translation", text_out_field="english_transcription",
                 audio_out_is="en", system=SYS_JA2EN, direction="ja2en")
    # japanese_only: audio=JA. input = english_translation (EN text). target text = japanese_transcription. dir en2ja
    en2ja = load("japanese_only/metadata/transcriptions.json",
                 audio_in_field="english_translation", text_out_field="japanese_transcription",
                 audio_out_is="ja", system=SYS_EN2JA, direction="en2ja")

    allrows = ja2en + en2ja
    random.shuffle(allrows)
    n_eval = max(1, int(len(allrows) * EVAL_RATIO))
    train, eval_ = allrows[n_eval:], allrows[:n_eval]

    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    (OUT / "eval.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in eval_), encoding="utf-8")

    from collections import Counter
    print(f"\nTrain {len(train)} | Eval {len(eval_)}")
    print("dir breakdown (train):", dict(Counter(r["direction"] for r in train)))
    print("\nsamples:")
    for r in random.sample(train, min(3, len(train))):
        print(f"  [{r['direction']}] sys='{r['system']}'")
        print(f"    in : {r['input_text'][:70]}")
        print(f"    out: {r['target_text'][:70]}  + audio {Path(r['audio_file']).name}")
    print(f"\nwrote {OUT}/train.jsonl, eval.jsonl")


if __name__ == "__main__":
    main()
