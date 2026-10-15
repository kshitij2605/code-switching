from __future__ import annotations
import os, torch, soundfile as sf
from safetensors.torch import load_file

BASE_MODEL   = "LiquidAI/LFM2.5-Audio-1.5B-JP"
CKPT         = "/data/lfm2.5/run-a/checkpoints/lfm_asr/checkpoints/checkpoint_6/model.safetensors"
ROOT         = "/data/lfm2.5/run-a"
IM_END_TOKEN = 7
MAX_NEW_TOKENS = 150

PROMPTS = [
    ("Native",    "Transcribe the audio."),
    ("Force-JP",  "Transcribe in Japanese."),
    ("Force-EN",  "Transcribe in English."),
]

SAMPLES = [
    dict(type="CS", id="cs_001", audio="data/code_switching/audio/cs_001.wav",
         native="絶対 この shirt かなり cool と思わない 値段は reasonable だよ よろしく",
         jp="絶対 この シャツ かなり かっこいい と思わない 値段は 手頃 だよ よろしく",
         en="You absolutely have to agree that this shirt is pretty cool, and the price is reasonable too."),
    dict(type="CS", id="cs_002", audio="data/code_switching/audio/cs_002.wav",
         native="明日の schedule を確認して right? それと the review も終わらせないと だね",
         jp="明日の スケジュール を確認して そうでしょ? それと レビュー も終わらせないと だね",
         en="Let's confirm tomorrow's schedule, right? And we also need to finish the review."),
    dict(type="CS", id="cs_003", audio="data/code_switching/audio/cs_003.wav",
         native="その 明日の presentation の準備できた for sure それと the budget も確認したほうがいい だよね",
         jp="その 明日の プレゼン の準備できた 確かに それと 予算 も確認したほうがいい だよね",
         en="You've definitely prepared for tomorrow's presentation, and you should also check the budget, right?"),
    dict(type="EN", id="en_001", audio="data/english_only/audio/en_001.wav",
         native="I need to schedule a checkup with my doctor.",
         jp="医者と検診の予約をしなければなりません。",
         en="I need to schedule a checkup with my doctor."),
    dict(type="EN", id="en_002", audio="data/english_only/audio/en_002.wav",
         native="Let's review the agenda before the meeting starts.",
         jp="会議が始まる前に議題を確認しましょう。",
         en="Let's review the agenda before the meeting starts."),
    dict(type="JA", id="ja_001", audio="data/japanese_only/audio/ja_001.wav",
         native="ピーク時のためにサーバーの容量を増やす必要があります。",
         jp="ピーク時のためにサーバーの容量を増やす必要があります。",
         en="We need to increase the server capacity for peak hours."),
    dict(type="JA", id="ja_002", audio="data/japanese_only/audio/ja_002.wav",
         native="このプロジェクトの締め切りは来週の金曜日です。",
         jp="このプロジェクトの締め切りは来週の金曜日です。",
         en="The deadline for this project is next Friday."),
]

def load_audio(path):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return torch.tensor(arr, dtype=torch.float).unsqueeze(0), sr

def infer(model, proc, wave, sr, prompt):
    from liquid_audio import ChatState
    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text(prompt); chat.end_turn()
    chat.new_turn("user");   chat.add_audio(wave, sr); chat.end_turn()
    chat.new_turn("assistant")
    toks = []
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=MAX_NEW_TOKENS):
            if t.numel() == 1:
                tid = int(t.reshape(-1)[0].item())
                if tid == IM_END_TOKEN:
                    break
                toks.append(tid)
    return proc.text.decode(toks, skip_special_tokens=True).strip() if toks else "(empty)"

def main():
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    print("Loading processor + checkpoint_6 (step 1400)...")
    proc  = LFM2AudioProcessor.from_pretrained(BASE_MODEL)
    model = LFM2AudioModel.from_pretrained(BASE_MODEL, device="cuda", dtype=torch.bfloat16)
    sd = load_file(CKPT, device="cuda")
    miss, unex = model.load_state_dict(sd, strict=False)
    print("  missing=%d  unexpected=%d" % (len(miss), len(unex)))
    model.eval()

    SEP = "=" * 108
    for s in SAMPLES:
        apath = os.path.join(ROOT, s["audio"])
        if not os.path.exists(apath):
            print("SKIP %s — not found: %s" % (s["id"], apath))
            continue
        wave, sr = load_audio(apath)
        print("\n" + SEP)
        print("[%s] %s" % (s["type"], s["id"]))
        for label, prompt in PROMPTS:
            ref = s["jp"] if label == "Force-JP" else (s["en"] if label == "Force-EN" else s["native"])
            hyp = infer(model, proc, wave, sr, prompt)
            print("\n  [%s]  prompt: \"%s\"" % (label, prompt))
            print("  REF: %s" % ref)
            print("  HYP: %s" % hyp)
    print("\n" + SEP + "\nDONE")

if __name__ == "__main__":
    main()
