"""Option C eval: minutes via dedicated text model vs clean audio backbone vs backbone+minutes-LoRA.

Runs on held-out eval transcripts + one out-of-distribution hand-written transcript. Prints all three outputs
and a quick format check (4 required sections present, English summary actually in English, no 「私」 owners).
"""
import os, re, sys, json
os.environ.setdefault("HF_HOME", "/data/lfm2.5/run-c/hf_cache")
sys.path.insert(0, "/data/lfm2.5/run-c/appsrc")
import torch
from csmeeting.minutes.prompt import build_messages, SYSTEM_PROMPT
SP = re.compile(r"<\|[^|]*\|>")
LORA = "/data/lfm2.5/run-c/optionC/checkpoints/minutes_lora/final"

OOD = "\n".join([  # different style than training (free, code-switched)
 "今日のmeetingのagendaをshareします。",
 "Let's confirm the budget before Friday's release.",
 "私が議事録をまとめます。来週月曜までに。",
 "田中さん、データの確認をお願いできますか。",
 "はい、明日までに確認します。",
 "あと、APIのレスポンス形式にバグがあるので、鈴木さんに修正をお願いします。",
])
held = [json.loads(l) for l in open("/data/lfm2.5/run-c/optionC/data/training/eval.jsonl", encoding="utf-8")]
TRANSCRIPTS = [("OOD-handwritten", OOD)] + [(f"held-{i}", held[i]["transcript"]) for i in range(2)]


def fmt_check(m):
    secs = all(s in m for s in ["## 要約", "## 決定事項", "## アクションアイテム", "## English Summary"])
    eng = m.split("English Summary",1)[-1]
    eng_ok = bool(re.search(r"[A-Za-z]{4,}", eng)) and not re.search(r"[ぁ-んァ-ン一-龯]", eng[:200])
    no_watashi = "担当: 私" not in m
    return f"sections={secs} eng_in_english={eng_ok} no_私_owner={no_watashi}"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-JP")
    lm = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-1.2B-JP", dtype=torch.bfloat16, device_map="cuda").eval()
    def m_text(tr):
        ids = tok.apply_chat_template(build_messages(tr), add_generation_prompt=True, return_tensors="pt", return_dict=True).to(lm.device)
        pl = ids["input_ids"].shape[1]
        with torch.no_grad():
            o = lm.generate(**ids, do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.05, max_new_tokens=768)
        t = tok.decode(o[0][pl:], skip_special_tokens=True).strip()
        return t.split("</think>",1)[1].strip() if "</think>" in t else t

    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
    from peft import PeftModel
    proc = LFM2AudioProcessor.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B-JP", device="cuda").eval()
    base = LFM2AudioModel.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B-JP").eval().to("cuda")
    tuned = PeftModel.from_pretrained(base, LORA).eval()

    def m_backbone(tr, use_lora, fewshot):
        msgs = build_messages(tr) if fewshot else None
        c = ChatState(proc)
        c.new_turn("system"); c.add_text(SYSTEM_PROMPT); c.end_turn()
        c.new_turn("user"); c.add_text(tr); c.end_turn(); c.new_turn("assistant")
        out = []
        ctx = (lambda: __import__("contextlib").nullcontext()) if use_lora else tuned.disable_adapter
        with torch.no_grad(), ctx():
            for t in tuned.generate_sequential(**c, max_new_tokens=768, text_temperature=0.2, text_top_k=50):
                if t.numel() == 1:
                    if int(t.view(-1)[0]) == 128: break
                    out.append(proc.text.decode(t))
        return SP.sub("", "".join(out)).strip()

    for name, tr in TRANSCRIPTS:
        print("\n" + "#"*70 + f"\n# {name}\n" + "#"*70)
        A = m_text(tr); B = m_backbone(tr, use_lora=False, fewshot=False); C = m_backbone(tr, use_lora=True, fewshot=False)
        print("\n--- (A) dedicated LFM2.5-1.2B-JP ---  ", fmt_check(A)); print(A)
        print("\n--- (B) clean backbone (baseline) ---  ", fmt_check(B)); print(B)
        print("\n--- (C) backbone + minutes-LoRA ---  ", fmt_check(C)); print(C)


if __name__ == "__main__":
    main()
