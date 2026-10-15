"""Generate bilingual meeting minutes from a transcript with a small LFM2.5 text model.

Default = base `LFM2.5-1.2B-JP` (validated). Use JP, not `-Thinking` (it over-reasons and doesn't converge).
An optional LoRA adapter slots in if a future in-domain eval warrants the minutes SFT.
"""

from __future__ import annotations

from csmeeting.minutes.prompt import build_messages


class MinutesGenerator:
    def __init__(self, model_id: str = "LiquidAI/LFM2.5-1.2B-JP", device: str = "cuda",
                 dtype: str = "bfloat16", adapter: str | None = None, max_new_tokens: int = 768):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        d = getattr(torch, dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device if device == "cuda" else None, dtype=d
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

    def generate(self, transcript: str, max_new_tokens: int | None = None) -> str:
        torch = self._torch
        inputs = self.tokenizer.apply_chat_template(
            build_messages(transcript), add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.05,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
            )
        txt = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        if "</think>" in txt:  # if someone points this at a Thinking model, drop the reasoning
            txt = txt.split("</think>", 1)[1].strip()
        return txt
