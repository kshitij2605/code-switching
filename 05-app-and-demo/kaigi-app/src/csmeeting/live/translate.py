"""MT stage translator (used by the CLI cascade in pipeline.py). Bidirectional via build_messages(text, src, tgt).

Default = base `LFM2.5-1.2B-JP` + a translation prompt, no fine-tune (test-first; see docs/translation.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from csmeeting.mt.prompt import build_messages

if TYPE_CHECKING:
    from csmeeting.live.config import LiveConfig


class Translator(Protocol):
    def translate(self, text: str, src: str = "ja", tgt: str = "en") -> str: ...


class IdentityTranslator:
    """Debug translator (no model). Used by --mock."""

    def translate(self, text: str, src: str = "ja", tgt: str = "en") -> str:
        return f"[mock-{tgt}] {text}"


class LFMTranslator:
    """Base (or LoRA-adapted) LFM2.5 text model as a JA<->EN translator via transformers."""

    def __init__(self, cfg: "LiveConfig"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.translator_model_id)
        dtype = getattr(torch, cfg.dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.translator_model_id, device_map=cfg.device if cfg.device == "cuda" else None, dtype=dtype
        )
        if cfg.translator_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, cfg.translator_adapter)
        if cfg.device != "cuda":
            self.model = self.model.to(cfg.device)
        self.model.eval()

    def translate(self, text: str, src: str = "ja", tgt: str = "en") -> str:
        torch = self._torch
        text = (text or "").strip()
        if not text:
            return ""
        inputs = self.tokenizer.apply_chat_template(
            build_messages(text, src, tgt), add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.05,
                max_new_tokens=self.cfg.max_new_tokens_mt,
            )
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
