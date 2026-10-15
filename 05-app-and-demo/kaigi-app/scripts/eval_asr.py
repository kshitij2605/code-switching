#!/usr/bin/env python
"""Evaluate a code-switch ASR checkpoint on a held-out manifest: CER / WER / PIER, per bucket.

Metrics:
  - CER  : char error rate (whole utterance; the JA-appropriate metric).
  - WER  : word error rate over Latin/English tokens (English competence).
  - PIER : switch-point error rate — for code-switch rows, the fraction of *embedded-language* spans (e.g. the English
           switch inside JA) NOT correctly rendered in the hypothesis. This is the metric that proves we fixed the
           hard part. Proxy: a switch span counts correct if its normalized text is a substring of the normalized hyp.

Run it for the BASE model and the FINE-TUNED checkpoint; each writes a JSON. Compare to show before/after + that the
monolingual buckets didn't regress (catastrophic-forgetting check).

  CUDA_VISIBLE_DEVICES=2 python scripts/eval_asr.py --model LiquidAI/LFM2.5-Audio-1.5B-JP \
      --manifest data/cs_mixed/eval.jsonl --out runs/eval_base.json
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_asr.py --model runs/cs_asr_jp/final \
      --manifest data/cs_mixed/eval.jsonl --out runs/eval_ft.json
  python scripts/eval_asr.py --compare runs/eval_base.json runs/eval_ft.json   # no model load; just diff
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.manifest import read_jsonl  # noqa: E402


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip().lower()


def _latin_tokens(s: str) -> list[str]:
    out, cur = [], []
    for ch in _norm(s):
        if ch.isascii() and (ch.isalnum() or ch == "'"):
            cur.append(ch)
        elif cur:
            out.append("".join(cur)); cur = []
    if cur:
        out.append("".join(cur))
    return out


def _levenshtein(a: list | str, b: list | str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def _matrix_lang(spans: list[dict]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for sp in spans:
        counts[sp["lang"]] += len(sp.get("text", ""))
    return max(counts, key=counts.get) if counts else "ja"


def score_row(ref: str, hyp: str, spans: list[dict]) -> dict:
    ref_c = _norm(ref).replace(" ", "")
    hyp_c = _norm(hyp).replace(" ", "")
    cer_err = _levenshtein(ref_c, hyp_c)
    ref_w = _latin_tokens(ref)
    hyp_w = _latin_tokens(hyp)
    wer_err = _levenshtein(ref_w, hyp_w)
    # PIER: embedded-language spans (lang != matrix)
    matrix = _matrix_lang(spans) if spans else "ja"
    switch_spans = [sp for sp in (spans or []) if sp["lang"] != matrix and sp.get("text", "").strip()]
    hyp_n = _norm(hyp)
    pier_hit = sum(1 for sp in switch_spans if _norm(sp["text"]) in hyp_n)
    return {
        "cer_err": cer_err, "cer_n": len(ref_c),
        "wer_err": wer_err, "wer_n": len(ref_w),
        "pier_hit": pier_hit, "pier_n": len(switch_spans),
    }


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def aggregate(per_utt: list[dict]) -> dict:
    buckets: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in per_utt:
        for key in ("all", r["bucket"]):
            b = buckets[key]
            for k in ("cer_err", "cer_n", "wer_err", "wer_n", "pier_hit", "pier_n"):
                b[k] += r[k]
            b["rows"] += 1
    summary = {}
    for name, b in buckets.items():
        summary[name] = {
            "rows": b["rows"],
            "CER": _rate(b["cer_err"], b["cer_n"]),
            "WER": _rate(b["wer_err"], b["wer_n"]),
            "PIER": round(1 - _rate(b["pier_hit"], b["pier_n"]), 4) if b["pier_n"] else None,
        }
    return summary


def _bucket(row: dict) -> str:
    style = row.get("style", "")
    if style.startswith("mono_en"):
        return "mono_en"
    if style.startswith("mono_ja"):
        return "mono_ja"
    return "cs"


def _print_summary(title: str, summary: dict) -> None:
    print(f"\n== {title} ==")
    print(f"{'bucket':10} {'rows':>6} {'CER':>8} {'WER':>8} {'PIER':>8}")
    for name in ("all", "cs", "mono_ja", "mono_en"):
        if name in summary:
            s = summary[name]
            pier = "-" if s["PIER"] is None else f"{s['PIER']:.4f}"
            print(f"{name:10} {s['rows']:>6} {s['CER']:>8.4f} {s['WER']:>8.4f} {pier:>8}")


_SPECIAL = re.compile(r"<\|[^|]*\|>")   # strip <|im_end|> etc. leaked by the ASR decode


def _enable_local_model_dirs() -> None:
    """Patch liquid_audio's get_model_dir so from_pretrained accepts a local checkpoint dir (our FT output),
    not only an HF repo id. The local dir must contain config.json + processor files (we copy them in)."""
    import importlib
    import sys
    import liquid_audio.utils as u

    orig = u.get_model_dir
    if getattr(orig, "_local_patched", False):
        return

    def patched(repo_id, revision=None, **kw):
        p = Path(str(repo_id))
        if p.is_dir():
            return p
        return orig(repo_id, revision=revision, **kw)

    patched._local_patched = True
    u.get_model_dir = patched
    for modname in ("liquid_audio.model.lfm2_audio", "liquid_audio.processor", "liquid_audio.detokenizer"):
        m = sys.modules.get(modname) or importlib.import_module(modname)
        if hasattr(m, "get_model_dir"):
            m.get_model_dir = patched


class _Transcriber:
    """Self-contained ASR via liquid_audio (no csmeeting.live dependency — works from a local FT dir or an HF id)."""

    def __init__(self, model_id: str, device: str = "cuda", adapter: str | None = None):
        import torch
        _enable_local_model_dirs()   # let from_pretrained load a local FT dir, not just an HF repo id
        from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
        self._torch, self._ChatState = torch, ChatState
        self.proc = LFM2AudioProcessor.from_pretrained(model_id, device=device).eval()
        self.model = LFM2AudioModel.from_pretrained(model_id, device=device).eval()
        try:
            self.model = self.model.to(device)
        except Exception:
            pass
        if adapter:                  # LoRA: base model_id + adapter dir
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
            print(f"  + LoRA adapter: {adapter}")

    def transcribe(self, wav, sr: int, prompt: str) -> str:
        import numpy as np
        torch = self._torch
        wt = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        if wt.ndim == 1:
            wt = wt.unsqueeze(0)
        chat = self._ChatState(self.proc)
        chat.new_turn("system"); chat.add_text(prompt); chat.end_turn()
        chat.new_turn("user"); chat.add_audio(wt, sr); chat.end_turn()
        chat.new_turn("assistant")
        pieces = []
        with torch.no_grad():
            for t in self.model.generate_sequential(**chat, max_new_tokens=256):
                if t.numel() == 1:
                    pieces.append(self.proc.text.decode(t))
        return _SPECIAL.sub("", "".join(pieces)).strip()


def run_eval(args) -> None:
    import soundfile as sf

    ja_prompt = args.system_ja or "Perform ASR in japanese."
    en_prompt = args.system_en or "Perform ASR."
    print(f"loading model: {args.model} (device={args.device})")
    asr = _Transcriber(args.model, device=args.device, adapter=args.adapter)

    rows = list(read_jsonl(args.manifest))
    if args.limit:
        rows = rows[:args.limit]
    per_utt = []
    for i, row in enumerate(rows):
        bucket = _bucket(row)
        prompt = en_prompt if bucket == "mono_en" else ja_prompt   # CS + JA use the JA ASR prompt
        wav, sr = sf.read(row["audio_path"], dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        hyp = asr.transcribe(wav, sr, prompt)
        sc = score_row(row["transcript"], hyp, row.get("spans", []))
        sc.update(bucket=bucket, id=row["id"], ref=row["transcript"], hyp=hyp)
        per_utt.append(sc)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} ...")

    summary = aggregate(per_utt)
    _print_summary(f"{args.model}", summary)
    out = {"model": args.model, "manifest": str(args.manifest), "summary": summary, "per_utt": per_utt}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


def run_compare(base_path: str, ft_path: str) -> None:
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    ft = json.loads(Path(ft_path).read_text(encoding="utf-8"))
    _print_summary(f"BASE  {base['model']}", base["summary"])
    _print_summary(f"FT    {ft['model']}", ft["summary"])
    print("\n== Δ (FT − BASE; negative CER/WER/PIER = improvement) ==")
    print(f"{'bucket':10} {'ΔCER':>9} {'ΔWER':>9} {'ΔPIER':>9}")
    for name in ("all", "cs", "mono_ja", "mono_en"):
        if name in base["summary"] and name in ft["summary"]:
            b, f = base["summary"][name], ft["summary"][name]
            def d(k):
                return None if b[k] is None or f[k] is None else round(f[k] - b[k], 4)
            dp = d("PIER")
            print(f"{name:10} {d('CER'):>9} {d('WER'):>9} {('-' if dp is None else dp):>9}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("BASE.json", "FT.json"), help="diff two result JSONs (no model)")
    ap.add_argument("--model", help="model id or checkpoint dir (base model when --adapter is used)")
    ap.add_argument("--adapter", help="LoRA adapter dir to apply on top of --model")
    ap.add_argument("--manifest", help="held-out eval manifest.jsonl")
    ap.add_argument("--out", default="runs/eval.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system-ja", help="override JA ASR prompt")
    ap.add_argument("--system-en", help="override EN ASR prompt")
    args = ap.parse_args()

    if args.compare:
        run_compare(*args.compare)
        return
    if not args.model or not args.manifest:
        ap.error("--model and --manifest are required (or use --compare)")
    run_eval(args)


if __name__ == "__main__":
    main()
