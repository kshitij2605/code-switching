"""Measure JA->EN translation quality so you can decide whether the MT stage needs fine-tuning.

Uses FLEURS as a text MT benchmark: ja_jp and en_us share a FLoRes `id`, so joining by id gives parallel
(JA text, EN text) pairs — a clean, ASR-independent test of just the translator.
"""

from __future__ import annotations

from collections.abc import Callable


def load_fleurs_pairs(n: int = 200, split: str = "test") -> list[tuple[str, str]]:
    from datasets import load_dataset

    ja = load_dataset("google/fleurs", "ja_jp", split=split)
    en = load_dataset("google/fleurs", "en_us", split=split)
    en_by_id: dict = {}
    for r in en:
        en_by_id.setdefault(r["id"], r["transcription"])
    pairs: list[tuple[str, str]] = []
    seen: set = set()
    for r in ja:
        rid = r["id"]
        if rid in en_by_id and rid not in seen:
            seen.add(rid)
            pairs.append((r["transcription"], en_by_id[rid]))
            if n and len(pairs) >= n:
                break
    return pairs


def evaluate(translate_fn: Callable[[str], str], pairs: list[tuple[str, str]]) -> dict:
    import sacrebleu

    hyps = [translate_fn(ja) for ja, _ in pairs]
    refs = [en for _, en in pairs]
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    samples = [{"ja": pairs[i][0], "ref": refs[i], "hyp": hyps[i]} for i in range(min(5, len(pairs)))]
    return {"n": len(pairs), "bleu": round(bleu, 2), "chrf": round(chrf, 2), "samples": samples}
