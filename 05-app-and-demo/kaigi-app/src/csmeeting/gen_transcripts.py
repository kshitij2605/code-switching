"""Generate code-switched JP<->EN meeting transcripts.

Three sources, in increasing cost/coverage:
  1. seed bank        — hand-written gold (data/seeds/meeting_codeswitch_seed.jsonl)
  2. templated pairs  — emit each business term BOTH ways (Latin switch vs katakana loanword) to teach the boundary
  3. LLM (optional)   — OpenAI-compatible expansion, few-shot-prompted with seeds + the convention (off by default)

Output: a list of rows {id, transcript, style, domain, source}. Spans are derived later (synthesize) from the script.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from csmeeting.convention import infer_style
from csmeeting.manifest import read_jsonl

# --- templated frames. Each frame has exactly one {X} slot we fill with a term (Latin OR katakana). --------------
NOUN_FRAMES = [
    "その{X}についてもう一度確認しましょう。",
    "来週までに{X}を整理します。",
    "{X}の件、あとで共有しますね。",
    "今の{X}、ちょっと確認させてください。",
    "次の{X}はいつにしましょうか。",
    "この{X}は優先度が高いと思います。",
    "{X}の状況を教えてもらえますか。",
]
VERBAL_FRAMES = [
    "あとで{X}しておきます。",
    "その点は私が{X}します。",
    "では、資料を{X}しておきますね。",
    "今日中に{X}してもらえると助かります。",
]
ACRONYM_FRAMES = [
    "今期の{A}について議論しましょう。",
    "その{A}の数字、あとで共有します。",
    "{A}をどう改善するか考えたいです。",
]


def load_seeds(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for r in read_jsonl(path):
        r.setdefault("source", "seed")
        r.setdefault("style", infer_style(r["transcript"]))
        r.setdefault("domain", "meeting")
        rows.append(r)
    return rows


def load_terms(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("terms", []), data.get("acronyms", [])


def gen_templates(terms: list[dict[str, str]], acronyms: list[str], n: int, *, contrast_pairs: bool,
                  rng: random.Random) -> list[dict[str, Any]]:
    """Produce up to n templated rows. With contrast_pairs, each noun/verbal term is emitted in BOTH scripts."""
    rows: list[dict[str, Any]] = []

    def emit(transcript: str) -> None:
        rows.append({"transcript": transcript, "style": infer_style(transcript),
                     "domain": "meeting", "source": "template"})

    nouns = [t for t in terms if t.get("pos") == "noun"]
    verbals = [t for t in terms if t.get("pos") == "verbal"]

    # build a big candidate pool, then sample down to n
    for t in nouns:
        forms = [t["en"], t["katakana"]] if contrast_pairs else [rng.choice([t["en"], t["katakana"]])]
        for frame in NOUN_FRAMES:
            for form in forms:
                emit(frame.format(X=form))
    for t in verbals:
        forms = [t["en"], t["katakana"]] if contrast_pairs else [rng.choice([t["en"], t["katakana"]])]
        for frame in VERBAL_FRAMES:
            for form in forms:
                emit(frame.format(X=form))
    for a in acronyms:
        for frame in ACRONYM_FRAMES:
            emit(frame.format(A=a))

    # dedupe, shuffle, cap
    seen: set[str] = set()
    unique = []
    for r in rows:
        if r["transcript"] not in seen:
            seen.add(r["transcript"])
            unique.append(r)
    rng.shuffle(unique)
    return unique[:n] if n and n > 0 else unique


def gen_llm(seeds: list[dict[str, Any]], n: int, convention_text: str, rng: random.Random) -> list[dict[str, Any]]:
    """Optional OpenAI-compatible expansion. Returns [] (with a note) if no API key configured."""
    if n <= 0:
        return []
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[gen_llm] OPENAI_API_KEY not set -> skipping LLM generation (seed + template only).")
        return []
    try:
        from openai import OpenAI  # lazy import; only needed with --use-llm
    except ImportError:
        print("[gen_llm] `openai` not installed (pip install -e '.[llm]') -> skipping.")
        return []

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    fewshot = "\n".join(f"- {s['transcript']}" for s in rng.sample(seeds, min(8, len(seeds))))
    sys_prompt = (
        "You generate realistic Japanese business-meeting utterances that contain natural Japanese<->English "
        "code-switching. Follow this transcription convention EXACTLY:\n" + convention_text +
        "\nReturn ONLY a JSON array of strings, each a single natural utterance. No commentary."
    )
    user_prompt = f"Here are examples in the target style:\n{fewshot}\n\nGenerate {n} NEW, varied utterances."
    resp = client.chat.completions.create(
        model=model, temperature=0.9,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
    )
    text = resp.choices[0].message.content or "[]"
    text = text[text.find("["): text.rfind("]") + 1] or "[]"
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        print("[gen_llm] could not parse LLM output as JSON -> skipping.")
        return []
    return [{"transcript": s.strip(), "style": infer_style(s), "domain": "meeting", "source": "llm"}
            for s in items if isinstance(s, str) and s.strip()]


def build(*, seeds_path: str | Path, terms_path: str | Path, n_template: int, contrast_pairs: bool,
          include_seeds: bool, use_llm: bool, llm_n: int, convention_path: str | Path,
          seed: int = 13) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seeds = load_seeds(seeds_path)
    if include_seeds:
        rows.extend(seeds)
    terms, acronyms = load_terms(terms_path)
    rows.extend(gen_templates(terms, acronyms, n_template, contrast_pairs=contrast_pairs, rng=rng))
    if use_llm:
        convention_text = Path(convention_path).read_text(encoding="utf-8")
        rows.extend(gen_llm(seeds, llm_n, convention_text, rng))

    # dedupe by transcript, assign stable ids
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    counters = {"seed": 0, "template": 0, "llm": 0}
    for r in rows:
        t = r["transcript"]
        if t in seen:
            continue
        seen.add(t)
        src = r.get("source", "template")
        if "id" not in r:
            counters[src] = counters.get(src, 0) + 1
            r["id"] = f"{src}-{counters[src]:05d}"
        out.append({k: r[k] for k in ("id", "transcript", "style", "domain", "source")})
    return out
