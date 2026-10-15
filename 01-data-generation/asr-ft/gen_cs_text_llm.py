#!/usr/bin/env python3
"""Phase 3 code-switched TEXT generator: LLM prompting plus profile-calibrated filtering.

Template corpora (e.g. 300 sentences built from 101 EN and 126 JA segments)
have switching statistics far from natural CS (measured with
common/cs_profile.py). This generator asks an OpenAI-compatible
endpoint (e.g. vLLM serving an Apache-2.0 LLM) for spoken Japanese meeting
utterances with *insertional* English, then keeps only sentences whose
switching matches natural speech:

  * both languages present; English spans of at most --max-en-span words; at
    most --max-switches switch points;
  * English function-word share of the sentence <= --max-func (natural CS inserts
    content words; LLM text over-inserts "the / of / is");
  * Japanese length within bounds. A sentence whose English appears only in
    katakana has no Latin token, so it fails the both-languages test: references
    keep English in Latin script;
  * no overlap with protected test/dev data (common/leak_guard.py), and exact
    de-duplication.
The accepted corpus's profile is printed against the natural targets.

--candidates FILE skips generation and filters lines from a file (testing, or
text generated elsewhere).

Output JSONL (one row per sentence, the renderer input format):
  {id, transcript, segments, en_words, ja_text, n_switches, domain, scenario,
   generator, model, license_source/license/license_use}

Example:
  python gen_cs_text_llm.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-32B \
      --n-per-call 20 --calls 50 --out data/text/cs_llm_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.cs_profile import EN_FUNCTION, profile, tag_tokens  # noqa: E402
from common.cs_segments import segments  # noqa: E402
from common.leak_guard import LeakGuard, text_hash  # noqa: E402
from common.licenses import tag_row  # noqa: E402

DOMAINS = {
    "it_dev": ("software team stand-up or code review",
               "deploy release pull request merge review branch bug fix test coverage refactor "
               "latency API endpoint database migration rollback hotfix staging production "
               "backlog sprint ticket spec requirement edge case log"),
    "incident": ("incident review / postmortem",
                 "incident outage root cause alert monitoring dashboard escalation workaround "
                 "impact timeline SLA on-call rollback patch retry timeout"),
    "procurement": ("IT procurement and vendor meeting",
                    "vendor quote contract license budget invoice renewal cost proposal "
                    "evaluation PoC requirement deadline approval support plan"),
    "sales": ("sales and customer meeting",
              "customer pipeline deal target forecast KPI demo proposal pricing discount "
              "follow up feedback onboarding churn upsell campaign"),
    "general": ("general office meeting",
                "meeting agenda slide schedule deadline action item task priority update "
                "idea point issue plan option decision summary team"),
}
PROMPT = """あなたは日本のIT企業で働くバイリンガル社員の話し言葉を書きます。
場面: {scene}
次のような発話を{n}個、1行に1つずつ書いてください。番号や引用符は付けないでください。
- 主に日本語で、英語の単語や短いフレーズを1〜3個、実際のバイリンガル社員のように自然に混ぜる
- 英語は主に名詞、または「〜する」を付けた動詞、「〜な」を付けた形容詞として使う
- 英語はローマ字（Latin）で書き、カタカナにしない
- the / of / is / to などの英語の機能語はほとんど使わない
- 長さはばらばらに（短い相づち程度から2文程度まで）。ときどき「えーと」「あの」などのフィラーも入れる
- 3つに1つくらいは英語の単語から始めてもよい
使ってもよい英単語の例: {terms}
"""
_LINE_JUNK = re.compile(r"^\s*(?:[-*・•]|\d+[.)、．:]|[「『\"'])\s*|[」』\"']\s*$")


def clean_line(line: str) -> str:
    return _LINE_JUNK.sub("", line.strip()).strip()


def accept(text: str, args) -> tuple[bool, str]:
    toks = tag_tokens(text)
    langs = [l for _, l in toks if l != "x"]
    if "en" not in langs or "ja" not in langs:
        return False, "monolingual"
    en = [t for t, l in toks if l == "en"]
    spans, cur, n = [], None, 0
    for l in langs:
        if l == cur:
            n += 1
        else:
            if cur == "en":
                spans.append(n)
            cur, n = l, 1
    if cur == "en":
        spans.append(n)
    switches = sum(1 for a, b in zip(langs, langs[1:]) if a != b)
    if max(spans) > args.max_en_span:
        return False, "en_span"
    if switches > args.max_switches:
        return False, "switches"
    if sum(1 for t in en if t in EN_FUNCTION) / len(en) > args.max_func:
        return False, "function_words"
    ja_chars = sum(1 for t, l in toks if l == "ja" for _ in t)
    if not args.min_ja_chars <= ja_chars <= args.max_ja_chars:
        return False, "length"
    return True, "ok"


def generate(args, rng) -> list[tuple[str, str, str]]:
    """[(line, domain, scene)] from the endpoint, or from --candidates."""
    if args.candidates:
        return [(clean_line(l), "file", "file") for l in open(args.candidates, encoding="utf-8")
                if clean_line(l)]
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="none")
    out = []
    for call in range(args.calls):
        dom = rng.choice(sorted(DOMAINS))
        scene, terms = DOMAINS[dom]
        pick = rng.sample(terms.split(), k=min(8, len(terms.split())))
        msg = PROMPT.format(scene=scene, n=args.n_per_call, terms=", ".join(pick))
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": msg}],
                    temperature=args.temperature, max_tokens=2048,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S)
                out += [(clean_line(l), dom, scene) for l in text.splitlines() if clean_line(l)]
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [call {call} attempt {attempt + 1}] {type(e).__name__}: {e}")
                time.sleep(2 * (attempt + 1))
        if (call + 1) % 10 == 0:
            print(f"  {call + 1}/{args.calls} calls, {len(out)} candidate lines", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--n-per-call", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--candidates", default=None, help="filter lines from this file instead of generating")
    ap.add_argument("--max-en-span", type=int, default=3)
    ap.add_argument("--max-switches", type=int, default=6)
    ap.add_argument("--max-func", type=float, default=0.34, help="per-sentence EN function-word share")
    ap.add_argument("--min-ja-chars", type=int, default=4)
    ap.add_argument("--max-ja-chars", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cands = generate(args, rng)
    guard = LeakGuard()
    seen, rows, why = set(), [], {}
    for line, dom, scene in cands:
        h = text_hash(line)
        if not h or h in seen:
            why["duplicate"] = why.get("duplicate", 0) + 1
            continue
        seen.add(h)
        ok, reason = accept(line, args)
        if ok and guard.check_row({"transcript": line}):
            ok, reason = False, "protected_overlap"
        why[reason] = why.get(reason, 0) + 1
        if not ok:
            continue
        segs = segments(line)
        rows.append(tag_row({
            "id": f"llmcs_{len(rows):06d}", "transcript": line, "segments": segs,
            "en_words": [w for s in segs if s["lang"] == "en" for w in s["text"].split()],
            "ja_text": "".join(s["text"] for s in segs if s["lang"] == "ja"),
            "n_switches": max(0, len(segs) - 1), "domain": dom, "scenario": scene,
            "generator": "llm-insertional-v1", "model": args.model if not args.candidates else "file"},
            "llm/apache-2.0"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"candidates {len(cands)} -> accepted {len(rows)}; reasons {why} -> {args.out}")
    if rows:
        p = profile([r["transcript"] for r in rows])
        keys = ("cmi", "i_index", "en_span_len", "single_word_insert", "en_func_share", "en_start")
        print("profile:", {k: round(p[k], 3) for k in keys})
        print("natural targets: en_func_share <= ~0.12, "
              "single_word_insert 0.65-0.75, i_index 0.18-0.26, en_start 0.25-0.35")
    return 0


if __name__ == "__main__":
    sys.exit(main())
