#!/usr/bin/env python3
"""Characterise the linguistic structure of the generated CS transcripts."""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "data/transcripts/cs_transcripts.jsonl"
rows = [json.loads(l) for l in open(path)]
n = len(rows)

sw = Counter(r["n_switches"] for r in rows)
dom = Counter(r["domain"] for r in rows)

en_seg_lens, ja_seg_chars, en_ins_per_utt = [], [], []
for r in rows:
    en_ins = 0
    for s in r["segments"]:
        if s["lang"] == "en":
            en_seg_lens.append(len(s["text"].split()))
            en_ins += 1
        else:
            ja_seg_chars.append(len(s["text"]))
    en_ins_per_utt.append(en_ins)

print(f"transcripts: {n}")
print(f"switches/utt (language boundaries): {dict(sorted(sw.items()))}")
print(f"EN insertions/utt:                  {dict(sorted(Counter(en_ins_per_utt).items()))}")
print(f"EN segment length (words):          {dict(sorted(Counter(en_seg_lens).items()))}  (1=word, >1=phrase)")
jc = sorted(ja_seg_chars)
print(f"JA segment length (chars):          min={jc[0]} med={jc[len(jc)//2]} max={jc[-1]}")
print(f"domains: {dict(dom)}")
if "style" in rows[0]:
    print()
    print("--- examples per style ---")
    by_style = {}
    for r in rows:
        by_style.setdefault(r["style"], []).append(r)
    order = ["sparse_word", "sparse_phrase", "moderate_word", "moderate_phrase",
             "dense_word", "dense_phrase", "clause_alt"]
    for st in [s for s in order if s in by_style] + [s for s in by_style if s not in order]:
        ex = by_style[st][:2]
        d, sp, nsw = ex[0]["density"], ex[0]["span_type"], ex[0]["n_switches"]
        print(f"  [{st}]  density={d} span={sp} switches~{nsw}")
        for r in ex:
            print(f"     {r['transcript']}")
else:
    print()
    print("--- one example per switch count ---")
    seen = set()
    for r in sorted(rows, key=lambda x: x["n_switches"]):
        k = r["n_switches"]
        if k in seen:
            continue
        seen.add(k)
        struct = "  ".join(f'{s["lang"]}[{s["text"]}]' for s in r["segments"])
        print(f"  sw={k}: {r['transcript']}")
        print(f"         {struct}")
