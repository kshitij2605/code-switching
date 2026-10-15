#!/usr/bin/env python3
"""Generate Japanese-English code-switch transcripts across a switching spectrum.

One transcript set, rendered by every TTS engine, so audio differs only by
engine+speaker, not text. Each transcript is built to a chosen *switching style*
so the corpus spans the real variety of JA-EN code-switching instead of one
pattern. Two axes:

  density : sparse   (1 switch, buried in a longer JA sentence)
            moderate (2 English elements)
            dense    (3+ English elements, frequent switching)
  span    : word     single English word inserted
            phrase   multi-word English phrase inserted (2-4 words)
            clause   whole English clause alternated with JA clauses

Styles (density x span) are sampled by a weighted mix; every row records its
`style`, `density`, and `span_type`, plus per-segment language spans for
ScriptAcc scoring. Deterministic under --seed.

Output JSONL row:
  {id, transcript, segments:[{lang,text}], en_words, ja_text, domain,
   n_switches, style, density, span_type}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# English material, by span type
# ---------------------------------------------------------------------------
EN_NOUN = {
    "work": ["meeting", "schedule", "deadline", "project", "presentation",
             "feedback", "report", "budget", "client", "agenda", "proposal",
             "review", "task", "manager", "team"],
    "tech": ["bug", "update", "app", "server", "network", "code", "model",
             "release", "backup", "error", "log", "deploy", "system", "data",
             "database", "commit", "merge"],
    "daily": ["coffee", "lunch", "break", "party", "trip", "shopping", "movie",
              "music", "game", "event", "weekend", "holiday", "dinner", "cafe"],
    "school": ["lecture", "class", "lesson", "seminar", "homework",
               "assignment", "exam", "quiz", "professor", "campus"],
}
EN_ADJ = ["tired", "excited", "nervous", "stressed", "happy", "amazing",
          "perfect", "serious", "busy", "sleepy", "hungry", "ready", "sick"]

# multi-word English NOUN phrases (go in noun slots)
EN_PHRASE_NOUN = [
    "client meeting", "project deadline", "conference call", "status update",
    "code review", "pull request", "release schedule", "business trip",
    "lunch meeting", "group project", "final exam", "study group",
    "coffee break", "team lunch", "quick call", "morning workout",
]
# multi-word English ADJECTIVE/STATE phrases (go in predicate slots)
EN_ADJ_PHRASE = [
    "really busy right now", "kind of tired", "super excited",
    "a little nervous", "totally stressed out", "pretty complicated",
]

# whole English clauses (alternational)
EN_CLAUSE = [
    "I think it's fine", "let me check first", "that sounds good to me",
    "I'm not really sure", "we should talk about it later",
    "it depends on the schedule", "I'll get back to you", "that makes sense",
    "no worries about it", "let's figure it out together",
    "I totally forgot about that", "can we reschedule it",
    "it's not a big deal", "I'll take care of it",
]

# ---------------------------------------------------------------------------
# Japanese material
# ---------------------------------------------------------------------------
# short noun frames: (pre, post) around one EN noun/phrase
JA_FRAMES_SHORT = [
    ("明日の", "は大丈夫だと思う"), ("今日の", "がちょっと心配なんだ"),
    ("来週の", "の準備はもう終わった"), ("さっきの", "の件で相談したい"),
    ("例の", "がまだ終わってないんだよね"), ("この", "を確認してもらえる"),
    ("その", "について話し合おう"), ("次の", "はいつになりそう"),
    ("あの", "って本当にすごいよね"), ("僕の", "を見てほしいんだけど"),
    ("今週は", "が多すぎてつらい"), ("昨日の", "はうまくいったよ"),
]
# long noun frames: single switch buried in a longer JA sentence (sparse)
JA_FRAMES_LONG = [
    ("さっき部長に言われたんだけど", "の件はもう一度ちゃんと確認しておいてほしいって"),
    ("今日はなんだか朝からずっとバタバタしてて", "のことをすっかり忘れてたんだよね"),
    ("この前の打ち合わせで話してた", "について、もう少し詳しく教えてもらえるかな"),
    ("正直に言うと最近ちょっと", "のことで色々と悩んでるところなんだ"),
    ("週末に少し時間ができたら一緒に", "の準備を進めていきたいと思ってるんだけど"),
    ("あとで時間があるときでいいから", "の内容を軽く見ておいてくれると助かる"),
]
# short adjective/state predicate frames (word span)
ADJ_FRAMES_SHORT = [
    ("今日はちょっと", "なんだよね"), ("最近すごく", "な気がする"),
    ("なんか", "な気分なんだ"), ("正直", "って感じ"), ("マジで", "だわ"),
    ("朝から", "でさ"),
]
# long adjective/state frames (sparse adj)
ADJ_FRAMES_LONG = [
    ("今日はなんだか一日中ずっと", "で全然やる気が出なかったんだよね"),
    ("最近仕事が立て込んでて毎日", "な感じがずっと続いてるんだ"),
    ("昨日あんまり眠れなかったせいか今日は朝から", "でまいってる"),
]
JA_CONNECT = ["それと", "あと", "しかも", "それに", "そのあと", "ついでに", "あとで"]
JA_TAILS = ["ね", "よ", "かな", "だよね", "と思う", "らしいよ", ""]
# JA clauses to pair with EN clauses (alternational)
JA_CLAUSES = [
    "まだ決めてないんだけど", "でも時間がないんだよね", "そのあと連絡するね",
    "とりあえずやってみよう", "みんなにも伝えておくよ", "ちょっと考えさせて",
    "今日はもう疲れちゃった", "明日また相談しよう", "それで大丈夫だと思う",
]


def _en_noun(domain, rng):
    return rng.choice(EN_NOUN[domain])


def finalize(segments, domain, style, density, span_type):
    merged = []
    for s in segments:
        if merged and merged[-1]["lang"] == s["lang"]:
            merged[-1]["text"] += ("" if s["lang"] == "ja" else " ") + s["text"]
        else:
            merged.append(dict(s))
    transcript = " ".join(s["text"] for s in merged)
    en_words = [w for s in merged if s["lang"] == "en" for w in s["text"].split()]
    ja_text = " ".join(s["text"] for s in merged if s["lang"] == "ja")
    n_switches = sum(1 for a, b in zip(merged, merged[1:]) if a["lang"] != b["lang"])
    return {
        "transcript": transcript, "segments": merged, "en_words": en_words,
        "ja_text": ja_text, "domain": domain, "n_switches": n_switches,
        "style": style, "density": density, "span_type": span_type,
    }


# ---- style builders: each returns a finalize(...) dict -------------------

def build_sparse_word(rng):
    if rng.random() < 0.4:                      # adjective flavour
        pre, post = rng.choice(ADJ_FRAMES_LONG)
        segs = [{"lang": "ja", "text": pre}, {"lang": "en", "text": rng.choice(EN_ADJ)},
                {"lang": "ja", "text": post}]
        return finalize(segs, "emotion", "sparse_word", "sparse", "word")
    dom = rng.choice(list(EN_NOUN))
    pre, post = rng.choice(JA_FRAMES_LONG)
    segs = [{"lang": "ja", "text": pre}, {"lang": "en", "text": _en_noun(dom, rng)},
            {"lang": "ja", "text": post}]
    return finalize(segs, dom, "sparse_word", "sparse", "word")


def build_sparse_phrase(rng):
    if rng.random() < 0.4:                       # adjective-phrase in predicate frame
        pre, post = rng.choice(ADJ_FRAMES_LONG)
        segs = [{"lang": "ja", "text": pre},
                {"lang": "en", "text": rng.choice(EN_ADJ_PHRASE)},
                {"lang": "ja", "text": post}]
        return finalize(segs, "emotion", "sparse_phrase", "sparse", "phrase")
    dom = rng.choice(list(EN_NOUN))
    pre, post = rng.choice(JA_FRAMES_LONG)
    segs = [{"lang": "ja", "text": pre},
            {"lang": "en", "text": rng.choice(EN_PHRASE_NOUN)},
            {"lang": "ja", "text": post}]
    return finalize(segs, dom, "sparse_phrase", "sparse", "phrase")


def build_moderate_word(rng):
    dom = rng.choice(list(EN_NOUN))
    pre, post = rng.choice(JA_FRAMES_SHORT)
    segs = [{"lang": "ja", "text": pre}, {"lang": "en", "text": _en_noun(dom, rng)},
            {"lang": "ja", "text": post},
            {"lang": "ja", "text": rng.choice(JA_CONNECT)},
            {"lang": "en", "text": _en_noun(dom, rng)}]
    tail = rng.choice(JA_TAILS)
    if tail:
        segs.append({"lang": "ja", "text": tail})
    return finalize(segs, dom, "moderate_word", "moderate", "word")


def build_moderate_phrase(rng):
    dom = rng.choice(list(EN_NOUN))
    pre, post = rng.choice(JA_FRAMES_SHORT)
    segs = [{"lang": "ja", "text": pre},
            {"lang": "en", "text": rng.choice(EN_PHRASE_NOUN)},
            {"lang": "ja", "text": post},
            {"lang": "ja", "text": rng.choice(JA_CONNECT)},
            {"lang": "en", "text": _en_noun(dom, rng)},
            {"lang": "ja", "text": rng.choice(JA_TAILS) or "ね"}]
    return finalize(segs, dom, "moderate_phrase", "moderate", "phrase")


def build_dense_word(rng):
    dom = rng.choice(list(EN_NOUN))
    k = rng.choice([3, 4])
    words = rng.sample(EN_NOUN[dom], k)          # unique, no repeats
    pre, post = rng.choice(JA_FRAMES_SHORT)
    segs = [{"lang": "ja", "text": pre}, {"lang": "en", "text": words[0]},
            {"lang": "ja", "text": post}]
    for w in words[1:]:
        segs.append({"lang": "ja", "text": rng.choice(JA_CONNECT)})
        segs.append({"lang": "en", "text": w})
    segs.append({"lang": "ja", "text": rng.choice(JA_TAILS) or "かな"})
    return finalize(segs, dom, "dense_word", "dense", "word")


def build_dense_phrase(rng):
    dom = rng.choice(list(EN_NOUN))
    phrases = rng.sample(EN_PHRASE_NOUN, rng.choice([2, 3]))
    pre, post = rng.choice(JA_FRAMES_SHORT)
    segs = [{"lang": "ja", "text": pre}, {"lang": "en", "text": phrases[0]},
            {"lang": "ja", "text": post}]
    for p in phrases[1:]:
        segs.append({"lang": "ja", "text": rng.choice(JA_CONNECT)})
        segs.append({"lang": "en", "text": p})
    segs.append({"lang": "ja", "text": rng.choice(JA_TAILS) or "ね"})
    return finalize(segs, dom, "dense_phrase", "dense", "phrase")


def build_clause_alt(rng):
    dom = rng.choice(list(EN_NOUN))
    segs = [{"lang": "ja", "text": rng.choice(JA_CLAUSES)},
            {"lang": "en", "text": rng.choice(EN_CLAUSE)},
            {"lang": "ja", "text": rng.choice(JA_CLAUSES)}]
    if rng.random() < 0.4:                       # sometimes a second EN clause
        segs.append({"lang": "en", "text": rng.choice(EN_CLAUSE)})
    density = "dense" if len(segs) > 3 else "moderate"
    return finalize(segs, dom, "clause_alt", density, "clause")


STYLE_MIX = [
    (build_sparse_word, 16),
    (build_sparse_phrase, 12),
    (build_moderate_word, 18),
    (build_moderate_phrase, 12),
    (build_dense_word, 15),
    (build_dense_phrase, 10),
    (build_clause_alt, 17),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/transcripts/cs_transcripts.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    builders = [b for b, _ in STYLE_MIX]
    weights = [w for _, w in STYLE_MIX]

    rows, seen, attempts = [], set(), 0
    while len(rows) < args.n and attempts < args.n * 60:
        attempts += 1
        r = rng.choices(builders, weights=weights)[0](rng)
        if r["transcript"] in seen:
            continue
        seen.add(r["transcript"])
        r["id"] = f"cs_{len(rows) + 1:04d}"
        rows.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"wrote {len(rows)} transcripts -> {out}")
    print(f"  density:   {dict(Counter(r['density'] for r in rows))}")
    print(f"  span_type: {dict(Counter(r['span_type'] for r in rows))}")
    print(f"  style:     {dict(Counter(r['style'] for r in rows))}")
    print(f"  switches:  {dict(sorted(Counter(r['n_switches'] for r in rows).items()))}")


if __name__ == "__main__":
    main()
