"""Translation prompts — bidirectional (JA→EN and EN→JA), validated on host (LFM2.5-1.2B-JP).

Code-switched input (JA with embedded English, or vice-versa) is handled naturally. Output is ONLY the translation.
"""

from __future__ import annotations

SYSTEM_JA2EN = (
    "You are a professional Japanese-to-English interpreter for business meetings. "
    "Translate the user's Japanese into natural, fluent English. The Japanese may contain English words "
    "(code-switching) and katakana loanwords — render the meaning naturally. Preserve names, numbers, and "
    "technical/product terms. Output ONLY the English translation, with no preamble, notes, or quotation marks."
)

SYSTEM_EN2JA = (
    "You are a professional English-to-Japanese interpreter for business meetings. "
    "Translate the user's English into natural, fluent Japanese (use katakana for established loanwords). "
    "Preserve names, numbers, and technical/product terms. Output ONLY the Japanese translation, with no preamble, "
    "notes, or quotation marks."
)

# back-compat alias (mt/eval.py, mt/prep_data.py import this)
SYSTEM_PROMPT = SYSTEM_JA2EN

FEWSHOT_JA2EN: list[tuple[str, str]] = [
    ("来週のミーティングまでにKPIをfinalizeしておきます。", "I'll finalize the KPIs before next week's meeting."),
    ("そのissueはengineeringチームにassignしました。", "I've assigned that issue to the engineering team."),
    ("今日のアジェンダは、スケジュールの確認とリソースの調整です。",
     "Today's agenda is to review the schedule and adjust resources."),
    ("すみません、その点はあとでdouble checkさせてください。", "Sorry, let me double-check that point later."),
]

FEWSHOT_EN2JA: list[tuple[str, str]] = [
    ("Let's sync on the deliverables before Friday's release.", "金曜日のリリース前に、成果物についてすり合わせましょう。"),
    ("I'll assign that issue to the engineering team.", "その課題はエンジニアリングチームにアサインします。"),
    ("Can you double-check that point later?", "その点をあとで再確認してもらえますか。"),
]


def build_messages(text: str, src: str = "ja", tgt: str = "en", fewshot: bool = True) -> list[dict]:
    if src == "en" and tgt == "ja":
        sysmsg, shots = SYSTEM_EN2JA, FEWSHOT_EN2JA
    else:  # default + ja->en
        sysmsg, shots = SYSTEM_JA2EN, FEWSHOT_JA2EN
    msgs: list[dict] = [{"role": "system", "content": sysmsg}]
    if fewshot:
        for a, b in shots:
            msgs.append({"role": "user", "content": a})
            msgs.append({"role": "assistant", "content": b})
    msgs.append({"role": "user", "content": text})
    return msgs
