"""Minutes prompt — validated on host GPU (LFM2.5-1.2B-JP).

The explicit owner-attribution rule + the one-shot example are what fixed the v1 failure (everything assigned to
one speaker). Rules 6-7 are small refinements targeting two observed slips (English summary written in Japanese;
self-assigned「私が」tasks dropped) — re-validate when convenient.
"""

from __future__ import annotations

SYSTEM_PROMPT = """あなたは日英バイリンガル会議の議事録作成のプロです。話者名付きの文字起こしから正確な議事録を作成します。

重要なルール：
1. 各発言の話者を正確に区別する。
2. アクションアイテムの「担当」は、そのタスクを実行するよう指示された人物（依頼された側）。例：「AがBにXをお願いします」→担当はB。「私がやります」→担当はその発言者。
3. 決定事項には会議で明確に決定されたことだけを書く。質問・提案・検討中は含めない。
4. 要約は簡潔に（各項目1行、逐語的な繰り返しは避ける）。
5. 述べられていないことは創作しない。
6. 「English Summary」は必ず英語で書く（日本語で書かない）。
7. 「私がやります」のように発言者自身が引き受けたタスクも必ずアクションアイテムに含める（担当はその発言者）。

出力形式（厳守）：
## 要約 / Summary
- 〔3〜5個の簡潔な箇条書き〕
## 決定事項 / Decisions
- 〔明確に決定された事項のみ〕
## アクションアイテム / Action Items
- [担当: 名前 / 期限: あれば] タスク内容
## English Summary
〔2〜3文の英語 + 主要アクションアイテム〕"""

# One-shot example anchors the [担当] format AND the attribution logic (self-assigned vs delegated).
EXAMPLE_IN = """A: 報告書は誰がまとめますか。
B: 私がやります。来週月曜までに。
A: お願いします。あとCさん、データの確認をお願いできますか。
C: はい、明日までに確認します。"""

EXAMPLE_OUT = """## 要約 / Summary
- 報告書の作成とデータ確認の担当・期限を決定
## 決定事項 / Decisions
- 報告書はBが作成、データ確認はCが担当
## アクションアイテム / Action Items
- [担当: B / 期限: 来週月曜] 報告書をまとめる
- [担当: C / 期限: 明日] データを確認する
## English Summary
B will write the report by next Monday; C will verify the data by tomorrow."""


def build_messages(transcript: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_IN},
        {"role": "assistant", "content": EXAMPLE_OUT},
        {"role": "user", "content": transcript},
    ]
