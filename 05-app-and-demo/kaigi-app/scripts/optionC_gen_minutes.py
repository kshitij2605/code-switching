"""Option C: generate (transcript -> minutes) training pairs for a minutes LoRA on the audio backbone.

Built from structured facts so labels are correct by construction (esp. owner attribution: delegated → the
asked person; self-assigned 「私がやります」→ the speaker). Transcripts are varied bilingual (JA + code-switch);
gold minutes follow the exact app format. Output: optionC/data/training/{train,eval}.jsonl  ({transcript, minutes}).
"""
import json, random
from pathlib import Path

random.seed(7)
OUT = Path("/data/lfm2.5/run-c/optionC/data/training"); OUT.mkdir(parents=True, exist_ok=True)

NAMES = ["田中", "鈴木", "佐藤", "山田", "高橋", "中村", "小林", "加藤", "渡辺", "伊藤", "松本", "井上"]
TOPICS = [("新製品のローンチ", "the new product launch"), ("第3四半期の予算レビュー", "the Q3 budget review"),
          ("システム移行の進捗", "the system migration progress"), ("採用計画", "the hiring plan"),
          ("マーケティング戦略", "the marketing strategy"), ("カスタマーサポートの改善", "improving customer support"),
          ("年度末の決算準備", "year-end financial closing"), ("新しいオフィスの移転", "the office relocation"),
          ("セキュリティ監査の対応", "the security audit response"), ("パートナー契約の更新", "the partner contract renewal")]
TASKS = [("資料をまとめる", "prepare the materials"), ("予算案を作成する", "draft the budget proposal"),
         ("スケジュールを更新する", "update the schedule"), ("APIのバグを修正する", "fix the API bug"),
         ("顧客にメールを送る", "email the client"), ("レポートをレビューする", "review the report"),
         ("デザインを確認する", "check the design"), ("テストを実施する", "run the tests"),
         ("見積もりを取る", "get a quote"), ("議事録をまとめる", "write up the minutes"),
         ("KPIを設定する", "set the KPIs"), ("リリースノートを書く", "write the release notes")]
DEADLINES = [("明日", "tomorrow"), ("今週中", "by the end of this week"), ("来週月曜", "by next Monday"),
             ("金曜まで", "by Friday"), ("月末まで", "by the end of the month"), ("来週中", "by next week")]
DECISIONS = [("予算を承認する", "approved the budget"), ("リリース日を金曜に決定する", "set the release date to Friday"),
             ("新しいツールを導入する", "adopt the new tool"), ("外部ベンダーに委託する", "outsource to an external vendor"),
             ("採用を2名に増やす", "increase hiring to two people"), ("プランAで進める", "proceed with Plan A")]
# code-switch fillers (spoken Japanese with embedded English), inserted into discussion lines
CS = ["スケジュールが tight なので", "この feature の priority を上げて", "budget の確認が必要で",
      "next steps を整理すると", "deadline は厳しいですが", "feedback を反映して", "リスクを review して"]


def render(facts):
    """Bare utterance lines (NO speaker labels) — matches the app's ASR transcript (no diarization).
    Owners are named inside the utterance ('鈴木さんに…お願いします') so attribution is inferable; some
    deadlines/owners are omitted to mirror real speech."""
    topic_ja, _ = facts["topic"]
    lines = [random.choice([f"本日は{topic_ja}について話し合います。", f"今日のagendaは{topic_ja}です。",
                            f"それでは{topic_ja}の件、始めましょう。"])]
    lines.append(f"{random.choice(CS)}、まず現状を共有します。")
    if random.random() < 0.7:
        lines.append(f"そうですね、{random.choice(CS)}、進めたいと思います。")
    for d_ja, _ in facts["decisions"]:
        lines.append(random.choice([f"では、{d_ja}ということで決定でよろしいでしょうか。",
                                    f"{d_ja}方向で合意ということにしましょう。"]))
        lines.append(random.choice(["はい、承認します。", "異議なしです。", "了解しました、それで進めましょう。"]))
    for ai in facts["actions"]:
        owner, (t_ja, _), (dl_ja, _), kind = ai["owner"], ai["task"], ai["deadline"], ai["kind"]
        has_dl = ai["has_deadline"]
        dl = (dl_ja if dl_ja.endswith(("まで", "中")) else dl_ja + "まで") if has_dl else ""
        when = f"{dl}に" if has_dl else ""
        if kind == "delegated":
            lines.append(random.choice([f"{owner}さん、{t_ja}のを{when}お願いできますか。",
                                        f"{t_ja}のは{owner}さんに{when}お願いしたいです。"]))
            lines.append(random.choice(["はい、承知しました。", f"わかりました、{when}対応します。" if has_dl else "わかりました、対応します。"]))
        elif kind == "self":  # no speaker label → owner stays 私
            lines.append(f"{t_ja}のは私が{when}やっておきます。")
        else:  # unassigned — no owner stated → minutes should mark 担当: 未指定
            lines.append(random.choice([f"{t_ja}のを{when}お願いします。", f"{t_ja}のが{when}必要ですね。"]))
    return "\n".join(lines)


def gold_minutes(facts):
    topic_ja, topic_en = facts["topic"]
    summ = [f"{topic_ja}について議論"]
    for d_ja, _ in facts["decisions"]:
        summ.append(f"{d_ja}ことを決定")
    for ai in facts["actions"][:2]:
        summ.append(f"{ai['owner']}が{ai['task'][0]}を担当")
    dec = [f"{d_ja}" for d_ja, _ in facts["decisions"]] or ["特になし"]
    acts = []
    for ai in facts["actions"]:
        head = f"担当: {ai['owner']} / 期限: {ai['deadline'][0]}" if ai["has_deadline"] else f"担当: {ai['owner']}"
        acts.append(f"- [{head}] {ai['task'][0]}")
    if not acts:
        acts = ["- 特になし"]
    dec_en = "; ".join(d_en for _, d_en in facts["decisions"])
    en_owner = {"未指定": "someone", "私": "I"}
    act_en = "; ".join((f"{en_owner.get(ai['owner'], ai['owner'])} will {ai['task'][1]} {ai['deadline'][1]}" if ai["has_deadline"]
                        else f"{en_owner.get(ai['owner'], ai['owner'])} will {ai['task'][1]}") for ai in facts["actions"])
    eng = f"The team discussed {topic_en}."
    if dec_en: eng += f" They {dec_en}."
    if act_en: eng += f" Action items: {act_en}."
    return ("## 要約 / Summary\n" + "\n".join(f"- {s}" for s in summ[:5]) +
            "\n## 決定事項 / Decisions\n" + "\n".join(f"- {d}" for d in dec) +
            "\n## アクションアイテム / Action Items\n" + "\n".join(acts) +
            "\n## English Summary\n" + eng)


def make_one():
    n = random.randint(3, 4)
    p = random.sample(NAMES, n)
    topic = random.choice(TOPICS)
    # include zero-decision AND zero/one-action meetings so the model learns to NOT pad on sparse transcripts
    ndec = random.choice([0, 0, 1, 1, 2]); nact = random.choice([0, 1, 1, 2, 2, 3, 4])
    decisions = random.sample(DECISIONS, ndec)
    actions = []
    for _ in range(nact):
        kind = random.choice(["delegated", "delegated", "self", "unassigned"])
        owner = {"self": "私", "unassigned": "未指定"}.get(kind) or random.choice(p)
        actions.append({"owner": owner, "task": random.choice(TASKS), "deadline": random.choice(DEADLINES),
                        "kind": kind, "has_deadline": random.random() < 0.75})
    facts = {"participants": p, "topic": topic, "decisions": decisions, "actions": actions}
    return {"transcript": render(facts), "minutes": gold_minutes(facts)}


def main():
    rows = [make_one() for _ in range(560)]
    # dedupe identical transcripts
    seen, uniq = set(), []
    for r in rows:
        if r["transcript"] not in seen:
            seen.add(r["transcript"]); uniq.append(r)
    n_eval = 40
    eval_, train = uniq[:n_eval], uniq[n_eval:]
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    (OUT / "eval.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in eval_), encoding="utf-8")
    print(f"Train {len(train)} | Eval {len(eval_)}  (unique {len(uniq)}/{len(rows)})")
    print("\n--- sample transcript ---\n" + train[0]["transcript"])
    print("\n--- sample minutes ---\n" + train[0]["minutes"])


if __name__ == "__main__":
    main()
