# Confidential bilingual minutes — design + test-first verdict

> **⚡ Update — minutes now run on the audio backbone via a fine-tuned LoRA, not the separate text model.**
> The original stage used base `LFM2.5-1.2B-JP` + the prompt below. It's been replaced by a **minutes LoRA** on
> `LFM2.5-Audio-1.5B-JP` (text→text), so the dedicated text model is dropped. The prompt here is still the system
> prompt used for both training and inference. Full data/iteration/results:
> [`../documentation/finetuning-minutes.md`](../documentation/finetuning-minutes.md).

The third product pillar: a (possibly code-switched) meeting transcript → structured **bilingual minutes**
(要約 / Summary · 決定事項 / Decisions · アクションアイテム / Action Items with owners + due dates · English Summary),
generated **locally** so confidential transcripts never leave the device.

## Verdict (validated on host GPU, GPU 0 / H100)

- **Use `LFM2.5-1.2B-JP` + the prompt in `src/csmeeting/minutes/prompt.py` (system + one-shot). No fine-tune needed
  for the demo.** ~1.9 s per meeting.
- The **explicit owner-attribution rule + one-shot example** were decisive: without them, the base model assigned
  *every* action item to a single speaker; with them, attribution is mostly correct
  (`[担当: Smith] bug`, `[担当: 佐藤] QA coverage`, `[担当: 鈴木] SSO check`).
- **Do NOT use `LFM2.5-1.2B-Thinking` for minutes** — it over-reasons (circular chain-of-thought) and fails to emit
  the minutes within budget; slower and unusable here.

### Known remaining gaps (promptable / SFT-upgradable)
- Occasionally **misses an action item** (e.g. a self-assigned「私が」task or the docs owner).
- Sometimes lists a **question as a decision**.
- Once wrote the **English Summary in Japanese**.

Prompt rules 6–7 target the last two; the completeness gap is the main reason a small **SFT** is the upgrade path
**if** an in-domain eval later demands production quality. Candidate data
(commercial-safe: AMI + QMSum + Aya-ja + dolly-ja + synthetic transcript→minutes).

## Usage

```bash
pip install -e ".[mt]"
python scripts/minutes_demo.py            # built-in sample transcripts
python scripts/minutes_demo.py --transcript meeting.txt
python scripts/minutes_demo.py --mock     # inspect the prompt, no model
```

In code:
```python
from csmeeting.minutes.generate import MinutesGenerator
minutes = MinutesGenerator().generate(transcript_text)
```

## Role in the app

The cascade already produces the JA transcript (ASR) per utterance; accumulate the turns into a running transcript
and call `MinutesGenerator.generate(...)` on demand (or at meeting end) to render the minutes panel. All on-device.
