# Fine-tuning: Meeting Minutes on the audio backbone

How the **minutes** LoRA was built — so the same `LFM2.5-Audio-1.5B-JP` audio model generates the bilingual minutes
(text→text), letting the separate `LFM2.5-1.2B-JP` text model be **dropped entirely**.

- **Method:** LoRA (PEFT). Same recipe as the other adapters (r16 / α32, 0.85 % trainable).
- **Deployed adapter:** `optionC/checkpoints/minutes_lora_v5/final` (~48 MB).
- **Scripts (mirrored in `scripts/optionC_*.py`):** `optionC_gen_minutes.py`, `optionC_preprocess.py`,
  `optionB/train_lora.py` (reused), `optionC_eval.py`.

---

## 1. Why

Minutes was the last feature still on the dedicated `LFM2.5-1.2B-JP` text model. The audio model's backbone **is** an
LFM2.5 LM, so in principle it can summarize. **Tested first:** the *clean* (un-adapted) backbone failed badly —
broke owner attribution (everything → 「私」), repeated/hallucinated tasks, dropped the English summary, garbled rare
kanji, and matched no speed gain. So a small LoRA was needed to teach the **format + attribution** reliably.

---

## 2. Task formulation

Pure **text → text**:

```
system : <the minutes SYSTEM_PROMPT from src/csmeeting/minutes/prompt.py>   (no few-shot — the LoRA learns the format)
user   : <running transcript>
assistant: <minutes markdown: 要約 / 決定事項 / アクションアイテム[担当/期限] / English Summary>
```

`LFM2AudioChatMapper`, assistant content = `TextSegment(minutes)` (supervised). `max_context 2048`.

---

## 3. Data — programmatic, correct-by-construction

No real minutes corpus exists, so transcripts and gold minutes are **generated from the same structured facts**
(`optionC_gen_minutes.py`): participants, topic, decisions, and action items (each with owner / deadline / kind).
Because the gold minutes are rendered deterministically from the facts, **owner attribution is always correct** — the
exact thing the clean backbone got wrong. 520 train / 40 eval.

### The critical insight (v1 → v2): match the app's transcript shape
The app's transcript is **bare utterance lines with no speaker labels** (ASR doesn't diarize). v1 trained with
`Name:` prefixes and therefore failed on real input. **v2 onward uses label-free transcripts** with owners named
*inside* the utterance (`鈴木さんに…お願いします` → owner 鈴木; `私が…やります` → 私) — matching reality.

### Iteration history (each fixed a real failure)
| Ver | Change | Fixes |
|---|---|---|
| v1 | `Name:`-prefixed transcripts | (mismatch — failed on app input) |
| v2 | **label-free** transcripts, owners in-sentence | works on real transcripts |
| v3 | + **zero-decision** meetings | Decisions no longer over-listed (uses 特になし) |
| v4 | + **unassigned** actions → `担当: 未指定` | stops inventing owner names |
| **v5 (deployed)** | + **0/1-action** meetings | stops padding sparse transcripts with phantom 未指定 items |

Transcripts are bilingual with code-switch fillers; deadlines are sometimes omitted (→ 期限 omitted in gold) to
mirror real speech. Preprocess: `optionC_preprocess.py` (text→text).

---

## 4. Training

`train_lora.py`, same LoRA recipe as translate-TTS:

| Param | Value |
|---|---|
| rank `r` / `alpha` / dropout | 16 / 32 / 0.05 |
| target modules (8) | `q,k,v,out_proj`, `in_proj`, `w1,w2,w3` |
| lr / optimizer | 1e-4 cosine / AdamW, warmup 50 |
| batch / ctx | 4 / 2048 |
| steps | 600 (~2.5 min on one H100) |
| val loss | ~0.0002 (templated data → low loss; generalization judged by eval, not loss) |

---

## 5. Evaluation (vs the dedicated `LFM2.5-1.2B-JP`)

Judged on **held-out + out-of-distribution** transcripts for: 4 sections present, English summary actually in English,
owner attribution (私 / named / 未指定), and no phantom items.

**v5 results:**
- **Sparse (2 lines):** exactly **one** action item, `決定事項: 特になし` — no padding.
- **Rich (6 lines, code-switched):** `[担当: 私 / 期限: 来週月曜] 議事録`, `[担当: 田中 / 期限: 明日] データ`,
  `[担当: 鈴木] API修正` — correct owners + deadlines, English summary in English, format perfect.

**Performance improvements vs the clean backbone:** clean backbone → broken (attribution all 私, repetition, JP
"English" summary, garbled kanji). v5 → correct format, correct attribution incl. 未指定 for unstated owners, English
summary in English, no phantom items — at parity with the dedicated text model on realistic input, while **removing
that model entirely**.

> **Decode note:** decode the **full token sequence at once** (`proc.text.decode(torch.tensor(ids))`). Per-token
> decoding splits some rare kanji into `���`. The engine's `generate_minutes` does whole-sequence decode.

---

## 6. Integration (multi-adapter on one base)

`LiveConfig.minutes_lora` → loaded as a **second named adapter** (`"minutes"`) on the same JA audio `PeftModel` that
holds the translate-TTS adapter (`"default"`) and the ASR adapter (`"asr"`). `AudioEngine.generate_minutes()` does
`set_adapter("minutes")`, runs `generate_sequential` (text only), whole-sequence-decodes, then restores the primary
adapter. `Assistant.minutes()` routes here when `has_minutes_lora`. With `--no-text-model` the `LFM2.5-1.2B-JP` is no
longer loaded. Run: `serve_api.py --minutes-lora <…/minutes_lora_v5/final> --no-text-model`.

All three adapters (`asr` / `default` / `minutes`) share **one** ~3 GB base, switched per task — no quality or speed
penalty vs separate instances (see `system-architecture.md`).

---

## 7. Lessons

1. **Test the un-adapted backbone first** — it told us a LoRA was actually needed (unlike translation, where the
   backbone was already close).
2. **Match training input to production** — the label-free transcript fix (v1→v2) was the single biggest gain.
3. **Correct-by-construction labels** beat distilling an imperfect teacher for attribution.
4. **Each visible failure → one targeted data tweak** (over-listing → zero-decision; invented names → 未指定;
   padding → 0/1-action). Eval on out-of-distribution transcripts, not val loss.
5. **Future:** to fully close the gap, paraphrase transcripts with an LLM for more natural variety.
