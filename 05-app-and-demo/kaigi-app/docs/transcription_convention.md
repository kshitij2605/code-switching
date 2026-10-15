# Transcription convention (READ FIRST — align the whole team on this)

The single most important decision in this project. JP↔EN code-switch ASR is hard *because the "correct"
transcription of an English-sounding word is ambiguous*: is コンピューター "computer" or「コンピューター」? Inconsistent
labels poison both training and evaluation. So we fix one convention and apply it to **every** gold transcript and
every eval reference.

## The rule, in one line

> **Script choice encodes the loanword↔switch distinction.** Naturalized into Japanese → **katakana**. A genuine
> English insertion (said in English) → **Latin**.

This is deliberate: it lets us (a) auto-derive language spans from the gold text by Unicode script, and (b) render
each span with a script-matched TTS voice, keeping audio and label aligned.

## Decisions

| Case | Write as | Example |
| --- | --- | --- |
| **Established loanword (外来語)** — naturalized, Japanese phonology | **katakana** | スケジュール, プロジェクト, ミーティング, データ, リリース |
| **Wasei-eigo (和製英語)** — pseudo-English coined in Japan | **katakana** | スキンシップ, アジェンダ, スクショ |
| **True code-switch** — real English word/phrase in English phonology | **Latin** | `deadline`, `let's align`, `root cause`, `next steps` |
| **Acronyms / initialisms** | **Latin (upper)** | `KPI`, `ROI`, `API`, `PR`, `OKR` |
| **Proper nouns / product names** | **Latin** (as branded) | `Slack`, `Figma`, `GitHub` |
| **Numbers** | **Arabic numerals** | `80%`, `Q3`, `3,776` |
| **English stem + する/った** | stem in the script that matches pronunciation | `assignする` (English-said) vs `アサインする` (naturalized) |
| **Punctuation** | Japanese 。、 for JP clauses; keep English `,`/`.` inside English spans | — |

### The judgment call (English stem + verb)

Many English verbs exist both ways in Japanese speech. Decide **by how it is pronounced in the audio**:
- Said with English phonology → Latin (`updateします`, `finalizeしておく`).
- Naturalized / katakana-phonology → katakana (`アップデートします`, `カバーする`).

When generating synthetic data we **control** this: a Latin-written stem is voiced by the English voice, a
katakana-written stem by the Japanese voice. So the label always matches the audio. (For *real* recordings, annotate
by ear and, when truly ambiguous, prefer the katakana/native form and log it — see eval multi-reference note below.)

## Spaces

No spaces around an English insertion embedded in Japanese text (`来週までにdeadlineを設定します`). Keep internal spaces
inside multi-word English spans (`root cause`, `next steps`). `convention.segment_into_spans()` relies on this.

## Why this is the demo's "aha"

In practice, a generic cloud ASR writes コンピューター as "computer" (or mangles a real `deadline` into デッドライン). Our
fine-tune draws the line correctly. The convention above is *what* it learns. Keep it consistent everywhere.

## Eval note (multi-reference)

Because the boundary is genuinely ambiguous for some tokens, the eval harness will also accept a **katakana-normalized
variant** of each reference to bound unfair penalties (toWER/polyWER-style). That is an *eval* relaxation only —
**training labels stay strict** per the table above.
