# Human Gold Eval Set — Recording & Transcription Protocol

This is the **real, spontaneous** EN-JP code-switching eval set — the project's
differentiator (CS-FLEURS is synthetic/induced; this is natural speech). Target
**30–40 utterances** (stretch 60), each **5–20 s**. Small-scale: one careful
transcriber is enough — **no blind double-annotation / IAA needed**.

Output must match the manifest schema the eval harness already consumes:
`{id, reference, hypothesis:"", audio, switch_type, register, matrix_language,
speaker_l1, speaker_id, duration_sec, scenario}` (template in
`data/human_eval/manifest.template.jsonl`).

---

## A. Recording flow (≈6 steps)

1. **Setup.** Quiet room, phone or laptop mic ~20 cm away. Record one utterance
   (or a short scenario) per file. Aim for natural, spontaneous mixing — *don't*
   read a script.
2. **Consent (record this verbally at the start of each session).**
   > "This is <name>, <date>. I consent to this audio and its transcript being
   > used and shared as part of a Japanese–English code-switching ASR research
   > project." Without this, the clip is unusable — keep it.
3. **Speak** using one of the scenarios in §B. Mix EN and JA the way you
   naturally would. Pause between utterances so they're easy to split.
4. **Save & name** files `utt_0001.wav`, `utt_0002.wav`, … in a working folder.
5. **Convert to 16 kHz mono WAV** (what the ASR models expect):
   ```bash
   ffmpeg -i input.m4a -ac 1 -ar 16000 data/human_eval/audio/utt_0001.wav
   # batch: for f in raw/*.m4a; do ffmpeg -i "$f" -ac 1 -ar 16000 \
   #   "data/human_eval/audio/$(basename "${f%.*}").wav"; done
   ```
6. **Transcribe** each clip per §C and add one manifest row per clip
   (`data/human_eval/manifest.jsonl`).

## B. Elicitation scenarios (pick a few; designed to induce natural switching)

1. **Mock standup** — "Describe yesterday's sprint / what you're working on
   today / any blockers," in your normal mixed speech.
2. **Bug walkthrough** — Explain a bug you recently fixed (root cause, the fix,
   how you tested it). Naturally pulls in English tech terms.
3. **Code review** — Talk through feedback you'd leave on a teammate's PR
   ("this function …", "let's rename …", "add a test for …").
4. **Architecture / design chat** — Explain how a system you know works
   (services, DB, deploy).
5. **Casual tech weekend** — Chat about a gadget, game, or side-project — lighter
   register, fewer switches.
6. **Tool/CLI narration** — Narrate doing a task (git, Docker, cloud console).

Stratify across: **switch_type** (intra-sentential / inter-sentential /
tag-insertion), **register** (work vs casual), **matrix_language** (ja vs en
dominant), **speaker_l1** (ja / en / other). A couple of speakers is plenty.

## C. Transcription conventions

**Write exactly what was said, choosing script by this policy:**

- **Established katakana loanwords → katakana.** デプロイ, バグ, レビュー,
  ミーティング, サーバー, リリース. (Said as assimilated Japanese.)
- **True English insertions → Latin script.** `pull request`, `merge`,
  `deadline`, `rollback`, `staging`, proper nouns (`GitHub`, `Slack`).
- Rule of thumb: would a monolingual JA speaker recognize it as a normal
  Japanese word? → katakana. Is it a deliberate English word/phrase the speaker
  code-switched into? → Latin.
  - e.g. "そこは **デプロイ** 済みだけど **rollback** した方がいい" — loanword
    katakana, true insertion Latin.

**Normalization (keep consistent; the scorer also normalizes, but be tidy):**
- Numbers: write as said; digits for figures (`6.5`, `1967`).
- Use full Japanese punctuation (、。) and normal English punctuation; casing
  natural for English.
- Put a space around Latin↔Japanese boundaries only where natural; the scorer is
  robust to spacing.
- **Fillers/disfluencies**: transcribe Japanese fillers as heard (えーと, あの);
  the scorer auto-drops English fillers (um/uh). Don't stress over these.
- Don't "clean up" grammar — transcribe the real utterance.

## D. Checklist (do this end-to-end)

- [ ] Recorded consent at session start.
- [ ] Spontaneous mixed speech captured, one utterance per clip, 5–20 s each.
- [ ] Converted every clip to **16 kHz mono WAV** in `data/human_eval/audio/`.
- [ ] Transcribed each clip per §C (correct script choice).
- [ ] Added one row per clip to `data/human_eval/manifest.jsonl` (see template).
- [ ] Filled metadata (switch_type, register, matrix_language, speaker_l1, scenario).
- [ ] Spot-checked ~5 transcripts with a second person (optional but nice).
- [ ] ≥30 utterances total, stratified across switch types & registers.

## E. How it feeds the harness

`manifest.jsonl` already has `{id, reference, audio, …}`. Run any model over it
with `eval/run_baseline.py --manifest data/human_eval/manifest.jsonl
--audio-root data/human_eval`, then `eval/score.py` — same flow as CS-FLEURS.
**Evaluate the final fine-tuned model on THIS set.**
