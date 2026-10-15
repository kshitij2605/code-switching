# Japanese↔English Code-Switching Speech

**A research project on Japanese↔English code-switching (CS) speech:** recognition, synthesis
and evaluation, aimed at on-device use in privacy-sensitive (APPI-bound) bilingual meetings.
Generic ASR mangles bilingual speech: it forces English into katakana and drops the minority
language. The project studies how to fix that with small, fully offline models, and how to
measure it reliably.

Models studied include `LiquidAI/LFM2.5-Audio-1.5B-JP` (research only, under its licence),
Whisper-large-v3 / -turbo, and Apache-licensed speech LLMs. Result files are in
[`04-evaluation/eval-ft/results/`](04-evaluation/eval-ft/results/README.md).

The project is organized into three workstreams — **ASR fine-tuning**, **TTS
fine-tuning**, and **evaluation** — grouped by pipeline stage. Within stages
01–04 each appears as an `asr-ft/`, `tts-ft/`, `eval-ft/` subfolder.

## The three workstreams

| Workstream | Focus | Lives mainly in |
|------------|-------|-----------------|
| **`asr-ft/`** | Synthetic data generation + **ASR LoRA fine-tuning** pipeline (data → preprocess → train → eval → GGUF export) | stages 01–04, 06 |
| **`eval-ft/`** | The **shared evaluation harness** + frozen CS-FLEURS / FLEURS benchmark (the held-out scoreboard every model is judged on) | stage 04 |
| **`tts-ft/`** | The **Kaigi (会議) meeting assistant** — one audio base + 3 LoRA adapters (ASR / translate-and-speak / minutes), React web UI, CPU/GGUF path | stages 01–06 |

## Layout (by pipeline stage)

```
01-data-generation/   synthetic CS/EN/JA speech + transcript generation, dataset downloads
02-preprocessing/     JSONL->Arrow preprocessing, dataset balancing, manifest tooling, augmentation
03-training/          LoRA training — ASR (asr-ft) + translate-TTS / minutes / ASR (tts-ft)
04-evaluation/        eval-ft shared benchmark (primary) + per-workstream eval scripts
05-app-and-demo/      kaigi-app (full Kaigi assistant + web UI) + asr-ft Gradio servers
06-deployment/        on-device export — GGUF (llama.cpp) + ONNX notes
docs/                 cross-cutting design docs: ASR training log, eval context, system architecture
```

Within stages 01–04 the work is split into `asr-ft/`, `eval-ft/`, `tts-ft/` subfolders.

## Results

Metric files, statistics and run records: `04-evaluation/eval-ft/results/phase0/`.

The benchmark is CS-FLEURS jpn-eng read/test (n = 196, real read speech, one speaker), with FLEURS `ja_jp` / `en_us` test (200 each) as the forgetting controls. Fine-tuned rows are mean ± sd over 3 seeds, selected on dev data only, and decoded greedily.

Metrics:
- **MER:** mixed error rate over Japanese characters and English words.
- **MER-len:** MER that accepts a transliteration in either script (katakana ↔ English), never a translation.
- **ScriptAcc:** share of English reference words written in Latin script.
- **SA-strict:** share written in Latin script *and* correct.
- **KataKeep:** share of the reference's katakana loanwords kept in katakana. ScriptAcc alone can be gamed by romanising everything; KataKeep catches that.

| Model | CS MER ↓ | CS MER-len ↓ | ScriptAcc ↑ | SA-strict ↑ | KataKeep ↑ | JA-CER ↓ | EN-WER ↓ |
|-------|------|------|------|------|------|------|------|
| LFM2.5-Audio-1.5B-JP, zero-shot | 61.2 | 51.7 | 36.2 | 21.6 | 80.9 | 6.2 | 68.6 |
| Whisper large-v3, zero-shot | 41.4 | 30.6 | 63.2 | 59.7 | 70.7 | 4.0 | **3.9** |
| LFM + LoRA, LM only | 50.4 ± 2.0 | 45.6 | 63.2 ± 3.6 | 39.2 ± 2.7 | 51.8 ± 1.3 | 6.6 ± 0.2 | 54.1 ± 4.5 |
| LFM + LoRA, LM + Conformer encoder | 50.1 ± 2.6 | 44.4 | 61.1 ± 2.7 | 40.7 ± 2.6 | 51.8 ± 1.8 | 6.1 ± 0.2 | 41.0 ± 9.6 |
| Whisper large-v3 + LoRA, CS data only | **16.9 ± 0.0** | **12.9** | **96.3 ± 0.2** | **87.6 ± 0.4** | 14.4 ± 1.9 | 15.1 ± 0.2 | 4.1 ± 0.0 |
| Whisper large-v3 + LoRA, CS + mono replay | 21.4 ± 1.9 | 16.5 | 83.4 ± 2.2 | 79.0 ± 2.0 | 68.1 ± 1.9 | **3.9 ± 0.1** | 3.8 ± 0.2 |

- **LFM fine-tuning helps but does not reach Whisper.** LoRA lowers LFM's code-switching MER from 61.2 to 50.1. That is still behind zero-shot Whisper large-v3 (41.4), and 28.6 points behind Whisper fine-tuned on the same data.
- **Adding the Conformer-encoder LoRA to the LM LoRA changes nothing measurable on CS:** −0.3 MER [−4.2, +3.1].
- **English is LFM's gap.** EN-WER is 41–54 against Whisper's 3.9.
- **Whisper + LoRA on CS data alone is the most accurate on CS, but forgets Japanese** (JA-CER 4.0 → 15.1) and romanises katakana loanwords (KataKeep 14.4).
- **Mono replay removes the forgetting and keeps katakana, at +4.5 MER.**

## Model

- Base: `LiquidAI/LFM2.5-Audio-1.5B-JP` — Liquid Foundation Model (hybrid SSM + attention)
  + 17-layer Conformer audio encoder, 1.5B params.
- Adaptation: **LoRA** (added manually via PEFT — not built into `liquid-audio`): LM LoRA r=8
  + Conformer encoder LoRA r=32, 3 seeds, trained by `03-training/asr-ft/phase0_ladder.py`.
  System prompt `Perform ASR.`
- Training: local H100 GPUs.

## Data and models

Datasets, model weights, checkpoints, virtual environments and logs are not in the
repository: they are large, several carry non-commercial licences, and all are
reproducible from the scripts in `01-data-generation/` and `02-preprocessing/`.
Experiment paths default to `/data/code-switching` (data, runs, environments under
`envs/`); `docs/environment.md` describes the setup.
