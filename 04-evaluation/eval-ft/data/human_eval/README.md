# data/human_eval — real spontaneous JA-EN gold eval set

Full protocol: [`../../docs/human-eval-protocol.md`](../../docs/human-eval-protocol.md).

## Layout
```
data/human_eval/
  audio/                     # 16 kHz mono WAVs: utt_0001.wav, ...  (git-ignored)
  manifest.jsonl             # one row per clip (you create this)
  manifest.template.jsonl    # 3 worked examples — copy the shape
```

## Manifest fields
| field | meaning |
|-------|---------|
| `id` | unique utterance id (`utt_0001`) |
| `reference` | gold transcript, correct script per policy (see protocol §C) |
| `hypothesis` | leave `""` — filled by `eval/run_baseline.py` |
| `audio` | path to WAV, relative to `--audio-root` (here `data/human_eval`) |
| `switch_type` | `intra-sentential` / `inter-sentential` / `tag-insertion` |
| `register` | `mock-standup` / `bug-walkthrough` / `code-review` / `casual` / … |
| `matrix_language` | dominant language of the utterance: `ja` or `en` |
| `speaker_l1` | speaker's first language: `ja` / `en` / `other` |
| `speaker_id` | stable per-speaker id (`spk01`) |
| `duration_sec` | clip length in seconds |
| `scenario` | short free-text description |

## Workflow (recording → manifest row)
1. Record spontaneous mixed speech (+ verbal consent). One utterance per clip.
2. `ffmpeg -i raw.m4a -ac 1 -ar 16000 audio/utt_0001.wav`
3. Transcribe per the script policy (loanwords→katakana, true English→Latin).
4. Append a row to `manifest.jsonl` with transcript + metadata.
5. Evaluate: `eval/run_baseline.py --manifest data/human_eval/manifest.jsonl
   --audio-root data/human_eval --model lfm --out artifacts/preds/lfm_human.jsonl`
   then `eval/score.py lfm:artifacts/preds/lfm_human.jsonl`.
