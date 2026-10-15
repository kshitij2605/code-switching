# 03 · Training

LoRA fine-tuning of `LFM2.5-Audio-1.5B-JP`. LoRA is added manually via PEFT (the
`liquid-audio` package ships only a full-fine-tune `Trainer`).

## asr-ft/ — code-switch ASR adapter
- `train_lora.py` — main LoRA trainer. LM targets `q/k/v/out_proj, in_proj, w1/w2/w3`;
  Conformer encoder targets `linear_q/k/v/out, linear1/linear2`. Two param groups:
  LM LR 3e-5, encoder LR 6e-6 (5x lower). ~8.16M trainable params (0.56%).
- `train_lfm_lora.py`, `train_asr_wandb.py` — earlier/variant trainers with W&B logging.
- `run_all.sh` — end-to-end data run (environment: `docs/environment.md`).
- `phase0_ladder.py`, `phase1_experiments.py` — resumable multi-GPU runners for the baseline ladder and experiments.

## tts-ft/ — translate-TTS, minutes, ASR adapters
- `train_cs_asr.py` — code-switch ASR adapter trainer.
- `optionB_train_lora.py` — translate-and-speak adapter (`transtts_lora_v3`): emits translated
  text then speech in one pass.
- `run_lora_sweep.sh`, `run_lora_lowdens.sh`, `run_aug_pipeline.sh`, `run_post_aug.sh`,
  `run_cosy_pipeline.sh`, `run_cosy_downstream.sh` — sweeps and data-variant pipelines.
- `kit_integration/` — drop-in for the Liquid4All LFM audio training kit.
