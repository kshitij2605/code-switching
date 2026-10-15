# Phase 0 ladder statistics

Paired bootstrap (5000 resamples, seeds pooled per utterance); p-values Holm-adjusted per test set and metric. A bootstrap p of 0 is shown as its bound (<m·2/n).


## csf_read

| comparison (A - B) | ΔMER [95% CI] p_holm | ΔScriptAcc [95% CI] p_holm | ΔScriptAcc_strict [95% CI] p_holm | ΔKataKeep [95% CI] p_holm |
|---|---|---|---|---|
| replay effect (Whisper cs+mono - cs) | +4.5 [+1.4, +7.9] 0.018 | -13.0 [-17.4, -9.0] <0.004 | -8.5 [-12.3, -5.2] <0.004 | +53.7 [+43.2, +64.0] <0.004 |
| fine-tuning (Whisper cs - zero-shot) | -24.5 [-29.6, -19.7] <0.004 | +33.1 [+26.8, +40.0] <0.004 | +27.8 [+21.9, +34.4] <0.004 | -56.3 [-66.2, -45.8] <0.004 |
| fine-tuning (Whisper cs+mono - zero-shot) | -20.0 [-24.3, -15.8] <0.004 | +20.2 [+14.5, +26.4] <0.004 | +19.3 [+13.9, +25.2] <0.004 | -2.5 [-9.7, +4.3] 1 |
| LFM encoder+LM - LM only | -0.3 [-4.2, +3.1] 0.914 | -2.1 [-4.5, +0.4] 0.178 | +1.5 [-0.7, +3.6] 0.383 | +0.0 [-3.0, +2.9] 1 |
| LFM encoder only - LM only | +20.8 [+10.9, +32.6] <0.004 | -11.4 [-17.8, -5.1] 0.0012 | -10.1 [-14.6, -5.6] <0.004 | -3.6 [-13.5, +6.7] 1 |
| fine-tuning (LFM both - zero-shot p1) | -11.1 [-21.6, -3.6] 0.0016 | +24.9 [+19.7, +29.9] <0.004 | +19.1 [+15.5, +22.6] <0.004 | -29.1 [-39.0, -18.8] <0.004 |
| architecture, same data (Whisper cs+mono - LFM both) | -28.6 [-33.5, -23.7] <0.004 | +22.2 [+16.7, +27.6] <0.004 | +38.3 [+32.6, +43.5] <0.004 | +16.3 [+7.2, +25.6] <0.004 |
| ja+rp decoding (Whisper ja+rp - ja) | -9.2 [-27.4, +0.6] 0.241 | +0.4 [-2.9, +3.1] 0.757 | +0.6 [-2.7, +3.3] 0.684 | -1.3 [-4.4, +1.4] 1 |

## mono_ja

| comparison (A - B) | ΔJA_CER [95% CI] p_holm |
|---|---|
| forgetting (Whisper cs - zero-shot) | +11.2 [+8.8, +13.6] <0.002 |
| forgetting (Whisper cs+mono - zero-shot) | -0.1 [-0.4, +0.3] 1 |
| forgetting (LFM both - zero-shot p1) | -0.2 [-0.7, +0.4] 1 |
| forgetting (LFM LM only - zero-shot p1) | +0.3 [-0.3, +1.0] 0.995 |
| forgetting (LFM encoder only - zero-shot p1) | +3.1 [+1.7, +4.7] <0.002 |

## mono_en

| comparison (A - B) | ΔEN_WER [95% CI] p_holm |
|---|---|
| forgetting (Whisper cs - zero-shot) | +0.2 [-0.1, +0.6] 0.402 |
| forgetting (Whisper cs+mono - zero-shot) | -0.2 [-0.7, +0.2] 0.495 |
| forgetting (LFM both - zero-shot p1) | -27.6 [-36.8, -20.4] <0.002 |
| forgetting (LFM LM only - zero-shot p1) | -14.5 [-24.2, -5.7] 0.0096 |
| forgetting (LFM encoder only - zero-shot p1) | +10.8 [-2.7, +27.2] 0.374 |

## Ranking fidelity (pre-registered Q4)

### synth_art vs csf_read, 11 systems

- Kendall τ-b (MER): +0.59 [95% CI +0.48, +0.66]
- Kendall τ-b (ScriptAcc): +0.83 [95% CI +0.65, +0.87]
- Kendall τ-b (MER_capped) (exploratory): +0.44 [95% CI +0.40, +0.55]
- Selection regret, all systems (exploratory): proxy picks ev_ft_whisper_csmono_auto; real best ev_ft_whisper_cs_auto; regret 4.5 MER
- Selection regret, fine-tuned only (exploratory): proxy picks ev_ft_whisper_csmono_auto; real best ev_ft_whisper_cs_auto; regret 4.5 MER

# Exploratory analyses (not pre-registered)
Own Holm family, separate from the pre-registered comparisons.

## csf_read

| comparison (A - B) | ΔMER [95% CI] p_holm | ΔScriptAcc [95% CI] p_holm | ΔScriptAcc_strict [95% CI] p_holm | ΔKataKeep [95% CI] p_holm |
|---|---|---|---|---|
| ja+rp decoding on Whisper cs (ja+rp - auto) | +0.6 [+0.2, +0.9] 0.0048 | -0.1 [-0.6, +0.3] 1 | -0.5 [-1.0, -0.1] 0.0288 | +0.0 [-1.6, +1.5] 1 |
| ja+rp decoding on Whisper cs+mono (ja+rp - auto) | +0.9 [-0.2, +2.0] 0.0988 | -0.5 [-2.1, +1.1] 1 | -1.0 [-2.5, +0.4] 0.169 | -2.5 [-5.8, +0.4] 0.232 |
| replay effect under ja+rp decoding (cs+mono - cs, both ja+rp) | +4.9 [+1.8, +8.1] 0.0048 | -13.3 [-17.6, -9.6] <0.003 | -9.0 [-12.7, -5.9] <0.003 | +51.2 [+41.4, +60.6] <0.003 |
| training length (Whisper cs, epoch 3 - epoch 1) | +0.9 [+0.5, +1.3] <0.003 | +0.2 [-0.3, +0.7] 1 | -0.7 [-1.2, -0.2] 0.0288 | -3.4 [-6.6, -1.0] 0.0128 |
| training length (Whisper cs+mono, epoch 3 - epoch 1) | +3.0 [+1.3, +4.7] <0.003 | -3.5 [-5.6, -1.5] <0.003 | -3.4 [-5.3, -1.6] <0.003 | +2.3 [+0.0, +5.2] 0.2 |
| replay effect at epoch 3 (cs+mono - cs) | +6.9 [+3.6, +10.6] <0.003 | -16.7 [-21.6, -12.5] <0.003 | -11.3 [-15.3, -7.8] <0.003 | +57.5 [+47.0, +67.8] <0.003 |

## mono_ja

| comparison (A - B) | ΔJA_CER [95% CI] p_holm |
|---|---|
| training length (Whisper cs, epoch 3 - epoch 1) | +0.7 [+0.2, +1.4] 0.0016 |
| training length (Whisper cs+mono, epoch 3 - epoch 1) | -0.0 [-0.3, +0.2] 0.848 |

## mono_en

| comparison (A - B) | ΔEN_WER [95% CI] p_holm |
|---|---|
| training length (Whisper cs, epoch 3 - epoch 1) | +0.0 [-0.1, +0.1] 1 |
| training length (Whisper cs+mono, epoch 3 - epoch 1) | -0.0 [-0.3, +0.2] 1 |

## Training reproducibility

The same trainings run again with the same seeds; a CI that covers 0 means the ladder's numbers reproduce.

## csf_read

| comparison (A - B) | ΔMER [95% CI] p_holm | ΔScriptAcc [95% CI] p_holm | ΔScriptAcc_strict [95% CI] p_holm | ΔKataKeep [95% CI] p_holm |
|---|---|---|---|---|
| repeat - ladder (Whisper cs, dev-selected epoch 1) | -0.1 [-0.3, +0.1] 0.498 | -0.1 [-0.3, +0.1] 0.433 | +0.0 [-0.2, +0.2] 1 | +1.1 [-0.2, +2.6] 0.357 |
| repeat - ladder (Whisper csmono, dev-selected epoch 1) | +0.2 [-0.1, +0.5] 0.498 | -0.1 [-0.3, +0.0] 0.298 | -0.1 [-0.3, +0.1] 1 | -0.8 [-2.2, +0.0] 0.357 |

## mono_ja

| comparison (A - B) | ΔJA_CER [95% CI] p_holm |
|---|---|
| repeat - ladder (Whisper cs, dev-selected epoch 1) | +0.0 [-0.1, +0.1] 1 |
| repeat - ladder (Whisper csmono, dev-selected epoch 1) | +0.0 [-0.0, +0.1] 1 |

## mono_en

| comparison (A - B) | ΔEN_WER [95% CI] p_holm |
|---|---|
| repeat - ladder (Whisper cs, dev-selected epoch 1) | +0.0 [+0.0, +0.0] 0.527 |
| repeat - ladder (Whisper csmono, dev-selected epoch 1) | +0.0 [-0.0, +0.1] 0.83 |

## Ranking fidelity, further set pairs

synth_diss (n=30) is a pre-registered ranking set, but only the synth_art τ was named in the protocol; the twins (same text as csf_read, synthetic audio) and csyodas are exploratory sets.

### synth_diss vs csf_read, 11 systems

- Kendall τ-b (MER): +0.66 [95% CI +0.52, +0.81]
- Kendall τ-b (ScriptAcc): +0.46 [95% CI +0.26, +0.62]
- Kendall τ-b (MER_capped) (exploratory): +0.64 [95% CI +0.52, +0.82]
- Selection regret, all systems (exploratory): proxy picks ev_ft_whisper_cs_auto; real best ev_ft_whisper_cs_auto; regret 0.0 MER
- Selection regret, fine-tuned only (exploratory): proxy picks ev_ft_whisper_cs_auto; real best ev_ft_whisper_cs_auto; regret 0.0 MER

### twins_kokoro vs csf_read, 11 systems

- Kendall τ-b (MER): +0.85 [95% CI +0.67, +0.93]
- Kendall τ-b (ScriptAcc): +0.70 [95% CI +0.49, +0.82]
- Kendall τ-b (MER_capped) (exploratory): +0.89 [95% CI +0.75, +0.93]
- Selection regret, all systems (exploratory): proxy picks ev_ft_whisper_csmono_auto; real best ev_ft_whisper_cs_auto; regret 4.5 MER
- Selection regret, fine-tuned only (exploratory): proxy picks ev_ft_whisper_csmono_auto; real best ev_ft_whisper_cs_auto; regret 4.5 MER

### twins_edge vs csf_read, 11 systems

- Kendall τ-b (MER): +0.60 [95% CI +0.45, +0.71]
- Kendall τ-b (ScriptAcc): +0.67 [95% CI +0.53, +0.85]
- Kendall τ-b (MER_capped) (exploratory): +0.53 [95% CI +0.42, +0.64]
- Selection regret, all systems (exploratory): proxy picks ev_ft_whisper_cs_auto; real best ev_ft_whisper_cs_auto; regret 0.0 MER
- Selection regret, fine-tuned only (exploratory): proxy picks ev_ft_whisper_cs_auto; real best ev_ft_whisper_cs_auto; regret 0.0 MER

### csyodas vs csf_read, 11 systems

- Kendall τ-b (MER): +0.16 [95% CI -0.05, +0.67]
- Kendall τ-b (ScriptAcc): +0.82 [95% CI +0.67, +0.93]
- Kendall τ-b (MER_capped) (exploratory): +0.75 [95% CI +0.56, +0.82]
- Selection regret, all systems (exploratory): proxy picks ev_ft_lfm_encoder; real best ev_ft_whisper_cs_auto; regret 54.3 MER
- Selection regret, fine-tuned only (exploratory): proxy picks ev_ft_lfm_encoder; real best ev_ft_whisper_cs_auto; regret 54.3 MER

## Runaway-robust MER (each hypothesis's errors capped at its reference length; seeds pooled)

Runaway = share of hypotheses with errors above 2x their reference tokens.

### csf_read

| system | MER | MER_capped [95% CI] | Runaway |
|---|---|---|---|
| ev_ft_whisper_cs_ja | 16.8 | 16.7 [14.4, 19.2] | 0.0% |
| ev_ft_whisper_cs_ep1_auto | 16.8 | 16.7 [14.5, 19.2] | 0.0% |
| ev_ft_whisper_cs_auto | 16.9 | 16.8 [14.6, 19.3] | 0.0% |
| ev_ft_whisper_cs_ja_rp | 17.5 | 17.4 [15.1, 20.0] | 0.0% |
| ev_ft_whisper_cs_ep2_auto | 17.5 | 17.4 [15.0, 19.7] | 0.0% |
| ev_ft_whisper_cs_ep3_auto | 17.7 | 17.5 [15.3, 20.1] | 0.0% |
| ev_ft_whisper_csmono_ja | 21.5 | 20.4 [17.4, 23.5] | 0.0% |
| ev_ft_whisper_csmono_auto | 21.4 | 20.4 [17.5, 23.8] | 0.0% |
| ev_ft_whisper_csmono_ep1_auto | 21.6 | 20.6 [17.4, 23.8] | 0.0% |
| ev_ft_whisper_csmono_ja_rp | 22.3 | 21.3 [18.5, 24.4] | 0.0% |
| ev_ft_whisper_csmono_ep3_auto | 24.6 | 23.5 [20.2, 26.7] | 0.0% |
| ev_ft_whisper_csmono_ep2_auto | 24.9 | 23.8 [20.5, 26.9] | 0.0% |
| zs_whisper_auto | 41.4 | 39.0 [34.9, 43.1] | 0.5% |
| zs_whisper_ja | 54.4 | 41.9 [37.7, 46.1] | 2.0% |
| zs_whisper_ja_rp | 45.2 | 42.1 [37.9, 46.1] | 0.0% |
| ev_ft_lfm_lm | 50.4 | 47.8 [43.9, 51.8] | 0.7% |
| ev_ft_lfm_both | 50.1 | 49.3 [45.6, 53.2] | 0.2% |
| zs_lfm_p1 | 61.2 | 53.6 [49.5, 57.8] | 2.0% |
| zs_lfm_p3 | 67.7 | 56.6 [52.2, 61.0] | 4.6% |
| ev_ft_lfm_encoder | 71.2 | 58.0 [53.6, 62.3] | 3.2% |
| zs_lfm_p2 | 109.4 | 58.9 [54.5, 63.3] | 12.2% |

### csyodas

| system | MER | MER_capped [95% CI] | Runaway |
|---|---|---|---|
| ev_ft_whisper_cs_auto | 55.1 | 37.2 [33.6, 40.7] | 3.2% |
| ev_ft_whisper_csmono_auto | 54.2 | 37.4 [33.9, 41.0] | 4.5% |
| zs_whisper_ja | 68.4 | 42.4 [38.5, 46.7] | 5.4% |
| zs_whisper_auto | 62.6 | 43.0 [39.0, 47.3] | 4.1% |
| zs_whisper_ja_rp | 52.5 | 43.1 [39.2, 47.1] | 4.5% |
| ev_ft_lfm_encoder | 48.8 | 44.3 [40.3, 48.3] | 2.8% |
| zs_lfm_p1 | 55.4 | 45.7 [41.8, 49.5] | 4.1% |
| ev_ft_lfm_lm | 51.8 | 46.5 [42.9, 50.4] | 2.5% |
| ev_ft_lfm_both | 52.6 | 48.1 [44.5, 51.7] | 1.9% |
| zs_lfm_p3 | 227.8 | 57.6 [52.9, 62.7] | 27.3% |
| zs_lfm_p2 | 584.2 | 71.2 [65.8, 76.5] | 50.0% |

### twins_kokoro

| system | MER | MER_capped [95% CI] | Runaway |
|---|---|---|---|
| ev_ft_whisper_csmono_auto | 8.7 | 8.6 [7.0, 10.3] | 0.0% |
| ev_ft_whisper_cs_auto | 13.7 | 13.7 [11.6, 16.2] | 0.0% |
| zs_whisper_auto | 31.6 | 29.5 [26.0, 33.2] | 0.5% |
| zs_whisper_ja | 41.1 | 30.1 [26.8, 33.6] | 2.0% |
| zs_whisper_ja_rp | 32.0 | 30.7 [27.4, 34.2] | 0.5% |
| ev_ft_lfm_both | 45.3 | 44.1 [40.5, 47.7] | 0.5% |
| ev_ft_lfm_lm | 46.6 | 44.7 [40.7, 48.9] | 0.7% |
| zs_lfm_p1 | 56.0 | 48.5 [44.0, 52.7] | 2.0% |
| ev_ft_lfm_encoder | 72.7 | 56.4 [52.0, 61.1] | 3.9% |
| zs_lfm_p3 | 112.9 | 69.1 [64.5, 73.5] | 12.8% |
| zs_lfm_p2 | 340.8 | 94.7 [92.9, 96.4] | 60.7% |

### twins_edge

| system | MER | MER_capped [95% CI] | Runaway |
|---|---|---|---|
| ev_ft_whisper_cs_auto | 16.0 | 16.0 [13.7, 18.6] | 0.0% |
| ev_ft_whisper_csmono_auto | 28.4 | 26.8 [23.2, 30.5] | 0.0% |
| ev_ft_lfm_lm | 54.3 | 49.9 [46.1, 53.8] | 0.7% |
| ev_ft_lfm_encoder | 57.9 | 52.3 [48.2, 56.3] | 0.5% |
| ev_ft_lfm_both | 55.1 | 52.4 [48.4, 56.1] | 0.5% |
| zs_whisper_auto | 71.1 | 61.9 [57.9, 66.1] | 2.6% |
| zs_whisper_ja_rp | 73.7 | 64.2 [60.3, 68.2] | 4.1% |
| zs_whisper_ja | 86.3 | 64.6 [60.4, 69.2] | 5.6% |
| zs_lfm_p1 | 76.2 | 65.6 [61.6, 69.5] | 1.5% |
| zs_lfm_p3 | 113.0 | 72.7 [68.4, 76.8] | 11.7% |
| zs_lfm_p2 | 370.9 | 91.5 [88.7, 94.1] | 61.2% |

## Script-choice gap (MER - MER_lenient, seeds pooled)


### csf_read

| system | MER | MER_lenient | gap [95% CI] |
|---|---|---|---|
| ev_ft_lfm_both | 50.1 | 44.4 | 5.7 [4.2, 7.5] |
| ev_ft_lfm_encoder | 71.2 | 68.4 | 2.8 [1.9, 3.7] |
| ev_ft_lfm_lm | 50.4 | 45.6 | 4.7 [3.4, 6.4] |
| ev_ft_whisper_cs_auto | 16.9 | 12.9 | 4.0 [2.6, 5.8] |
| ev_ft_whisper_cs_ep1_auto | 16.8 | 12.9 | 4.0 [2.5, 5.9] |
| ev_ft_whisper_cs_ep2_auto | 17.5 | 13.5 | 4.0 [2.5, 5.8] |
| ev_ft_whisper_cs_ep3_auto | 17.7 | 13.7 | 4.0 [2.5, 5.8] |
| ev_ft_whisper_cs_ja | 16.8 | 12.8 | 4.0 [2.6, 5.8] |
| ev_ft_whisper_cs_ja_rp | 17.5 | 13.4 | 4.0 [2.6, 5.9] |
| ev_ft_whisper_csmono_auto | 21.4 | 16.5 | 4.9 [3.3, 6.7] |
| ev_ft_whisper_csmono_ep1_auto | 21.6 | 16.7 | 4.9 [3.4, 6.8] |
| ev_ft_whisper_csmono_ep2_auto | 24.9 | 18.9 | 6.0 [4.5, 8.0] |
| ev_ft_whisper_csmono_ep3_auto | 24.6 | 18.8 | 5.8 [4.1, 7.7] |
| ev_ft_whisper_csmono_ja | 21.5 | 16.6 | 4.9 [3.3, 6.9] |
| ev_ft_whisper_csmono_ja_rp | 22.3 | 17.7 | 4.6 [3.2, 6.6] |
| zs_lfm_p1 | 61.2 | 51.7 | 9.5 [7.7, 11.4] |
| zs_lfm_p2 | 109.4 | 99.6 | 9.8 [7.5, 12.7] |
| zs_lfm_p3 | 67.7 | 59.5 | 8.2 [6.5, 9.9] |
| zs_whisper_auto | 41.4 | 30.6 | 10.8 [8.5, 13.2] |
| zs_whisper_ja | 54.4 | 43.9 | 10.6 [8.7, 12.8] |
| zs_whisper_ja_rp | 45.2 | 36.0 | 9.2 [7.6, 11.0] |

### csyodas

| system | MER | MER_lenient | gap [95% CI] |
|---|---|---|---|
| ev_ft_lfm_both | 52.6 | 48.6 | 4.0 [2.8, 5.3] |
| ev_ft_lfm_encoder | 48.8 | 43.8 | 5.0 [3.6, 6.5] |
| ev_ft_lfm_lm | 51.8 | 47.5 | 4.3 [3.1, 5.6] |
| ev_ft_whisper_cs_auto | 55.1 | 48.5 | 6.7 [5.1, 8.3] |
| ev_ft_whisper_csmono_auto | 54.2 | 48.0 | 6.2 [4.7, 7.8] |
| zs_lfm_p1 | 55.4 | 47.3 | 8.2 [6.4, 10.2] |
| zs_lfm_p2 | 584.2 | 575.2 | 9.0 [6.4, 12.1] |
| zs_lfm_p3 | 227.8 | 217.7 | 10.2 [7.0, 14.4] |
| zs_whisper_auto | 62.6 | 54.4 | 8.3 [6.5, 10.3] |
| zs_whisper_ja | 68.4 | 60.1 | 8.4 [6.6, 10.4] |
| zs_whisper_ja_rp | 52.5 | 44.2 | 8.3 [6.6, 10.2] |

### twins_kokoro

| system | MER | MER_lenient | gap [95% CI] |
|---|---|---|---|
| ev_ft_lfm_both | 45.3 | 41.2 | 4.1 [2.9, 5.6] |
| ev_ft_lfm_encoder | 72.7 | 68.9 | 3.8 [2.4, 5.7] |
| ev_ft_lfm_lm | 46.6 | 44.1 | 2.5 [1.6, 3.7] |
| ev_ft_whisper_cs_auto | 13.7 | 9.7 | 4.0 [2.5, 5.9] |
| ev_ft_whisper_csmono_auto | 8.7 | 6.9 | 1.8 [0.8, 3.1] |
| zs_lfm_p1 | 56.0 | 48.3 | 7.7 [5.6, 9.9] |
| zs_lfm_p2 | 340.8 | 335.6 | 5.2 [2.8, 8.0] |
| zs_lfm_p3 | 112.9 | 108.1 | 4.8 [3.5, 6.3] |
| zs_whisper_auto | 31.6 | 28.0 | 3.6 [2.0, 5.7] |
| zs_whisper_ja | 41.1 | 37.6 | 3.5 [2.2, 5.0] |
| zs_whisper_ja_rp | 32.0 | 28.8 | 3.2 [2.1, 4.5] |

### twins_edge

| system | MER | MER_lenient | gap [95% CI] |
|---|---|---|---|
| ev_ft_lfm_both | 55.1 | 39.8 | 15.3 [13.0, 17.7] |
| ev_ft_lfm_encoder | 57.9 | 42.9 | 15.0 [12.7, 17.6] |
| ev_ft_lfm_lm | 54.3 | 40.0 | 14.4 [12.3, 16.8] |
| ev_ft_whisper_cs_auto | 16.0 | 11.8 | 4.3 [2.8, 6.2] |
| ev_ft_whisper_csmono_auto | 28.4 | 19.0 | 9.4 [7.2, 12.0] |
| zs_lfm_p1 | 76.2 | 51.2 | 25.0 [21.6, 28.4] |
| zs_lfm_p2 | 370.9 | 337.2 | 33.7 [26.7, 41.8] |
| zs_lfm_p3 | 113.0 | 90.6 | 22.4 [18.6, 26.1] |
| zs_whisper_auto | 71.1 | 46.1 | 25.0 [21.9, 28.5] |
| zs_whisper_ja | 86.3 | 60.8 | 25.5 [22.3, 29.1] |
| zs_whisper_ja_rp | 73.7 | 50.1 | 23.6 [20.8, 26.8] |
