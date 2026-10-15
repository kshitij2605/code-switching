# Research environment

Every number we publish (evaluation, scoring, LoRA training) comes from one pinned
Python environment. Three files at the repo root define it:

| File | Role |
|---|---|
| `pyproject.toml` | Direct dependencies, the key pins, and the `pytorch-cu126` index |
| `uv.lock` | Full resolution: 135 packages (including the optional `extras` group), exact versions with sha256 hashes. Linux x86_64 only. |
| `.python-version` | CPython 3.12 |

The environment itself lives outside the repo at `/data/code-switching/envs/liquid-ai-research`. It is
about 7 GB on disk, but its files are hard links into the uv cache, which sits on the same
filesystem.

## Recreate or refresh the environment

Host: `mactrn01` (RHEL 9.3, 4x H100 NVL, NVIDIA driver 565.57.01). Requires uv 0.12.5 or newer (`required-version` in `pyproject.toml`).

```bash
# 1. Corporate proxy. It is normally already set in the login environment
#    (check with `env | grep -i proxy`).
export http_proxy=http://<proxy-host>:<port>
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy

# 2. From the repo root, build the environment exactly as locked.
cd <repo>
export UV_PROJECT_ENVIRONMENT=/data/code-switching/envs/liquid-ai-research
uv sync --locked

# 3. Keep model weights under /data. The HF token stays in its default location.
export HF_HOME=/data/code-switching/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
```

- `uv sync --locked` stops with an error if `pyproject.toml` and `uv.lock` disagree, instead of quietly re-resolving. It also removes anything that is not in the lock, so it undoes any ad-hoc `pip install`. Never `pip install` into this environment.
- Always export `UV_PROJECT_ENVIRONMENT` before running `uv sync` or `uv run`. Without it, uv creates a second environment at `./.venv`.
- Three ways to use the environment:
  - call `/data/code-switching/envs/liquid-ai-research/bin/python` directly,
  - `source /data/code-switching/envs/liquid-ai-research/bin/activate`,
  - `uv run --locked python ...`, with `UV_PROJECT_ENVIRONMENT` exported.
- Optional packages, not used for published numbers: `uv sync --locked --extra extras` adds streamlit 1.64.0 and faster-whisper 1.2.1 (ctranslate2 4.8.2). A later plain `uv sync --locked` removes them again.
- To check for drift, run `uv lock --check` (lock matches `pyproject.toml`) and `uv sync --locked --dry-run` (environment matches the lock; expect `Would make no changes`).
- To change a pin:
  1. Edit `pyproject.toml`.
  2. Run `uv lock`, then `uv sync --locked`.
  3. Re-run the tests below.
  4. Update the table below in the same commit.

  Numbers produced under different locks are not comparable until they are re-run under the same lock.
- uv cache: the default is `~/.cache/uv`, which is on the same filesystem as `/data`, so hard links work. If you set `UV_CACHE_DIR`, keep it on that same filesystem.

## Pinned versions

Built 2026-09-29 with uv 0.12.5. `uv.lock` sha256: `0a0049d0997753e10e4bf97294caf350e16c5805e839fb27aa40813e59030227`.

| Package | Version | Reason |
|---|---|---|
| Python | 3.12 (CPython 3.12.8, uv-managed) | liquid-audio requires `>=3.12` |
| torch | **2.11.0+cu126** | Newest torch that has a matching torchaudio on the cu126 index (see below). Ships the CUDA 12.6 runtime. |
| torchaudio | **2.11.0+cu126** | The last torchaudio release. No 2.12+ exists on PyPI or on the cu126 index. |
| transformers | **4.57.6** | Last 4.x release. liquid-audio 1.3.0's own `uv.lock` tests 4.56.1, and its code is written against the 4.x LFM2 cache API (`Lfm2HybridConvCache`, which is absent in transformers 5.14.1). |
| peft | **0.21.0** | Newest |
| accelerate | **1.15.0** | Newest |
| liquid-audio | **1.3.0** | Newest (2026-06-05). Unlike 1.2.0, it no longer needs torchcodec. |
| datasets | 4.8.5 | Newest 4.x. liquid-audio's lock tests 4.8.4. |
| huggingface-hub | 0.36.2 | transformers 4.57 requires `<1.0` |
| fugashi | **1.5.2** | See "Japanese dictionary" |
| unidic-lite | **1.0.8** (bundles UniDic 2.1.2) | See "Japanese dictionary" |
| tokenizers | 0.22.2 | Pulled in by transformers |
| numpy | 2.5.3 | |
| librosa | 1.0.0 | The mel filterbank that liquid-audio builds with `librosa.filters.mel` was checked to be bit-identical to librosa 0.11.0 (the version upstream tests) on 16 configurations. |
| soundfile | 0.14.0 | |
| jiwer | 4.0.0 | |
| regex | 2026.9.29 | |
| pyarrow / pandas | 25.0.1 / 3.0.6 | |
| safetensors / wandb / pytest | 0.8.0 / 0.30.0 / 9.1.1 | `wandb` is needed because `03-training/asr-ft/train_lora.py` imports it at module level |
| CUDA user-space libraries (from PyPI, required by torch) | cuda-toolkit 12.6.3, nvidia-cudnn-cu12 9.10.2.21, nvidia-nccl-cu12 2.28.9, triton 3.6.0 | |

**Why torch and torchaudio are held at the same version.** The torchaudio 2.11 wheel declares no
requirement on torch, so pip and uv will happily install it next to torch 2.12+. That is exactly
what the old `04-evaluation/eval-ft/requirements.txt` did (torch 2.12.0 with torchaudio 2.11.0).
But torchaudio's compiled extension is built against torch 2.11's C++ ABI. Holding both at 2.11.0
keeps them matched. Verified in this environment:
- `_IS_TORCHAUDIO_EXT_AVAILABLE=True`
- the compiled `forced_align` op runs
- `torchaudio.functional.resample` runs on the GPU

**Differences from liquid-audio 1.3.0's upstream lock.** Upstream pins torch 2.8.0, torchaudio 2.8.0,
transformers 4.56.1, accelerate 1.10.1, datasets 4.8.4, huggingface-hub 0.34.4 and librosa 0.11.0.
Ours stay on the same major versions, except:
- librosa: checked above.
- torch/torchaudio: liquid-audio only requires `>=2.8`. We take the newest pair that shares an ABI.

## Japanese dictionary (scorer)

JA-WER segmentation uses **fugashi 1.5.2 + unidic-lite 1.0.8**. The unidic-lite wheel bundles UniDic 2.1.2 (2013).
The full `unidic` package is deliberately **not** installed, for two reasons:

- `eval/score.py` calls `fugashi.Tagger()` with no arguments. That call picks `unidic` whenever it can be imported, so having both packages installed silently changes JA-WER.
- `unidic` fetches its dictionary data in a separate `python -m unidic download` step (~770 MB), which `uv.lock` does not cover. unidic-lite's dictionary ships inside the wheel, so the lock's hash pins it.

Evidence from this environment:

```
Tagger().dictionary_info = [{'filename': '/data/code-switching/envs/liquid-ai-research/lib/python3.12/site-packages/unidic_lite/dicdir/sys.dic',
                             'charset': 'utf8', 'size': 756264, 'version': 102}]
full `unidic` package: not importable
segmentation of "明日pull requestをmergeします": ['明日', 'pull', 'request', 'を', 'merge', 'し', 'ます']
```

In that output, `version: 102` is MeCab's binary dictionary format, not the UniDic release.

The older venv `/data/code-switching/.venv` has the full `unidic` package installed. JA-WER computed there is therefore not comparable with numbers from this environment.

## GPU and driver notes

- Driver 565.57.01 supports CUDA up to 12.7. The default torch on PyPI is a CUDA 13 build and fails with "driver too old". torch and torchaudio are therefore taken only from `https://download.pytorch.org/whl/cu126`, through `[tool.uv.sources]`. That index is marked `explicit = true`, so no other package is taken from it.
- Verified in this environment:
  - `torch.version.cuda == "12.6"`, cuDNN 91002
  - `torch.cuda.device_count() == 4`
  - a 4096x4096 bf16 matmul on each H100 NVL (sm_90), relative error against fp32 1.7e-3
- According to the docstring of `eval/run_whisper_hf.py` (not re-checked here), faster-whisper (CTranslate2) fails on this driver with `cudaErrorUnsupportedPtxVersion`. Whisper numbers therefore come from `eval/run_whisper_hf.py` (transformers + this torch). faster-whisper is only in the optional `extras` group.
- triton 3.6.0 ships its own `ptxas`. If `torch.compile` is ever used and fails with a PTX version error, that is the same driver limit. The current code does not use `torch.compile`.
- After a driver upgrade that supports CUDA 13:
  - switch the index to `cu130` and run `uv lock`,
  - treat the result as a new environment and re-run every number.
- The box is shared. Pin GPUs with `CUDA_VISIBLE_DEVICES`.

## Running the tests

```bash
cd <repo>
PY=/data/code-switching/envs/liquid-ai-research/bin/python
$PY 04-evaluation/eval-ft/tests/test_score.py      # all PASS (34/34)
$PY common/tests/test_leak_guard.py                # all PASS (14/14)
$PY -m pytest -q                                   # both suites via testpaths in pyproject.toml (36 passed)
make -C 04-evaluation/eval-ft test PYTHON=$PY      # the Makefile otherwise looks for ./.venv, then python3
```

## GPU smoke test (3 CS-FLEURS read_test clips, GPU 0)

```bash
cd <repo>/04-evaluation/eval-ft
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/code-switching/hf_cache HF_TOKEN_PATH=$HOME/.cache/huggingface/token
PY=/data/code-switching/envs/liquid-ai-research/bin/python
M=/data/code-switching/data/eval-real/csfleurs/read_test/manifest.jsonl
A=/data/code-switching/data/eval-real/csfleurs
$PY eval/run_baseline.py --model lfm --manifest $M --audio-root $A --out /tmp/lfm_read3.jsonl --limit 3
$PY eval/run_whisper_hf.py --model-id openai/whisper-large-v3 --manifest $M --audio-root $A --out /tmp/whisper_read3.jsonl --limit 3
$PY eval/score.py lfm:/tmp/lfm_read3.jsonl whisper:/tmp/whisper_read3.jsonl
```

Expected first hypotheses (`jpn_1662_SS`):
- LFM: `合金とは、ベースが a-mixed of 2種類以上のメロスです。…`
- Whisper: `ゴムキンとは、基本的に西米上のメタルの混合物です。…`

Both runners decode greedily, and two consecutive runs produced byte-identical outputs.

Model snapshots used:

| Model | Snapshot |
|---|---|
| `LiquidAI/LFM2.5-Audio-1.5B-JP` | `6c34b4d590f80563f8cb2939c2ebd7686d952394` |
| `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` |

The existing Whisper prediction files under `/data/code-switching/data/eval-preds/` do **not** reproduce
byte-for-byte in this environment (0/3 identical on these clips). Regenerate every prediction in this
environment rather than mixing old files with new ones.

## Known gaps (not fixed as part of the environment setup)

- `04-evaluation/eval-ft/requirements.txt` still lists torch 2.12.0 with torchaudio 2.11.0, which is an ABI mismatch, and does not pin versions. It should point to the root `pyproject.toml` / `uv.lock`.
- `04-evaluation/eval-ft/Makefile` looks for `<repo>/.venv/bin/python`. Pass `PYTHON=/data/code-switching/envs/liquid-ai-research/bin/python`.
- `eval/score.py` `ja_tokenizer_info()` records `"fugashi": None`, because fugashi 1.5.2 has no `__version__`. It should use `importlib.metadata.version("fugashi")` and also record `unidic-lite`.
- The runners do not pin model revisions. Record or pin the snapshot hashes listed above.
- The two Whisper paths are not equivalent. `eval/run_baseline.py --model whisper` uses faster-whisper with `beam_size=5`, while `eval/run_whisper_hf.py` decodes greedily by default. Use a single path for every published Whisper number.

## Model-specific environments

Some baselines need packages whose pins differ from the research lock. Each gets its own environment with the same torch build, and its full resolution is frozen in `envs/`:

| Environment | Purpose | Freeze file |
|---|---|---|
| `/data/code-switching/envs/qwen3-asr` | Qwen3-ASR 1.7B / 0.6B (`qwen-asr`; transformers 4.57.6, torch 2.11.0+cu126) | `envs/qwen3-asr.freeze.txt` |
| `/data/code-switching/envs/qwen3-tts` | Qwen3-TTS 1.7B/0.6B (`qwen-tts`; transformers 4.57.3, torch 2.11.0+cu126); SDPA attention, no flash-attn | `envs/qwen3-tts.freeze.txt` |
| `/data/code-switching/envs/funasr` | Fun-ASR-Nano (`funasr` 1.4.16 + Fun-ASR repo requirements @0339018; model code via trust_remote_code), torch 2.11.0+cu126 | `envs/funasr.freeze.txt` |
| `/data/code-switching/envs/qwen3-omni` | Qwen3-Omni-30B-A3B via transformers 4.57.6 (needs pillow + torchvision for its processor), torch 2.11.0+cu126; HF MoE is slow (~24 s/utt at batch 8) | `envs/qwen3-omni.freeze.txt` |

Recreate an environment exactly:

    uv venv /data/code-switching/envs/qwen3-asr --python 3.12
    uv pip install --python /data/code-switching/envs/qwen3-asr/bin/python \
        --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple -r envs/qwen3-asr.freeze.txt

## Multi-GPU (DDP) on mactrn01

Both trainers support `torchrun` data parallelism (`03-training/asr-ft/train_lora.py`, `train_whisper.py`). The login environment on mactrn01 sets `NCCL_SOCKET_IFNAME=eno1` and `NCCL_IB_DISABLE=1`, but the host has no `eno1` (its NICs are `ens17f0np0` and `ens17f1np1`). NCCL then fails with `Bootstrap : no socket interface found`. For single-node runs, bootstrap over loopback:

    NCCL_SOCKET_IFNAME=lo torchrun --nproc_per_node=4 03-training/asr-ft/train_lora.py ...

Verified 2026-09-30: an NCCL all-reduce and 2-GPU smoke runs of both trainers (rank-0 logging and saving, all-reduced dev loss) all worked. The effective batch is the per-GPU batch × world size, and the LR is not rescaled automatically.
