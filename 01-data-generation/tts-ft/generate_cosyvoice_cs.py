#!/usr/bin/env python
"""Render single-switch code-switch transcripts natively with CosyVoice2 (one utterance, real switch-point prosody,
no splice seams). RUN IN THE cosyvoice-venv with PYTHONPATH=<cosyvoice_repo>:<cosyvoice_repo>/third_party/Matcha-TTS.

Reads CS rows (style=='cs') from an input manifest, generates audio in a registered zero-shot voice (a Kokoro prompt),
and writes wav + a manifest in our schema (absolute paths) for mix_manifests/preprocess.

  PYTHONPATH=... CUDA_VISIBLE_DEVICES=1 python scripts/generate_cosyvoice_cs.py \
      --in data/cs_mixed/train_lowdens.jsonl --out-dir data/cosy_cs \
      --model /data/lfm2.5/run-c/cosyvoice_repo/pretrained_models/CosyVoice2-0.5B \
      --prompt-wav /data/lfm2.5/run-c/data/cosy_prompt/prompt_ja.wav \
      --prompt-text-file /data/lfm2.5/run-c/data/cosy_prompt/prompt_ja.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "05-app-and-demo" / "kaigi-app"  # stage-folder copy of kaigi-app/scripts
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.manifest import append_jsonl, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="input manifest (CS rows used; mono ignored)")
    ap.add_argument("--out-dir", default="data/cosy_cs")
    ap.add_argument("--model", required=True, help="CosyVoice2-0.5B model dir")
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--prompt-text-file", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2

    m = CosyVoice2(args.model, load_jit=False, load_trt=False, fp16=False)
    sr = m.sample_rate
    ptext = Path(args.prompt_text_file).read_text(encoding="utf-8").strip()
    m.add_zero_shot_spk(ptext, args.prompt_wav, "cs_spk")   # register once -> reuse (faster)

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    # CS = everything that isn't mono replay (our Kokoro CS is styled 'splice'/'single_jp', not 'cs')
    rows = [r for r in read_jsonl(args.inp) if not str(r.get("style", "")).startswith("mono")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"rendering {len(rows)} CS utterances with CosyVoice2 ...")
    n_ok = n_err = 0
    for i, row in enumerate(rows):
        text = row["transcript"]
        try:
            outs = list(m.inference_zero_shot(text, "", "", zero_shot_spk_id="cs_spk", stream=False))
            import torch
            wav = torch.cat([o["tts_speech"] for o in outs], dim=-1) if outs else None
            if wav is None or wav.shape[-1] == 0:
                raise RuntimeError("empty audio")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"  ERR {row['id']}: {type(e).__name__}: {e}")
            continue
        uid = f"cosy_{row['id']}"
        torchaudio.save(str(audio_dir / f"{uid}.wav"), wav, sr)
        append_jsonl(manifest_path, {
            "id": uid,
            "audio_path": str((audio_dir / f"{uid}.wav").resolve()),
            "transcript": text,
            "spans": row.get("spans", []),
            "style": "cs",
            "voice": "cosyvoice2",
            "domain": row.get("domain", "meeting"),
            "source": "cosy_cs",
            "duration_s": round(wav.shape[-1] / sr, 3),
            "sr": sr,
        })
        n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} ...")
    print(f"done: {n_ok} ok, {n_err} err -> {manifest_path}")


if __name__ == "__main__":
    main()
