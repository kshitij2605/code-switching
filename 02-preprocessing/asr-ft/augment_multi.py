#!/usr/bin/env python3
"""Multi-type acoustic augmentation for the synthetic CS corpus (issue #8).

Takes clean engine manifests and produces augmented copies spanning several
distortion families — deliberately "beyond MUSAN noise":

  noise      additive MUSAN noise at a swept SNR
  music      additive MUSAN music at a swept SNR
  babble     several MUSAN speech files summed -> multi-speaker babble
  reverb     convolution with a room impulse response (RIRS_NOISES)
  telephony  300-3400 Hz band-limit + G.711 mu-law codec round-trip

The reference transcript never changes — augmentation alters acoustics, not
words — so every augmented row keeps the source row's transcript / en_words /
segments and just swaps audio_path and records the augmentation applied.

RIRs come only from the corpus's own rir_list files (325 real + 60,000
simulated); the corpus's point-source and isotropic *noise* recordings are
never used as RIRs.
Multichannel RIRs use channel 0, and each RIR is trimmed to start at its
direct-path peak so the reverberant clip stays time-aligned.

Augmented rows inherit the licence of the exact material mixed in
(common/licenses.py): MUSAN noise, speech or music, and real or simulated RIR,
with reverb rows named reverb_real / reverb_sim. For commercial-clean
(product) audio use
    --families noise,babble,reverb,telephony --rir-kinds sim
because MUSAN music has CC BY-SA/ND tracks and the real RIRs have third-party
terms.

Run (on mactrn01, in the venv):
  .venv/bin/python scripts/02-preprocessing/asr-ft/augment_multi.py \
     --manifests data/synth/edge_tts/manifest.jsonl \
                 data/synth/kokoro/manifest.jsonl \
                 data/synth/melo/manifest.jsonl \
     --musan  data/noise_sources/musan \
     --rir    data/noise_sources/RIRS_NOISES \
     --out    data/augmented --seed 42
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.licenses import add_derivation, augmentation_sources  # noqa: E402

SR = 16000

# --- I/O ----------------------------------------------------------------------

def load_wav(path: str, target_sr: int = SR) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != target_sr:
        a = resample_poly(a, target_sr, sr).astype(np.float32)
    return a


def peak_norm(a: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(a))) if a.size else 0.0
    return (a * (peak / m)).astype(np.float32) if m > peak else a.astype(np.float32)


# --- augmentation primitives --------------------------------------------------

def _fit_len(noise: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    if len(noise) > n:
        s = rng.randint(0, len(noise) - n)
        noise = noise[s:s + n]
    return noise


def add_at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: random.Random) -> np.ndarray:
    noise = _fit_len(noise, len(sig), rng)
    s_rms = np.sqrt(np.mean(sig ** 2)) + 1e-9
    n_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    noise = noise * ((s_rms / (10 ** (snr_db / 20))) / n_rms)
    return peak_norm(sig + noise)


def make_babble(speech_pool: list[str], n: int, k: int,
                rng: random.Random) -> np.ndarray:
    """Sum k random speech files into a multi-speaker babble bed of length n."""
    bed = np.zeros(n, dtype=np.float32)
    for _ in range(k):
        sp = load_wav(rng.choice(speech_pool))
        bed += _fit_len(sp, n, rng)
    return bed


def load_rir(path: str, target_sr: int = SR) -> np.ndarray:
    """Channel 0 of a (possibly multichannel) RIR, resampled and trimmed to start
    at the direct-path peak. Averaging channels would smear mics with different delays."""
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a[:, 0]
    if sr != target_sr:
        a = resample_poly(a, target_sr, sr).astype(np.float32)
    return a[int(np.argmax(np.abs(a))):] if a.size else a


def apply_reverb(sig: np.ndarray, rir_path: str) -> np.ndarray:
    rir = load_rir(rir_path)
    if rir.size == 0:
        return sig
    rir = rir / (np.max(np.abs(rir)) + 1e-9)
    wet = fftconvolve(sig, rir)[:len(sig)]        # keep original length
    return peak_norm(wet)


def apply_telephony(sig: np.ndarray) -> np.ndarray:
    """Band-limit to 300-3400 Hz then mu-law (G.711) encode/decode."""
    sos = butter(6, [300 / (SR / 2), 3400 / (SR / 2)], btype="band", output="sos")
    band = sosfilt(sos, sig).astype(np.float32)
    mu = 255.0
    band = np.clip(band, -1.0, 1.0)
    comp = np.sign(band) * np.log1p(mu * np.abs(band)) / np.log1p(mu)   # encode
    q = np.round((comp + 1) / 2 * mu)                                   # 8-bit
    comp = q / mu * 2 - 1
    dec = np.sign(comp) * (1 / mu) * ((1 + mu) ** np.abs(comp) - 1)     # decode
    return peak_norm(dec.astype(np.float32))


# --- driver -------------------------------------------------------------------

def collect(root: str, sub: str) -> list[str]:
    files = glob.glob(os.path.join(root, sub, "**", "*.wav"), recursive=True)
    return files


FAMILIES = ("noise", "music", "babble", "reverb", "telephony")


def rir_kind(path: str) -> str:
    """'real' (RWCP / REVERB 2014 / AIR, third-party terms) or 'sim' (Apache-2.0)."""
    return "real" if "real_rirs_isotropic_noises" in path else "sim"


def collect_rirs(rir_root: str, kinds: tuple[str, ...] = ("real", "sim")) -> list[str]:
    """RIR paths listed in RIRS_NOISES' rir_list files (never its noise_list files),
    restricted to `kinds`. List entries are relative to the directory that
    contains RIRS_NOISES/."""
    lists = sorted(glob.glob(os.path.join(rir_root, "**", "rir_list"), recursive=True))
    assert lists, f"no rir_list files under {rir_root}"
    base = Path(rir_root).resolve().parent
    rirs = []
    for lf in lists:
        for line in Path(lf).read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = str(base / line.split()[-1])
                if rir_kind(p) in kinds:
                    rirs.append(p)
    missing = [p for p in rirs[:50] if not os.path.exists(p)]
    assert not missing, f"rir_list entries not found, e.g. {missing[:3]}"
    return rirs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--musan", required=True, help="MUSAN root (noise/ music/ speech/)")
    ap.add_argument("--rir", required=True, help="RIRS_NOISES root")
    ap.add_argument("--out", default="data/augmented")
    ap.add_argument("--snr-sweep", nargs="+", type=float, default=[20, 15, 10, 5])
    ap.add_argument("--babble-k", type=int, default=4, help="speakers per babble bed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="Augmentation families to apply. Product (commercial-scope) runs: drop "
                         "'music' (MUSAN music has CC BY-SA/ND tracks, see common/licenses.py)")
    ap.add_argument("--rir-kinds", default="real,sim",
                    help="RIRs to use: real (third-party terms) and/or sim (Apache-2.0). "
                         "Product runs: sim")
    args = ap.parse_args()
    fams = [f for f in args.families.split(",") if f]
    bad = [f for f in fams if f not in FAMILIES]
    assert not bad, f"unknown families {bad}; choose from {FAMILIES}"

    # With the default options the random draws happen in the original order, so
    # a seed reproduces earlier runs (reverb rows are now named reverb_real/_sim).
    rng = random.Random(args.seed)
    noise_pool = collect(args.musan, "noise")
    music_pool = collect(args.musan, "music")
    speech_pool = collect(args.musan, "speech")
    rir_pool = collect_rirs(args.rir, tuple(k for k in args.rir_kinds.split(",") if k))
    need = {"noise": noise_pool, "music": music_pool, "babble": speech_pool, "reverb": rir_pool}
    for fam, pool in need.items():
        assert fam not in fams or pool, f"empty pool for family {fam!r} under given roots"
    print(f"pools: noise={len(noise_pool)} music={len(music_pool)} "
          f"speech={len(speech_pool)} rir={len(rir_pool)}")

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    written = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for man in args.manifests:
            rows = [json.loads(l) for l in Path(man).read_text(encoding="utf-8").splitlines() if l.strip()]
            print(f"[{man}] {len(rows)} clean rows")
            for row in rows:
                sig = load_wav(row["audio_path"])
                if sig.size == 0:
                    continue
                # one augmentation of each family per clip (SNR/file drawn at random)
                snr_n = rng.choice(args.snr_sweep)
                snr_m = rng.choice(args.snr_sweep)
                snr_b = rng.choice(args.snr_sweep)
                plan = []
                if "noise" in fams:
                    plan.append((f"noise_snr{int(snr_n)}",
                                 add_at_snr(sig, load_wav(rng.choice(noise_pool)), snr_n, rng)))
                if "music" in fams:
                    plan.append((f"music_snr{int(snr_m)}",
                                 add_at_snr(sig, load_wav(rng.choice(music_pool)), snr_m, rng)))
                if "babble" in fams:
                    plan.append((f"babble_snr{int(snr_b)}",
                                 add_at_snr(sig, make_babble(speech_pool, len(sig), args.babble_k, rng),
                                            snr_b, rng)))
                if "reverb" in fams:
                    rir = rng.choice(rir_pool)
                    plan.append((f"reverb_{rir_kind(rir)}", apply_reverb(sig, rir)))
                if "telephony" in fams:
                    plan.append(("telephony", apply_telephony(sig)))
                for aug_name, aug_audio in plan:
                    uid = f"{row['id']}_{aug_name}"
                    wav_path = audio_dir / f"{uid}.wav"
                    sf.write(str(wav_path), aug_audio, SR, subtype="PCM_16")
                    new = dict(row)
                    new.update({
                        "id": uid,
                        "audio_path": str(wav_path.resolve()),
                        "duration_s": round(len(aug_audio) / SR, 3),
                        "augmentation": aug_name,
                        "clean_id": row["id"],
                        "source": f"augmented/{row.get('engine','?')}",
                    })
                    new = add_derivation(new, augmentation_sources(aug_name))
                    mf.write(json.dumps(new, ensure_ascii=False) + "\n")
                    written += 1

    from collections import Counter
    kinds = Counter(json.loads(l)["augmentation"].split("_")[0]
                    for l in manifest_path.read_text(encoding="utf-8").splitlines())
    print(f"\ndone: {written} augmented clips -> {manifest_path}")
    print(f"  by family: {dict(kinds)}")


if __name__ == "__main__":
    main()
