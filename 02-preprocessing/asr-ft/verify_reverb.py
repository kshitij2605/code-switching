#!/usr/bin/env python3
"""Sanity-check that reverb-augmented clips embed the dry signal (delayed).

Sample-aligned correlation reads ~0 for reverb because the RIR shifts the signal
by its propagation delay. A lag-tolerant cross-correlation recovers the true
relationship: a clear peak at a positive lag, with reverberant energy >= dry.
"""
import json
import sys
import numpy as np
import soundfile as sf
from scipy.signal import correlate

SR = 16000
ROOT = "data"
clean = {}
for man in ("edge_tts", "kokoro", "melo"):
    for line in open(f"{ROOT}/synth/{man}/manifest.jsonl"):
        r = json.loads(line)
        clean[r["id"]] = r["audio_path"]

aug = [json.loads(l) for l in open(f"{ROOT}/augmented/manifest.jsonl")]


def load(p):
    a, _ = sf.read(p)
    return (a.mean(1) if a.ndim > 1 else a).astype(np.float32)


revs = [x for x in aug if x["augmentation"] == "reverb"][:6]
for r in revs:
    c = load(clean[r["clean_id"]])
    x = load(r["audio_path"])
    L = min(len(c), len(x))
    c, x = c[:L], x[:L]
    cc = correlate(x, c, mode="full")
    lag = cc.argmax() - (len(c) - 1)
    peakcorr = cc.max() / (np.sqrt(np.sum(c ** 2) * np.sum(x ** 2)) + 1e-9)
    e_ratio = np.sum(x ** 2) / (np.sum(c ** 2) + 1e-9)
    rid = r["id"]
    print(f"{rid:28} lag={lag:+5d} smp ({lag / SR * 1000:+.0f} ms)  "
          f"peak-xcorr={peakcorr:.2f}  energy_ratio={e_ratio:.2f}")
