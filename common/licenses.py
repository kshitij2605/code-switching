"""Licence registry: which data may train which model.

Policy: the product trains only on commercial-clean data;
research-only data (CC BY-NC corpora, edge-tts audio, ...) is allowed in
paper-only runs. Every manifest row therefore carries

    "license_source": registry key(s), e.g. "google/fleurs" or "tts/kokoro+noise/musan"
    "license":        the licence text of those sources
    "license_use":    "commercial" | "research" | "check"   ("check" = not yet
                      verified, treated as research-only)

Data scripts add these fields with ``tag_rows``; legacy manifests are tagged on
load with ``ensure_tags`` (from their ``engine`` / ``type`` fields). Trainers call
``enforce_scope(rows, scope)`` so that a ``--license-scope commercial`` run
refuses anything that is not verified commercial-clean. Derived audio (TTS,
augmentation) inherits the most restrictive licence of its inputs.

Details and sources: the dataset cards and licence pages linked in each entry.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable

USES = ("commercial", "research", "check")


@dataclass(frozen=True)
class Source:
    license: str
    use: str            # commercial | research | check
    url: str
    note: str = ""


REGISTRY: dict[str, Source] = {
    # --- speech corpora ---
    "google/fleurs": Source("CC-BY-4.0", "commercial",
                            "https://huggingface.co/datasets/google/fleurs"),
    "byan/cs-fleurs": Source("CC-BY-NC-4.0", "research",
                             "https://huggingface.co/datasets/byan/cs-fleurs",
                             "read/test and xtts/test1 are protected test/dev sets; "
                             "only xtts/train may be used for (research) training"),
    "byan/cs-yodas": Source("CC-BY-NC-4.0", "research",
                            "https://huggingface.co/datasets/byan/cs-yodas",
                            "jpn subset is a protected TEST set (T-MULTI); never train on it"),
    "Shelton1013/SwitchLingua_audio": Source(
        "CC-BY-NC-4.0 + authors' licensing agreement (research/non-commercial, no redistribution)", "research",
        "https://huggingface.co/datasets/Shelton1013/SwitchLingua_audio",
        "Japanese subset is a protected TEST set (T-MULTI); evaluation in research only; never train on it, "
        "never use it for product (commercial) decisions"),
    # --- TTS engines used for synthetic training audio (outputs inherit these) ---
    "tts/edge-tts": Source("Microsoft Edge Read Aloud online service (no training licence)",
                           "research", "https://github.com/rany2/edge-tts",
                           "integrity item F3: not commercial-clean"),
    # Verified 2026-09-30 against the model cards and VOICES.md.
    "tts/kokoro": Source("Apache-2.0 (Kokoro-82M weights)", "check",
                         "https://huggingface.co/hexgrad/Kokoro-82M",
                         "voices we use: jf_gongitsune, jf_nezumi, jf_tebukuro and jm_kumo were "
                         "trained on koniwa readings (CC BY: attribution); jf_alpha provenance is "
                         "undocumented. The model card states training included synthetic audio "
                         "from closed commercial TTS providers, which counsel should clear before "
                         "product use"),
    "tts/melo": Source("MIT (myshell-ai/MeloTTS-Japanese and -English checkpoints: 'free for both "
                       "commercial and non-commercial use')", "commercial",
                       "https://huggingface.co/myshell-ai/MeloTTS-Japanese",
                       "training data undisclosed by the authors"),
    "tts/qwen3-tts": Source("Apache-2.0 (Qwen/Qwen3-TTS checkpoints)", "commercial",
                            "https://github.com/QwenLM/Qwen3-TTS"),
    # A recording licence (e.g. CC BY) is not the speaker's consent to have their
    # voice cloned: cloned audio is research-grade until the consented voice bank exists.
    "voiceprompt/cloned": Source("voice cloned from a reference recording", "check",
                                 "project licence policy",
                                 "product data should clone only voice-bank talents who consented "
                                 "to AI voice synthesis"),
    # --- augmentation material (augmented audio inherits these too) ---
    # MUSAN per-file licences (musan/*/*/LICENSE, checked 2026-09-30):
    "noise/musan-noise": Source("CC BY 3.0 (sound-bible, 68 files) + public domain", "commercial",
                                "https://www.openslr.org/17/",
                                "attribution: musan/noise/*/LICENSE"),
    "noise/musan-speech": Source("US public domain (LibriVox, US government recordings)", "commercial",
                                 "https://www.openslr.org/17/"),
    "noise/musan-music": Source("per file: CC BY, CC BY-SA (45), CC BY-ND 3.0 (5), public domain",
                                "check", "https://www.openslr.org/17/",
                                "ask counsel whether mixing SA/ND tracks into training audio is an "
                                "adaptation; product runs should drop the music family or use "
                                "only its public-domain / CC BY tracks"),
    "noise/musan": Source("whole corpus: see the noise/speech/music entries", "check",
                          "https://www.openslr.org/17/", "legacy key for rows tagged before the split"),
    # RIRS_NOISES: the package is Apache-2.0, but its 325 real RIRs come from RWCP,
    # REVERB 2014 and Aachen AIR, each under its own terms.
    "rir/rirs_noises-sim": Source("Apache-2.0 (60,000 simulated RIRs)", "commercial",
                                  "https://www.openslr.org/26/"),
    "rir/rirs_noises-real": Source("RWCP / REVERB 2014 / Aachen AIR terms (not covered by the "
                                   "package's Apache-2.0)", "check", "https://www.openslr.org/28/"),
    "rir/rirs_noises": Source("real + simulated RIRs: see the -real/-sim entries", "check",
                              "https://www.openslr.org/28/", "legacy key: which RIR was used is unknown"),
    # --- LLM-generated text (the exact model is recorded in each row's "model" field) ---
    "llm/apache-2.0": Source("text generated by an Apache-2.0 open-weights LLM", "commercial",
                             "project licence policy", "only for models verified as Apache-2.0 (e.g. "
                             "Qwen3 dense); other licences need their own registry key"),
    # --- machine translation used to build targets ---
    "mt/qwen": Source("depends on checkpoint (Apache-2.0 or Qwen licence)", "check",
                      "https://huggingface.co/Qwen", "record the exact checkpoint"),
}

# Legacy manifests: infer the registry key from the row's engine / type field.
ENGINE_SOURCES = {"edge_tts": "tts/edge-tts", "edge-tts": "tts/edge-tts", "edge": "tts/edge-tts",
                  "kokoro": "tts/kokoro", "melo": "tts/melo"}
TYPE_PREFIX_SOURCES = (("fleurs_", "google/fleurs"), ("cs_fleurs", "byan/cs-fleurs"),
                       # LFM-pipeline synthetic data (generate_{cs,en,ja}_data.py) is edge-tts
                       ("cs_", "tts/edge-tts"), ("en_", "tts/edge-tts"), ("ja_", "tts/edge-tts"))
AUGMENTATION_SOURCES = {"noise": ["noise/musan-noise"], "music": ["noise/musan-music"],
                        "babble": ["noise/musan-speech"], "reverb": ["rir/rirs_noises"],
                        "reverb_real": ["rir/rirs_noises-real"], "reverb_sim": ["rir/rirs_noises-sim"],
                        # augment_data.py: aug1/aug2 = MUSAN noise, aug3 = MUSAN music
                        "aug1": ["noise/musan-noise"], "aug2": ["noise/musan-noise"],
                        "aug3": ["noise/musan-music"], "telephony": []}


def lookup(source: str) -> Source:
    try:
        return REGISTRY[source]
    except KeyError:
        raise KeyError(f"unknown data source {source!r}; add it to common/licenses.py "
                       f"with a verified licence first") from None


def combine_use(*uses: str) -> str:
    """Most restrictive use of several inputs: research > check > commercial."""
    for u in ("research", "check"):
        if u in uses:
            return u
    return "commercial"


def tag_row(row: dict, source: str, derived_from: Iterable[str] = ()) -> dict:
    derived_from = [s for s in derived_from if s != source]
    srcs = [lookup(source)] + [lookup(s) for s in derived_from]
    row = dict(row)
    row["license_source"] = "+".join([source, *derived_from])
    row["license"] = " + ".join(s.license for s in srcs)
    row["license_use"] = combine_use(*(s.use for s in srcs))
    return row


def tag_rows(rows: Iterable[dict], source: str, derived_from: Iterable[str] = ()) -> list[dict]:
    """Add license_source/license/license_use. ``derived_from`` lists further
    registry keys whose licences the rows inherit (e.g. the noise corpus)."""
    derived_from = list(derived_from)
    return [tag_row(r, source, derived_from) for r in rows]


def add_derivation(row: dict, extra_sources: Iterable[str]) -> dict:
    """Re-tag a (possibly already tagged) row whose audio was further derived
    from ``extra_sources`` (e.g. mixed with MUSAN noise). Untaggable rows are
    returned unchanged."""
    extra = list(extra_sources)
    if not row.get("license_source"):
        row = ensure_tags([row])[0]
        if not row.get("license_source"):
            return row
    base, *prev = row["license_source"].split("+")
    return tag_row(row, base, list(dict.fromkeys(prev + extra)))


def augmentation_sources(augmentation: str | None) -> list[str]:
    """'noise_snr10' -> ['noise/musan-noise']; 'reverb_sim' -> ['rir/rirs_noises-sim'];
    'reverb' (legacy, RIR kind unknown) -> ['rir/rirs_noises']; 'telephony' -> []."""
    if not augmentation:
        return []
    parts = str(augmentation).split("_")
    if parts[0] == "reverb" and len(parts) > 1 and f"reverb_{parts[1]}" in AUGMENTATION_SOURCES:
        return AUGMENTATION_SOURCES[f"reverb_{parts[1]}"]
    if parts[0] not in AUGMENTATION_SOURCES:
        raise KeyError(f"unknown augmentation family {parts[0]!r}; add it to AUGMENTATION_SOURCES")
    return AUGMENTATION_SOURCES[parts[0]]


def refresh_tags(row: dict) -> dict:
    """Recompute license / license_use from the *current* registry for a row that
    carries license_source, so registry corrections reach manifests written earlier."""
    src = row.get("license_source")
    if not src:
        return row
    base, *rest = src.split("+")
    return tag_row(row, base, rest)


def infer_source(row: dict) -> str | None:
    """Registry key for a legacy row without tags, or None if it cannot be told."""
    eng = row.get("engine")
    if eng in ENGINE_SOURCES:
        return ENGINE_SOURCES[eng]
    typ = str(row.get("type") or "")
    for prefix, src in TYPE_PREFIX_SOURCES:
        if typ.startswith(prefix):
            return src
    return None


def ensure_tags(rows: Iterable[dict], default_source: str | None = None) -> list[dict]:
    """Refresh tagged rows from the current registry and tag rows that carry no
    licence fields (legacy manifests). Untaggable rows are left untagged: allowed
    in research scope, refused in commercial scope."""
    out = []
    for r in rows:
        if r.get("license_source"):
            out.append(refresh_tags(r))
            continue
        src = infer_source(r) or default_source
        out.append(tag_row(r, src, augmentation_sources(r.get("augmentation"))) if src else r)
    return out


def enforce_scope(rows: list[dict], scope: str, context: str = "") -> list[dict]:
    """scope='research': allow everything (report untagged rows).
    scope='commercial': raise unless every row is tagged license_use='commercial'."""
    if scope not in ("research", "commercial"):
        raise ValueError(f"scope must be research|commercial, got {scope!r}")
    rows = [refresh_tags(r) for r in rows]      # never trust a stored, possibly stale, tag
    tag = f"[licence{': ' + context if context else ''}]"
    counts: dict[str, int] = {}
    for r in rows:
        u = r.get("license_use", "untagged")
        counts[u] = counts.get(u, 0) + 1
    print(f"{tag} scope={scope} rows={len(rows)} by use: {counts}")
    if scope == "commercial":
        bad = {u: n for u, n in counts.items() if u != "commercial"}
        if bad:
            raise ValueError(f"{tag} commercial scope refuses non-commercial rows: {bad}. "
                             f"Remove them or verify their licences in common/licenses.py.")
    elif counts.get("untagged"):
        print(f"{tag} WARNING: {counts['untagged']} rows carry no licence tag", file=sys.stderr)
    return rows
