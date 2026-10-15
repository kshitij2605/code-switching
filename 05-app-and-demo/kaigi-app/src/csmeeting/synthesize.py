"""transcripts.jsonl  ->  wav files + manifest.jsonl.

For each transcript: derive language spans from the script (convention), pick a rotating bilingual speaker, and
either splice per-span voices (when there's a true English switch) or use a single Japanese voice (pure-loanword
utterances). Writes 24 kHz wav and one manifest row per utterance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from csmeeting import convention
from csmeeting.manifest import append_jsonl, read_jsonl
from csmeeting.tts_backends import make_backend, save_wav, silence


def synthesize(
    transcripts_path: str | Path,
    out_dir: str | Path,
    *,
    backend: str = "edge",
    speakers: list[dict] | None = None,
    target_sr: int = 24000,
    gap_ms: int = 60,
    limit: int = 0,
    placeholder: bool = False,
) -> Path:
    """Returns the manifest path. `placeholder=True` writes silent wavs (offline smoke; no network/TTS).

    `backend`: "edge" (default, network) | "kokoro" (Apache-2.0, local, commercial-clean). When `speakers` is None,
    the backend's own default speaker profiles are used (edge voice ids vs Kokoro voice ids).
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()  # fresh run

    tts, default_speakers = make_backend(backend, target_sr=target_sr, gap_ms=gap_ms)
    speakers = speakers or default_speakers

    rows = list(read_jsonl(transcripts_path))
    if limit and limit > 0:
        rows = rows[:limit]

    n_ok = n_err = 0
    for i, row in enumerate(rows):
        text = row["transcript"]
        speaker = speakers[i % len(speakers)]
        spans = convention.segment_into_spans(text)
        convention.validate_spans(text, spans)
        style = row.get("style") or convention.infer_style(text)

        try:
            if placeholder:
                audio = silence(target_sr, 0.6)
            elif style == "single_jp" or not convention.has_latin(text):
                audio = tts.synth_text(text, speaker["ja"])
            else:
                audio = tts.synth_spans(spans, speaker)
        except Exception as e:  # one bad utterance shouldn't kill the run
            n_err += 1
            print(f"[synthesize] ERROR on {row['id']}: {type(e).__name__}: {e}")
            continue

        wav_name = f"{row['id']}.wav"
        save_wav(str(audio_dir / wav_name), audio, target_sr)
        append_jsonl(manifest_path, {
            "id": row["id"],
            "audio_path": f"audio/{wav_name}",   # relative to the manifest's directory
            "transcript": text,
            "spans": spans,
            "style": style,
            "voice": speaker["name"],
            "domain": row.get("domain", "meeting"),
            "source": row.get("source", "unknown"),
            "duration_s": round(len(audio) / target_sr, 3),
            "sr": target_sr,
        })
        n_ok += 1
        if (i + 1) % 25 == 0:
            print(f"[synthesize] {i + 1}/{len(rows)} ...")

    print(f"[synthesize] done: {n_ok} ok, {n_err} errors -> {manifest_path}")
    return manifest_path


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
