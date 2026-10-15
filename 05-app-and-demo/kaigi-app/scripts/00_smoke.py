#!/usr/bin/env python
"""Offline end-to-end smoke: seeds -> transcripts -> SILENT audio + manifest -> preview. No network, no model.

Proves the data-gen wiring works before you spend any GPU credit. For real audio, run 01 then 02 (without
--placeholder), which calls edge-tts (needs network).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.cs_asr_iterator import preview  # noqa: E402
from csmeeting.gen_transcripts import build  # noqa: E402
from csmeeting.manifest import write_jsonl  # noqa: E402
from csmeeting.synthesize import synthesize  # noqa: E402

OUT = ROOT / "data" / "synth"


def main() -> None:
    transcripts = build(
        seeds_path=ROOT / "data" / "seeds" / "meeting_codeswitch_seed.jsonl",
        terms_path=ROOT / "data" / "seeds" / "domain_terms.json",
        n_template=12, contrast_pairs=True, include_seeds=True,
        use_llm=False, llm_n=0, convention_path=ROOT / "docs" / "transcription_convention.md", seed=13,
    )
    tpath = OUT / "transcripts.smoke.jsonl"
    write_jsonl(tpath, transcripts)
    print(f"[smoke] generated {len(transcripts)} transcripts")

    manifest = synthesize(tpath, OUT, limit=6, placeholder=True)  # silent wavs, offline
    print("\n[smoke] manifest preview:")
    preview(manifest, 4)
    print("\n[smoke] PASS — wiring is good. Re-run 02 without --placeholder for real audio.")


if __name__ == "__main__":
    main()
