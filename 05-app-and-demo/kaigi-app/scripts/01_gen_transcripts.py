#!/usr/bin/env python
"""Generate code-switched meeting transcripts -> transcripts.jsonl (seeds + templated pairs [+ optional LLM])."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csmeeting.gen_transcripts import build  # noqa: E402
from csmeeting.manifest import write_jsonl  # noqa: E402


def load_cfg(path: str) -> dict:
    try:
        import yaml
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else ROOT / pp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "pipeline.yaml"))
    ap.add_argument("--out", help="output transcripts.jsonl (default from config)")
    ap.add_argument("--n-template", type=int, help="number of templated rows to emit")
    ap.add_argument("--no-seeds", action="store_true", help="exclude the hand-written seed bank")
    ap.add_argument("--use-llm", action="store_true", help="also expand with an OpenAI-compatible model (.env)")
    ap.add_argument("--llm-n", type=int, help="how many LLM utterances to request")
    ap.add_argument("--seed", type=int, help="RNG seed")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths, gen = cfg.get("paths", {}), cfg.get("generate", {})
    out = args.out or paths.get("transcripts", "data/synth/transcripts.jsonl")

    rows = build(
        seeds_path=resolve(paths.get("seeds", "data/seeds/meeting_codeswitch_seed.jsonl")),
        terms_path=resolve(paths.get("domain_terms", "data/seeds/domain_terms.json")),
        n_template=args.n_template if args.n_template is not None else gen.get("n_template", 300),
        contrast_pairs=gen.get("contrast_pairs", True),
        include_seeds=gen.get("include_seeds", True) and not args.no_seeds,
        use_llm=args.use_llm or gen.get("use_llm", False),
        llm_n=args.llm_n if args.llm_n is not None else gen.get("llm_n", 0),
        convention_path=ROOT / "docs" / "transcription_convention.md",
        seed=args.seed if args.seed is not None else gen.get("seed", 13),
    )
    n = write_jsonl(resolve(out), rows)
    print(f"wrote {n} transcripts -> {out}")
    print("by source:", dict(Counter(r["source"] for r in rows)))
    print("by style :", dict(Counter(r["style"] for r in rows)))


if __name__ == "__main__":
    main()
