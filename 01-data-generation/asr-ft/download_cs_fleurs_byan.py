"""
Download Japanese<->English code-switching TRAINING data from byan/cs-fleurs.

Only the `jpn-eng` subset of **xtts/train** is used (synthetic XTTS speech,
3 voices, ~5.4 h, 2,097 clips of ~800 sentences).

The other jpn-eng splits are protected and are never downloaded here:
  read/test   -> 196 real recordings: the frozen evaluation benchmark
  xtts/test1  -> 650 synthetic clips: held-out-engine dev set
Every output row passes common/leak_guard.py, which raises on any overlap.

The xtts text is MeCab-spaced with **English** markers, e.g.
  "旅行 **agents** **agents** の 多く は 、 ..."
clean_xtts_text() removes the markers and the tokeniser spacing but keeps the
words: the audio was synthesised from this text, so repeated words
("agents agents") are really spoken. Such rows get "garbled": true so that
downstream filters can drop them (--drop-garbled drops them here).

Licence: CC BY-NC 4.0 -> rows are tagged use="research" (paper-only runs).

Usage:
  python download_cs_fleurs_byan.py --data-root /data/lfm2.5/run-a
  # writes data/cs_fleurs_xtts_train/*.wav and
  #        data/training/cs_fleurs_xtts_train_rows.jsonl
"""
from __future__ import annotations

import argparse, json, os, random, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.csfleurs import LANG, REPO, SPLIT_ROLES, clean_xtts_text, is_garbled  # noqa: E402
from common.csfleurs import sentence_id as _sentence_id  # noqa: E402
from common.leak_guard import LeakGuard  # noqa: E402
from common.licenses import tag_rows  # noqa: E402
from common.run_registry import write_run_record  # noqa: E402

SPLIT = "xtts/train"
PROTECTED_SPLITS = tuple(s for s, role in SPLIT_ROLES.items() if role != "train")
OUT_NAME = "cs_fleurs_xtts_train_rows.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="/data/lfm2.5/run-a")
    p.add_argument("--hf-token",  default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--n-sentences", type=int, default=0,
                   help="Keep only N sentences (all their voices); 0 = all")
    p.add_argument("--drop-garbled", action="store_true",
                   help="Drop rows whose text repeats an English span (align-then-swap artefact)")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def sentence_id(row: dict) -> str:
    """'1097_jpn-spk0_sample0' -> '1097' (one FLEURS sentence, several voices)."""
    return _sentence_id(row["id"])


def main():
    args = parse_args()
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    from huggingface_hub import hf_hub_download

    guard = LeakGuard()  # fails fast if the blocklist is missing
    root = Path(args.data_root)
    out_audio = root / "data" / "cs_fleurs_xtts_train"
    out_audio.mkdir(parents=True, exist_ok=True)

    mf = hf_hub_download(REPO, f"{SPLIT}/metadata.jsonl", repo_type="dataset", token=args.hf_token)
    rows = [json.loads(l) for l in open(mf, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("language") == LANG]
    print(f"{SPLIT}: {len(rows)} jpn-eng clips, {len({sentence_id(r) for r in rows})} sentences")

    if args.n_sentences:
        sids = sorted({sentence_id(r) for r in rows})
        random.Random(args.seed).shuffle(sids)
        keep = set(sids[: args.n_sentences])
        rows = [r for r in rows if sentence_id(r) in keep]
        print(f"  kept {len(rows)} clips of {len(keep)} sentences (--n-sentences)")

    out_rows, n_garbled, fail = [], 0, 0
    for r in rows:
        garbled = is_garbled(r["text"])
        n_garbled += garbled
        if garbled and args.drop_garbled:
            continue
        text = clean_xtts_text(r["text"])
        if not text:
            continue
        repo_path = f"{SPLIT}/{r['file_name']}".replace("//", "/")
        fname = os.path.basename(repo_path)
        dest = out_audio / fname
        if not dest.exists():
            try:
                cached = hf_hub_download(REPO, repo_path, repo_type="dataset", token=args.hf_token)
            except Exception as e:
                fail += 1
                print(f"  fail {repo_path}: {e}")
                continue
            shutil.copy(cached, dest)
        out_rows.append({
            "id": f"csf_xtts_{r['id']}",
            "source_id": r["id"],
            "group": f"csf-xtts:{sentence_id(r)}",
            "speaker": r.get("speaker"),
            "audio_file": os.path.join("data", "cs_fleurs_xtts_train", fname),
            "system": "Perform ASR.",
            "target": text,
            "type": "cs_fleurs_xtts",
            "garbled": garbled,
        })
        if len(out_rows) % 200 == 0:
            print(f"  {len(out_rows)}/{len(rows)}", flush=True)

    if fail:
        raise SystemExit(f"{fail} audio downloads failed; rerun (finished files are kept)")
    out_rows = tag_rows(out_rows, "byan/cs-fleurs")
    guard.filter_rows(out_rows, on_leak="raise", context=OUT_NAME)

    out = root / "data" / "training" / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(out_rows)} rows -> {out}  (garbled: {n_garbled}"
          f"{', dropped' if args.drop_garbled else ', kept and flagged'})")
    write_run_record(out.parent, vars(args), inputs=[mf, str(out)],
                     extra={"rows": len(out_rows), "garbled": n_garbled,
                            "protected_splits_excluded": list(PROTECTED_SPLITS)},
                     filename=OUT_NAME.replace(".jsonl", ".build.json"))


if __name__ == "__main__":
    main()
