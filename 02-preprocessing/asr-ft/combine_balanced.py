"""
Combine N FLEURS clips per language + synthetic clips into balanced JSONL.
Run after download_fleurs_500.py (and, optionally, download_cs_fleurs_byan.py).

Notes:
  * CS-FLEURS read/test and xtts/test1 are never used for training. CS-FLEURS xtts/train
    (cs_fleurs_xtts_train_rows.jsonl) is added only with --include-cs-fleurs-xtts.
  * The train/eval split is group-aware (common.leak_guard.row_group_key): all
    recordings of a FLEURS sentence, all voices of a CS-FLEURS sentence and all
    rows of one synthetic clip land on the same side, and check_disjoint
    verifies this.
  * Every row passes the leak guard (test + dev roles) and gets a licence tag.
  * A build record with input hashes is written next to the outputs.

Usage:
  python combine_balanced.py --data-root /data/lfm2.5/run-a [--include-cs-fleurs-xtts]
"""
import argparse, collections, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.leak_guard import LeakGuard, check_disjoint, group_split, row_group_key  # noqa: E402
from common.licenses import ensure_tags  # noqa: E402
from common.run_registry import write_run_record  # noqa: E402

EXCLUDED = "cs_fleurs_byan_rows.jsonl"            # holds CS-FLEURS test rows: never read
CSF_XTTS = "cs_fleurs_xtts_train_rows.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="/data/lfm2.5/run-a")
    p.add_argument("--n-synthetic", type=int, default=300)
    p.add_argument("--synthetic-jsonl", nargs="+",
                   default=["data/training/native_asr_train.jsonl",
                            "data/training/native_asr_eval.jsonl"],
                   help="Multi-task synthetic JSONL(s); only *_native rows are used")
    p.add_argument("--include-cs-fleurs-xtts", action="store_true",
                   help=f"Add CS-FLEURS xtts/train rows ({CSF_XTTS}; CC BY-NC, research only)")
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    root = Path(args.data_root)
    tdir = root / "data" / "training"
    guard = LeakGuard()  # fails fast if the blocklist is missing

    if (tdir / EXCLUDED).exists():
        print(f"WARNING: ignoring {tdir / EXCLUDED}: it contains CS-FLEURS read/test "
              f"evaluation rows.",
              file=sys.stderr)

    # FLEURS rows (saved by download_fleurs_500.py)
    fleurs_path = tdir / "fleurs500_rows.jsonl"
    fleurs_rows = read_jsonl(fleurs_path)
    print(f"FLEURS loaded: {dict(collections.Counter(r['type'] for r in fleurs_rows))}")
    inputs = [str(fleurs_path)]

    cs_fleurs_rows = []
    if args.include_cs_fleurs_xtts:
        csf_path = tdir / CSF_XTTS
        cs_fleurs_rows = read_jsonl(csf_path)          # missing file -> error, not a warning
        inputs.append(str(csf_path))
        print(f"CS-FLEURS xtts/train loaded: {len(cs_fleurs_rows)} rows")

    # Synthetic rows from the multi-task JSONL(s), one row per audio clip
    by_syn = collections.defaultdict(list)
    seen_af = set()
    for p in args.synthetic_jsonl:
        inputs.append(str(root / p))
        for r in read_jsonl(root / p):
            t = r["type"]
            if t.startswith("fleurs") or not t.endswith("_native"):
                continue
            if r["audio_file"] not in seen_af:
                seen_af.add(r["audio_file"])
                by_syn[t].append(r)
    print(f"Synthetic unique: { {k: len(v) for k, v in by_syn.items()} }")

    n = args.n_synthetic
    # Use ALL cs_native (code-switching is the primary task)
    syn_cs = by_syn.get("cs_native", [])
    syn_en = rng.sample(by_syn.get("en_native", []), min(n, len(by_syn.get("en_native", []))))
    syn_ja = rng.sample(by_syn.get("ja_native", []), min(n, len(by_syn.get("ja_native", []))))
    print(f"Synthetic: CS={len(syn_cs)} (all) EN={len(syn_en)} JA={len(syn_ja)}")

    combined = ensure_tags(fleurs_rows + cs_fleurs_rows + syn_cs + syn_en + syn_ja)
    guard.filter_rows(combined, on_leak="raise", context="balanced mix")
    train_rows, eval_rows = group_split(combined, args.eval_frac, rng, row_group_key)
    check_disjoint(train_rows, eval_rows, context="balanced mix")
    rng.shuffle(train_rows)
    print(f"Total: {len(combined)} -> train {len(train_rows)} / eval {len(eval_rows)}")

    train_path = tdir / "balanced_train.jsonl"
    eval_path  = tdir / "balanced_eval.jsonl"
    for path, rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Written: {path}")
    write_run_record(tdir, vars(args), inputs=inputs + [str(train_path), str(eval_path)],
                     extra={"train": len(train_rows), "eval": len(eval_rows),
                            "by_use": dict(collections.Counter(r.get("license_use", "untagged")
                                                               for r in combined))},
                     filename="balanced_build.json")


if __name__ == "__main__":
    main()
