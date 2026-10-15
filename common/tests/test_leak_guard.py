#!/usr/bin/env python3
"""Unit tests for common/leak_guard.py and common/build_eval_blocklist.py.

Run directly:   python common/tests/test_leak_guard.py
Or with pytest: pytest common/tests/test_leak_guard.py
"""
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import leak_guard as lg  # noqa: E402
from common.build_eval_blocklist import Blocklist  # noqa: E402

TEST_SENT = "合金とは、basically a mixture of 2種類以上のmetalsです。"
EN_TEST = "The scientists announced a new diagnostic tool that sorts cells by type."
DEV_SENT = "旅行agents agentsの多くは、特定のホテルとdealsしています。"


def _guard(tmp: Path, roles=lg.ROLES) -> lg.LeakGuard:
    path = tmp / "blocklist.json"
    if not path.exists():
        bl = Blocklist(lg.NGRAM_N)
        bl.add("csfleurs-like", [{"id": "jpn_1662_SS", "file_name": "audio/jpn/jpn_1662_SS.wav",
                                  "text": TEST_SENT}], "strict+ids")
        bl.add("fleurs-like", [{"transcription": EN_TEST}], "strict")
        bl.add("legacy-synth", [{"transcript": "今日の project がちょっと心配"}], "exact")
        bl.add("dev-like", [{"id": "1097_jpn-spk0_sample0", "text": DEV_SENT}], "strict+ids", "dev")
        bl.dump(path)
    return lg.LeakGuard(path, roles=roles)


def test_exact_match_ignores_formatting():
    with tempfile.TemporaryDirectory() as d:
        g = _guard(Path(d))
        # MeCab spacing, ** markers, case and full-width forms must not hide a copy.
        reformatted = "合金 とは 、 **Basically** **a** mixture of ２種類以上の metals です"
        assert "text:target:exact@test" in g.check_row({"target": reformatted})


def test_ngram_overlap_catches_edited_copy_without_false_positives():
    with tempfile.TemporaryDirectory() as d:
        g = _guard(Path(d))
        edited = EN_TEST.replace("new", "novel")          # light edit -> still a leak
        reasons = g.check_row({"text": edited})
        assert any(r.startswith("text:text:ngram") for r in reasons), reasons
        unrelated = "Let's review the quarterly sales pipeline before the deadline."
        assert g.check_row({"text": unrelated}) == []


def test_exact_level_sources_do_not_block_shared_segments():
    with tempfile.TemporaryDirectory() as d:
        g = _guard(Path(d))
        # Same template segments, different sentence -> allowed (exact level only).
        assert g.check_row({"transcript": "今日の project がちょっと心配なんだ ついでに team"}) == []
        assert g.check_row({"transcript": "今日の project がちょっと心配"}) != []


def test_id_and_audio_matching():
    with tempfile.TemporaryDirectory() as d:
        g = _guard(Path(d))
        assert "id:source_id@test" in g.check_row({"source_id": "jpn_1662_SS", "target": "x"})
        assert "audio:audio_file@test" in g.check_row({"audio_file": "/data/cs/jpn_1662_SS.wav"})
        assert g.check_row({"id": "ja_jp_1", "audio_file": "ja_jp_1.wav", "target": "y"}) == []


def test_dev_role_blocks_training_but_not_validation():
    with tempfile.TemporaryDirectory() as d:
        row = {"id": "x", "target": DEV_SENT}
        assert "text:target:exact@dev" in _guard(Path(d)).check_row(row)       # training
        assert _guard(Path(d), roles=("test",)).check_row(row) == []           # validation
        assert _guard(Path(d), roles=("test",)).check_row({"target": TEST_SENT}) != []


def test_v1_blocklist_is_read_as_test_role():
    with tempfile.TemporaryDirectory() as d:
        import json
        path = Path(d) / "v1.json"
        path.write_text(json.dumps({"version": 1, "ngram_n": lg.NGRAM_N, "ids": ["jpn_1_SS"],
                                    "audio_basenames": [], "text_hashes": [], "ngram_hashes": []}))
        assert lg.LeakGuard(path).check_row({"id": "jpn_1_SS"}) == ["id:id@test"]


def test_check_disjoint_catches_shared_groups_and_audio():
    tr = [{"group": "csf-xtts:1097", "audio_file": "a/1.wav", "target": "t1"}]
    ok = [{"group": "csf-xtts:362", "audio_file": "a/2.wav", "target": "t2"}]
    lg.check_disjoint(tr, ok)
    for bad in ([{"group": "csf-xtts:1097", "audio_file": "a/3.wav", "target": "t3"}],
                [{"audio_file": "a/1.wav", "target": "a different task target"}]):
        try:
            lg.check_disjoint(tr, bad)
            assert False, "expected LeakError"
        except lg.LeakError:
            pass


def test_licence_tags_and_scope():
    from common import licenses as lic
    rows = lic.tag_rows([{"id": 1}], "tts/kokoro", derived_from=["rir/rirs_noises"])
    assert rows[0]["license_use"] == "check" and rows[0]["license_source"] == "tts/kokoro+rir/rirs_noises"
    assert lic.combine_use("commercial", "research", "check") == "research"
    fleurs = lic.tag_rows([{"id": 2}], "google/fleurs")
    lic.enforce_scope(fleurs, "commercial")
    for bad in (rows, [{"id": 3}]):
        try:
            lic.enforce_scope(bad, "commercial")
            assert False, "expected ValueError"
        except ValueError:
            pass
    lic.enforce_scope(rows + [{"id": 3}], "research")          # allowed, with a warning
    try:
        lic.tag_rows([{"id": 4}], "some/unregistered-corpus")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_augmentation_licences_follow_the_material_used():
    from common import licenses as lic
    assert lic.augmentation_sources("reverb_sim") == ["rir/rirs_noises-sim"]
    assert lic.augmentation_sources("reverb_real") == ["rir/rirs_noises-real"]
    assert lic.augmentation_sources("reverb") == ["rir/rirs_noises"]          # legacy: kind unknown
    assert lic.augmentation_sources("music_snr10") == ["noise/musan-music"]
    assert lic.augmentation_sources("aug3") == ["noise/musan-music"]
    clean = lic.add_derivation(lic.tag_row({"id": 1}, "tts/melo"), lic.augmentation_sources("reverb_sim"))
    assert clean["license_use"] == "commercial", clean
    assert lic.add_derivation(clean, lic.augmentation_sources("music_snr5"))["license_use"] == "check"


def test_stored_tags_are_refreshed_from_the_registry():
    from common import licenses as lic
    stale = {"id": 1, "license_source": "tts/melo+rir/rirs_noises",
             "license": "old text", "license_use": "commercial"}      # written before a correction
    fresh = lic.ensure_tags([stale])[0]
    assert fresh["license_use"] == "check" and fresh["license"] != "old text", fresh
    try:
        lic.enforce_scope([stale], "commercial")    # refreshes itself: the stale tag can't slip through
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_filter_rows_modes():
    with tempfile.TemporaryDirectory() as d:
        g = _guard(Path(d))
        rows = [{"id": "ok", "target": "Let's ship the release"}, {"id": "bad", "target": TEST_SENT}]
        try:
            g.filter_rows(rows)
            assert False, "expected LeakError"
        except lg.LeakError:
            pass
        assert [r["id"] for r in g.filter_rows(rows, on_leak="drop")] == ["ok"]
        assert len(g.filter_rows(rows, on_leak="warn")) == 2


def test_missing_blocklist_is_an_error():
    try:
        lg.LeakGuard("/nonexistent/blocklist.json")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_group_split_keeps_groups_together():
    rows = []
    for t in range(20):
        for engine in ("edge", "kokoro", "melo"):
            rows.append({"transcript": f"sentence number {t}", "engine": engine})
    train, evl = lg.group_split(rows, 0.1, random.Random(0), lg.transcript_group_key)
    train_keys = {lg.transcript_group_key(r) for r in train}
    eval_keys = {lg.transcript_group_key(r) for r in evl}
    assert train_keys.isdisjoint(eval_keys)
    assert len(evl) == 6 and len(train) == 54   # 2 of 20 groups x 3 engines


try:  # under pytest a Skip is reported as skipped, not as an error
    import pytest
    Skip = pytest.skip.Exception
except ImportError:
    class Skip(Exception):
        pass


def _csfleurs_metadata(split: str) -> list[dict]:
    """jpn-eng rows of a cached byan/cs-fleurs split, or Skip if not cached."""
    import glob
    import json
    caches = [os.environ.get("HF_HOME", ""), os.path.expanduser("~/.cache/huggingface"),
              "/data/code-switching/hf_cache"]
    for c in filter(None, caches):
        for root in (os.path.join(c, "hub"), c):
            hits = glob.glob(os.path.join(root, "datasets--byan--cs-fleurs", "snapshots", "*",
                                          split, "metadata.jsonl"))
            if hits:
                rows = [json.loads(l) for l in open(hits[0], encoding="utf-8") if l.strip()]
                return [r for r in rows if r.get("language") == "jpn-eng"]
    raise Skip(f"byan/cs-fleurs {split} not in any HF cache")


def test_committed_blocklist_catches_renamed_test_rows():
    """Positive control on real data: CS-FLEURS read/test rows with renamed audio,
    no id and '**' stripped must all be caught, and CS-FLEURS xtts/train must pass."""
    if not lg.DEFAULT_BLOCKLIST.exists():
        raise Skip("common/eval_blocklist.json not built")
    g = lg.LeakGuard()                       # training check: test + dev roles
    g_test = lg.LeakGuard(roles=("test",))   # validation check: test role only

    def old_row(i, r):
        text = " ".join(r["text"].replace("**", "").split())
        return {"audio_file": f"data/cs_fleurs_byan/csf_{i:04d}.wav", "target": text}

    read = _csfleurs_metadata("read/test")
    missed = [r["id"] for i, r in enumerate(read) if not g_test.check_row(old_row(i, r))]
    assert read and not missed, f"read/test: {len(missed)} of {len(read)} rows not caught"
    test1 = _csfleurs_metadata("xtts/test1")
    missed = [r["id"] for i, r in enumerate(test1) if not g.check_row(old_row(i, r))]
    assert test1 and not missed, f"xtts/test1: {len(missed)} of {len(test1)} rows not caught"
    train = _csfleurs_metadata("xtts/train")
    flagged = [r["id"] for i, r in enumerate(train) if g.check_row(old_row(i, r))]
    assert not flagged, f"xtts/train falsely flagged: {flagged[:5]}"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Skip as e:
            skipped += 1
            print(f"SKIP  {fn.__name__}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed - skipped}/{len(fns)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    sys.exit(_run_all())
