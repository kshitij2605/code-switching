#!/usr/bin/env python3
"""Hand-made unit tests for the scoring harness.

Run directly:   python tests/test_score.py
Or with pytest: pytest tests/test_score.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import score  # noqa: E402

# A realistic code-switching utterance:
#   "tomorrow I'll merge the pull request"  rendered the way a bilingual JP dev
#   would actually say/write it.
REF = "明日 pull request を merge します"


def test_normalize_folds_fullwidth():
    # Full-width Latin + digits should fold to half-width via NFKC.
    assert score.normalize("ＡＢＣ１２３") == "abc123"
    # Casing normalized; ALL punctuation (incl. hyphens) -> space, applied
    # identically to ref & hyp, so "pull-request" becomes two tokens.
    assert score.normalize("Pull-Request, merge!") == "pull request merge"


def test_tokenizers():
    n = score.normalize(REF)
    assert score.english_words(n) == ["pull", "request", "merge"]
    assert score.japanese_chars(n) == ["明", "日", "を", "し", "ま", "す"]
    # Mixed stream: 1 token per JP char, 1 per English word.
    assert score.mixed_tokens(n) == ["明", "日", "pull", "request", "を", "merge", "し", "ま", "す"]


def test_perfect_hypothesis_is_zero_error():
    parts = score.score_pair(REF, REF)
    assert parts["mer"].rate == 0.0
    assert parts["ja"].rate == 0.0
    assert parts["en"].rate == 0.0
    assert parts["script"].en_ref == 3
    assert parts["script"].latin == 3
    assert parts["script"].accuracy == 1.0


def test_katakana_forcing_tanks_script_accuracy():
    # The classic cloud-ASR failure: English insertions forced into katakana.
    hyp = "明日 プルリクエスト を マージ します"
    parts = score.score_pair(REF, hyp)
    sc = parts["script"]
    assert sc.en_ref == 3
    assert sc.latin == 0
    assert sc.katakana == 3          # all three English words katakana-ized
    assert sc.accuracy == 0.0
    # English words vanished from the Latin stream -> 100% EN-WER (all deletions).
    assert parts["en"].rate == 1.0


def test_dropping_minority_language_is_penalized():
    # Model drops the English entirely.
    hyp = "明日 を します"
    parts = score.score_pair(REF, hyp)
    sc = parts["script"]
    assert sc.en_ref == 3
    assert sc.dropped == 3
    assert sc.accuracy == 0.0


def test_corpus_aggregation_and_row():
    records = [
        {"id": "a", "reference": REF, "hypothesis": REF},
        {"id": "b", "reference": REF, "hypothesis": "明日 プルリクエスト を マージ します"},
    ]
    mm = score.score_records(records, label="demo")
    row = mm.as_row()
    assert row["n"] == 2
    assert row["en_ref_words"] == 6          # 3 + 3
    assert row["script_breakdown"]["latin"] == 3
    assert row["script_breakdown"]["katakana"] == 3
    # Pooled script accuracy = 3 latin / 6 english ref words = 0.5
    assert row["ScriptAcc"] == 0.5


def test_partial_transliteration_counts_as_other_jp():
    # Reference English word partly translated to Japanese (katakana + kanji).
    hyp = "明日 ミーティング会議 を merge します"
    parts = score.score_pair(REF, hyp)
    sc = parts["script"]
    assert sc.en_ref == 3
    # "pull" and "request" become one Japanese token => other_jp, not latin.
    assert sc.other_jp >= 1


def test_romaji_false_positive_is_known_limitation():
    # A romaji phonetic rendering is made of Latin letters, so the current
    # token-only ScriptAcc counts it as "latin". This is a documented limitation.
    ref = "明日 meeting を merge します"
    hyp = "明日 miitingu を マージ します"
    parts = score.score_pair(ref, hyp)
    sc = parts["script"]
    assert sc.en_ref == 2
    assert sc.latin == 1          # miitingu (incorrectly counted as Latin)
    assert sc.katakana == 1       # マージ


def test_bootstrap_ci_structure():
    records = [
        {"id": "a", "reference": REF, "hypothesis": REF},
        {"id": "b", "reference": REF, "hypothesis": "明日 プルリクエスト を マージ します"},
    ]
    counts = score.per_utterance_counts(records)
    cis = score.bootstrap_ci(counts, n_samples=10, ci=95, rng=score.random.Random(0))
    for metric in ("MER", "JA_CER", "JA_WER", "EN_WER", "ScriptAcc"):
        assert metric in cis
        assert "lo" in cis[metric] and "hi" in cis[metric] and "mean" in cis[metric]


def test_paired_bootstrap_requires_same_length():
    a = score.per_utterance_counts([{"reference": REF, "hypothesis": REF}])
    b = score.per_utterance_counts([{"reference": REF, "hypothesis": REF},
                                     {"reference": REF, "hypothesis": REF}])
    try:
        score.paired_bootstrap(a, b, 10, 95, score.random.Random(0))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_switch_point_analysis_counts():
    counts = score.analyze_switch_points(REF, REF, REF)
    # The reference has 3 English words adjacent to Japanese chars -> switch points.
    assert counts["switch_type"]["intra"]["en_ref"] == 3
    # "pull request" is a consecutive English phrase, "merge" is a single word.
    assert counts["span_length"]["single"]["en_ref"] == 1
    assert counts["span_length"]["phrase"]["en_ref"] == 2
    # Perfect hypothesis: all correct; downstream errors are zero because there
    # are no katakana substitutions.
    assert counts["distance"]["0"]["tokens"] > 0
    assert counts["distance"]["0"]["correct"] == counts["distance"]["0"]["tokens"]


# ---------------------------------------------------------------------------
# v2.0 tests: normalisation, switch report, bootstrap
# ---------------------------------------------------------------------------


def test_iteration_mark_kept_and_middle_dot_separates():
    n = score.normalize("人々はジョン・スミスと話した")
    assert "々" in n                                   # 々 is a Japanese character
    assert score.japanese_chars(n)[:2] == ["人", "々"]
    assert "・" not in score.mixed_tokens(n)            # ・ is a separator, not a char


def test_apostrophe_kept_inside_english_words():
    n = score.normalize("Don’t merge it")
    assert n == "don't merge it"                        # one token, not don + t
    assert score.english_words(n) == ["don't", "merge", "it"]
    assert score.normalize("'quoted' word") == "quoted word"  # quotes still stripped


def test_accented_latin_letters_kept():
    n = score.normalize("Café のメニュー")
    assert score.english_words(n) == ["café"]           # accent kept, not "caf"


def test_strict_script_accuracy_exposes_romaji():
    ref = "明日 meeting を merge します"
    hyp = "明日 miitingu を merge します"
    sc = score.score_pair(ref, hyp)["script"]
    assert sc.accuracy == 1.0            # v1 ScriptAcc: any Latin token counts
    assert sc.strict_accuracy == 0.5     # v2: only the exact Latin rendering counts
    assert sc.latin_exact == 1 and sc.latin_other == 1


def test_katakana_keep_detects_over_romanisation():
    ref = "エジプトのピラミッドを見た"
    kept = score.score_pair(ref, ref)["kata"]
    assert kept.runs == 2 and kept.kept == 2 and kept.keep_rate == 1.0
    romanised = score.score_pair(ref, "egypt の pyramid を見た")["kata"]
    assert romanised.runs == 2 and romanised.latinized == 2
    assert romanised.keep_rate == 0.0


def test_switch_report_handles_monolingual_utterances():
    # An utterance without an EN<->JP switch used to raise KeyError('>=0').
    report = score.aggregate_switch_analysis([
        {"reference": "今日は晴れです", "hypothesis": "今日は晴れです"},
        {"reference": REF, "hypothesis": REF},
    ])
    assert report["token_accuracy_by_distance"]["none"] == 1.0


def test_switch_distance_uses_reference_positions():
    # Insertions before the switch shift alignment-op positions; distances must
    # be measured in reference positions ('merge' is ref token 2 = the switch).
    ref = "明日 merge"
    counts = score.analyze_switch_points(ref, "明日 あいう merge", ref)
    assert counts["en_distance"]["0"]["tokens"] == 1
    assert counts["en_distance"]["0"]["correct"] == 1
    assert counts["en_distance"][">=3"]["tokens"] == 0


def test_inter_sentential_switch_type():
    ref = "今日は会議です。Let's start. では始めます。"
    counts = score.analyze_switch_points(ref, ref, ref)
    assert counts["switch_type"]["inter"]["en_ref"] == 2
    assert counts["switch_type"]["intra"]["en_ref"] == 0


def test_paired_bootstrap_aligns_by_id():
    a = [{"id": "1", "reference": REF, "hypothesis": REF},
         {"id": "2", "reference": REF, "hypothesis": "明日 プルリクエスト を マージ します"}]
    b = list(reversed(a))  # identical predictions, different file order
    ra, rb = score.align_records_by_id(a, b)
    diff = score.paired_bootstrap(score.per_utterance_counts(ra), score.per_utterance_counts(rb),
                                  50, 95, score.random.Random(0))
    assert diff["MER"]["lo"] == 0.0 and diff["MER"]["hi"] == 0.0
    assert diff["MER"]["p_boot"] == 1.0
    # Unaligned inputs are rejected instead of being silently paired by position.
    try:
        score.paired_bootstrap(score.per_utterance_counts(a), score.per_utterance_counts(b),
                               10, 95, score.random.Random(0))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_align_records_by_id_rejects_mismatched_sets():
    a = [{"id": "1", "reference": REF, "hypothesis": REF}]
    b = [{"id": "2", "reference": REF, "hypothesis": REF}]
    try:
        score.align_records_by_id(a, b)
        assert False, "expected ValueError"
    except ValueError:
        pass
    ra, rb = score.align_records_by_id(a + b, b, allow_mismatch=True)
    assert [r["id"] for r in ra] == ["2"] and [r["id"] for r in rb] == ["2"]


def test_bootstrap_ignores_nan_resamples():
    # No English words anywhere -> ScriptAcc is undefined in every resample.
    recs = [{"id": "1", "reference": "今日は晴れ", "hypothesis": "今日は晴れ"},
            {"id": "2", "reference": "明日は雨", "hypothesis": "明日は雨"}]
    cis = score.bootstrap_ci(score.per_utterance_counts(recs), 20, 95, score.random.Random(0))
    assert cis["ScriptAcc"]["mean"] is None and cis["ScriptAcc"]["n_valid"] == 0
    assert cis["MER"]["n_valid"] == 20


def test_cluster_bootstrap_resamples_speakers():
    recs = [{"id": str(i), "speaker": f"s{i % 2}", "reference": REF,
             "hypothesis": REF if i % 2 else "明日 を します"} for i in range(6)]
    counts = score.per_utterance_counts(recs, cluster_key="speaker")
    assert {c["cluster"] for c in counts} == {"s0", "s1"}
    cis = score.bootstrap_ci(counts, 200, 95, score.random.Random(0), clustered=True)
    assert 0.0 <= cis["ScriptAcc"]["lo"] <= cis["ScriptAcc"]["hi"] <= 1.0


def test_small_denominator_is_flagged():
    mm = score.score_records([{"id": "a", "reference": REF, "hypothesis": REF}], "tiny")
    row = mm.as_row()
    assert row["flags"]["en_small_denominator"] is True
    assert row["scorer_version"] == score.SCORER_VERSION
    assert "n/a" in score.render_table([mm])


_LEX = {"ミーティング": ["meeting"], "パン": ["pan"], "ルーム": ["room"],
        "ミーティングルーム": ["meeting room"]}


def _lenient(ref, hyp):
    score.set_loanwords(_LEX)
    try:
        parts = score.score_pair(ref, hyp)
    finally:
        score.set_loanwords(None)
    return parts["mer"], parts["mer_len"]


def test_lenient_mer_accepts_transliteration_both_ways():
    strict, len_ = _lenient("今日のmeetingは長い", "今日のミーティングは長い")
    assert strict.errors > 0 and len_.errors == 0
    strict, len_ = _lenient("今日のミーティングは長い", "今日のmeetingは長い")
    assert strict.errors > 0 and len_.errors == 0
    # same denominator as strict MER: the reference is never rewritten
    assert len_.ref_len == strict.ref_len


def test_lenient_mer_handles_compounds_and_multiword():
    _, len_ = _lenient("meeting roomを予約", "ミーティングルームを予約")
    assert len_.errors == 0
    _, len_ = _lenient("ミーティングルームを予約", "meeting roomを予約")
    assert len_.errors == 0


def test_lenient_mer_never_accepts_translations():
    # "bread" is a translation of パン (from Portuguese pão), not a transliteration
    strict, len_ = _lenient("パンを食べた", "breadを食べた")
    assert len_.errors == strict.errors > 0
    # a katakana word whose counterpart is absent from the reference stays as it is
    strict, len_ = _lenient("今日は長い", "今日はミーティング長い")
    assert len_.errors == strict.errors


def test_lenient_mer_is_off_without_a_lexicon():
    assert "mer_len" not in score.score_pair(REF, REF)
    mm = score.score_records([{"id": "a", "reference": REF, "hypothesis": REF}], "m")
    assert mm.as_row()["MER_lenient"] is None


def test_lenient_mer_in_bootstrap():
    import random
    score.set_loanwords(_LEX)
    try:
        recs = [{"id": str(i), "reference": "今日のmeetingは長い",
                 "hypothesis": "今日のミーティングは長い"} for i in range(5)]
        counts = score.per_utterance_counts(recs)
    finally:
        score.set_loanwords(None)
    ci = score.bootstrap_ci(counts, 50, 95, random.Random(0))
    assert ci["MER_lenient"]["mean"] == 0.0 and ci["MER"]["mean"] > 0


def test_capped_mer_bounds_a_runaway_utterance():
    loop = "それでもあるかもしれません。" * 30
    recs = [{"id": "a", "reference": "今日は雨", "hypothesis": "今日は雨"},
            {"id": "b", "reference": "明日は晴れ", "hypothesis": loop}]
    row = score.score_records(recs, "m").as_row()
    # the loop's insertions dominate pooled MER; capped, it costs at most its own 5 tokens
    assert row["MER"] > 10
    assert row["MER_capped"] == round(5 / 9, 4)
    assert row["Runaway"] == 0.5 and row["runaway_utts"] == 1


def test_capped_mer_equals_mer_without_runaways():
    recs = [{"id": "a", "reference": REF, "hypothesis": REF},
            {"id": "b", "reference": "今日のmeetingは長い", "hypothesis": "今日のミーティングは長い"}]
    row = score.score_records(recs, "m").as_row()
    counts = score.per_utterance_counts(recs)
    # each utterance's errors stay below its reference length, so nothing is capped
    assert all(c["mer"]["err"] <= c["mer"]["ref"] for c in counts)
    assert row["MER_capped"] == row["MER"] and row["Runaway"] == 0.0


def test_capped_mer_and_runaway_in_bootstrap_and_pooling():
    import random
    recs = [{"id": str(i), "reference": "明日は晴れ",
             "hypothesis": "明日は晴れ" if i else "はれ" * 40} for i in range(4)]
    counts = score.per_utterance_counts(recs)
    rates = score._aggregate_rates(counts)
    assert rates["Runaway"] == 0.25
    assert rates["MER_capped"] == 5 / 20 < rates["MER"]
    ci = score.bootstrap_ci(counts, 50, 95, random.Random(0))
    assert 0.0 <= ci["MER_capped"]["lo"] <= ci["MER_capped"]["hi"] <= 1.0


def test_cer_yv_matches_ye_vickers_definition():
    # lowercase, punctuation and whitespace removed, then characters; English counts per letter
    assert score.yv_chars("今日の Meeting は、長い。") == list("今日のmeetingは長い")
    p = score.score_pair("今日のmeetingは長い", "今日のミーティングは長い")
    c = p["cer_yv"]
    assert c.ref_len == len("今日のmeetingは長い")
    assert c.errors == 7          # "meeting" (7 letters) vs ミーティング (6 kana): 6 sub + 1 del
    row = score.score_records([{"id": "a", "reference": "Hello, world!", "hypothesis": "hello world"}], "m").as_row()
    assert row["CER_yv"] == 0.0


def test_cer_yv_in_bootstrap():
    import random
    recs = [{"id": str(i), "reference": "今日は雨", "hypothesis": "今日は雪"} for i in range(4)]
    ci = score.bootstrap_ci(score.per_utterance_counts(recs), 50, 95, random.Random(0))
    assert ci["CER_yv"]["mean"] == 0.25


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
