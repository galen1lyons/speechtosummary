"""
Tests for src/evaluation/asr_metrics.py

Covers WER, CER, RTF calculations and the ASRMetrics container.
"""

import pytest

from src.evaluation.asr_metrics import (
    ASRMetrics,
    RTFTimer,
    calculate_cer,
    calculate_rtf,
    calculate_wer,
    evaluate_transcription,
    levenshtein_distance,
)
from src.exceptions import ASRMetricsError


# ---------------------------------------------------------------------------
# levenshtein_distance
# ---------------------------------------------------------------------------

class TestLevenshteinDistance:
    def test_identical_sequences(self):
        errors, subs, dels, ins = levenshtein_distance(["a", "b"], ["a", "b"])
        assert errors == 0
        assert subs == 0 and dels == 0 and ins == 0

    def test_all_substitutions(self):
        errors, subs, dels, ins = levenshtein_distance(["a", "b"], ["x", "y"])
        assert subs == 2
        assert dels == 0 and ins == 0

    def test_all_deletions(self):
        errors, subs, dels, ins = levenshtein_distance(["a", "b", "c"], [])
        assert dels == 3
        assert subs == 0 and ins == 0

    def test_all_insertions(self):
        errors, subs, dels, ins = levenshtein_distance([], ["a", "b"])
        assert ins == 2
        assert subs == 0 and dels == 0

    def test_mixed_operations(self):
        # "hello world" → "hello" = 1 deletion
        errors, subs, dels, ins = levenshtein_distance(
            ["hello", "world"], ["hello"]
        )
        assert errors == 1
        assert dels == 1

    def test_empty_sequences(self):
        errors, subs, dels, ins = levenshtein_distance([], [])
        assert errors == 0

    def test_single_element_match(self):
        errors, *_ = levenshtein_distance(["word"], ["word"])
        assert errors == 0

    def test_single_element_mismatch(self):
        errors, subs, dels, ins = levenshtein_distance(["word"], ["other"])
        assert errors == 1
        assert subs == 1


# ---------------------------------------------------------------------------
# calculate_wer
# ---------------------------------------------------------------------------

class TestCalculateWER:
    def test_perfect_match(self):
        wer, errors, total, *_ = calculate_wer("hello world", "hello world")
        assert wer == pytest.approx(0.0)
        assert errors == 0
        assert total == 2

    def test_complete_mismatch(self):
        wer, errors, total, *_ = calculate_wer("hello world", "foo bar")
        assert wer == pytest.approx(1.0)

    def test_one_word_wrong(self):
        wer, errors, total, *_ = calculate_wer("hello world", "hello earth")
        assert wer == pytest.approx(0.5)
        assert errors == 1
        assert total == 2

    def test_empty_reference_raises(self):
        with pytest.raises(ASRMetricsError, match="empty"):
            calculate_wer("", "hello")

    def test_normalization_case_insensitive(self):
        wer, errors, *_ = calculate_wer("Hello World", "hello world", normalize=True)
        assert errors == 0

    def test_normalization_strips_punctuation(self):
        wer, errors, *_ = calculate_wer(
            "Hello, world!", "hello world", normalize=True, remove_punctuation=True
        )
        assert errors == 0

    def test_no_normalization_case_sensitive(self):
        wer, errors, *_ = calculate_wer(
            "Hello World", "hello world", normalize=False
        )
        assert errors == 2

    def test_returns_six_values(self):
        result = calculate_wer("a b c", "a b c")
        assert len(result) == 6


# ---------------------------------------------------------------------------
# calculate_cer
# ---------------------------------------------------------------------------

class TestCalculateCER:
    def test_perfect_match(self):
        cer, errors, total, *_ = calculate_cer("hello", "hello")
        assert cer == pytest.approx(0.0)
        assert errors == 0

    def test_one_char_wrong(self):
        cer, errors, total, *_ = calculate_cer("hello", "helo", normalize=False)
        # "hello" (5 chars) vs "helo" (4 chars) = 1 deletion
        assert errors == 1
        assert total == 5

    def test_empty_reference_raises(self):
        with pytest.raises(ASRMetricsError, match="empty"):
            calculate_cer("", "hello")

    def test_spaces_excluded_from_char_count(self):
        # "a b" has 2 non-space chars; "a b" → 2 chars compared
        cer, errors, total, *_ = calculate_cer("a b", "a b")
        assert total == 2

    def test_normalization_lowercases(self):
        cer, errors, *_ = calculate_cer("ABC", "abc", normalize=True)
        assert errors == 0

    def test_returns_six_values(self):
        result = calculate_cer("hello", "hello")
        assert len(result) == 6


# ---------------------------------------------------------------------------
# calculate_rtf
# ---------------------------------------------------------------------------

class TestCalculateRTF:
    def test_faster_than_realtime(self):
        rtf = calculate_rtf(audio_duration=60.0, processing_time=10.0)
        assert rtf == pytest.approx(10.0 / 60.0)

    def test_slower_than_realtime(self):
        rtf = calculate_rtf(audio_duration=10.0, processing_time=60.0)
        assert rtf == pytest.approx(6.0)

    def test_realtime_factor_one(self):
        rtf = calculate_rtf(audio_duration=30.0, processing_time=30.0)
        assert rtf == pytest.approx(1.0)

    def test_zero_audio_duration_raises(self):
        with pytest.raises(ASRMetricsError):
            calculate_rtf(audio_duration=0.0, processing_time=10.0)

    def test_negative_audio_duration_raises(self):
        with pytest.raises(ASRMetricsError):
            calculate_rtf(audio_duration=-5.0, processing_time=10.0)


# ---------------------------------------------------------------------------
# ASRMetrics
# ---------------------------------------------------------------------------

class TestASRMetrics:
    def test_to_dict_structure(self):
        m = ASRMetrics(wer=0.1, cer=0.05, rtf=0.5)
        d = m.to_dict()
        assert "wer" in d and "cer" in d and "rtf" in d
        assert d["wer"]["rate"] == pytest.approx(0.1)
        assert d["cer"]["rate"] == pytest.approx(0.05)
        assert d["rtf"]["factor"] == pytest.approx(0.5)

    def test_str_shows_wer(self):
        m = ASRMetrics(
            wer=0.15,
            word_errors=3,
            word_total=20,
            word_substitutions=1,
            word_deletions=1,
            word_insertions=1,
        )
        s = str(m)
        assert "15.00%" in s

    def test_str_shows_cer(self):
        m = ASRMetrics(
            cer=0.05,
            char_errors=2,
            char_total=40,
            char_substitutions=1,
            char_deletions=1,
            char_insertions=0,
        )
        s = str(m)
        assert "5.00%" in s

    def test_str_shows_rtf(self):
        m = ASRMetrics(rtf=0.25, audio_duration=60.0, processing_time=15.0)
        s = str(m)
        assert "0.25" in s

    def test_none_fields_omitted_from_str(self):
        m = ASRMetrics()
        s = str(m)
        assert "WER" not in s
        assert "CER" not in s
        assert "RTF" not in s

    def test_defaults_are_none(self):
        m = ASRMetrics()
        assert m.wer is None
        assert m.cer is None
        assert m.rtf is None


# ---------------------------------------------------------------------------
# evaluate_transcription
# ---------------------------------------------------------------------------

class TestEvaluateTranscription:
    def test_returns_asr_metrics_instance(self):
        result = evaluate_transcription("hello world", "hello world")
        assert isinstance(result, ASRMetrics)

    def test_perfect_transcript_zero_wer(self):
        result = evaluate_transcription("hello world", "hello world")
        assert result.wer == pytest.approx(0.0)

    def test_calculates_cer_by_default(self):
        result = evaluate_transcription("hello", "hello")
        assert result.cer is not None

    def test_rtf_calculated_when_durations_provided(self):
        result = evaluate_transcription(
            "hello", "hello", audio_duration=10.0, processing_time=2.0
        )
        assert result.rtf == pytest.approx(0.2)

    def test_rtf_none_when_durations_missing(self):
        result = evaluate_transcription("hello", "hello")
        assert result.rtf is None

    def test_wer_can_be_disabled(self):
        result = evaluate_transcription(
            "hello world", "hello world", calculate_wer_metric=False
        )
        assert result.wer is None
        assert result.cer is not None

    def test_cer_can_be_disabled(self):
        result = evaluate_transcription(
            "hello world", "hello world", calculate_cer_metric=False
        )
        assert result.cer is None
        assert result.wer is not None


# ---------------------------------------------------------------------------
# RTFTimer
# ---------------------------------------------------------------------------

class TestRTFTimer:
    def test_measures_processing_time(self):
        timer = RTFTimer(audio_duration=60.0)
        with timer:
            pass  # near-instant
        assert timer.processing_time is not None
        assert timer.processing_time >= 0.0

    def test_computes_rtf(self):
        timer = RTFTimer(audio_duration=60.0)
        with timer:
            pass
        assert timer.rtf is not None
        assert timer.rtf >= 0.0
