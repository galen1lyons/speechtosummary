"""
Tests for src/config.py

Covers validate() methods on all config dataclasses.
"""

import pytest

from src.config import ASRMetricsConfig, DiarizationConfig, PreprocessConfig, SummaryConfig, WhisperConfig


# ---------------------------------------------------------------------------
# WhisperConfig
# ---------------------------------------------------------------------------

class TestWhisperConfigValidate:
    def test_valid_defaults(self):
        WhisperConfig().validate()

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            WhisperConfig(backend="invalid").validate()

    def test_huggingface_model_with_faster_whisper_raises(self):
        with pytest.raises(ValueError, match="HuggingFace model"):
            WhisperConfig(
                backend="faster-whisper",
                model_name="mesolitica/malaysian-whisper-base",
            ).validate()

    def test_huggingface_model_with_openai_backend_valid(self):
        WhisperConfig(
            backend="openai-whisper",
            model_name="mesolitica/malaysian-whisper-base",
        ).validate()

    def test_invalid_model_name_raises(self):
        with pytest.raises(ValueError, match="Invalid model"):
            WhisperConfig(model_name="nonexistent").validate()

    def test_valid_model_names(self):
        for model in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]:
            WhisperConfig(model_name=model).validate()

    def test_beam_size_zero_raises(self):
        with pytest.raises(ValueError, match="beam_size"):
            WhisperConfig(beam_size=0).validate()

    def test_beam_size_one_valid(self):
        WhisperConfig(beam_size=1).validate()

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            WhisperConfig(temperature=-0.1).validate()

    def test_temperature_above_one_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            WhisperConfig(temperature=1.1).validate()

    def test_temperature_boundaries_valid(self):
        WhisperConfig(temperature=0.0).validate()
        WhisperConfig(temperature=1.0).validate()


# ---------------------------------------------------------------------------
# SummaryConfig
# ---------------------------------------------------------------------------

class TestSummaryConfigValidate:
    def test_valid_defaults(self):
        SummaryConfig().validate()

    def test_invalid_content_type_raises(self):
        with pytest.raises(ValueError, match="content_type"):
            SummaryConfig(content_type="lecture").validate()

    def test_valid_content_types(self):
        for ct in ["meeting", "interview", "podcast", "general"]:
            SummaryConfig(content_type=ct).validate()

    def test_max_summary_length_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_summary_length"):
            SummaryConfig(max_summary_length=10, min_summary_length=100).validate()

    def test_max_length_less_than_min_length_raises(self):
        with pytest.raises(ValueError, match="max_length"):
            SummaryConfig(max_length=10, min_length=100).validate()

    def test_equal_lengths_valid(self):
        SummaryConfig(max_summary_length=100, min_summary_length=100).validate()
        SummaryConfig(max_length=50, min_length=50).validate()


# ---------------------------------------------------------------------------
# ASRMetricsConfig
# ---------------------------------------------------------------------------

class TestASRMetricsConfigValidate:
    def test_valid_defaults(self):
        ASRMetricsConfig().validate()

    def test_both_disabled_raises(self):
        with pytest.raises(ValueError):
            ASRMetricsConfig(calculate_cer=False, calculate_wer=False).validate()

    def test_only_wer_enabled_valid(self):
        ASRMetricsConfig(calculate_cer=False, calculate_wer=True).validate()

    def test_only_cer_enabled_valid(self):
        ASRMetricsConfig(calculate_cer=True, calculate_wer=False).validate()


# ---------------------------------------------------------------------------
# DiarizationConfig
# ---------------------------------------------------------------------------

class TestDiarizationConfigValidate:
    def test_valid_defaults(self):
        DiarizationConfig().validate()

    def test_min_speakers_zero_raises(self):
        with pytest.raises(ValueError, match="min_speakers"):
            DiarizationConfig(min_speakers=0).validate()

    def test_max_speakers_zero_raises(self):
        with pytest.raises(ValueError, match="max_speakers"):
            DiarizationConfig(max_speakers=0).validate()

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="min_speakers"):
            DiarizationConfig(min_speakers=5, max_speakers=2).validate()

    def test_min_equals_max_valid(self):
        DiarizationConfig(min_speakers=2, max_speakers=2).validate()

    def test_valid_speaker_range(self):
        DiarizationConfig(min_speakers=1, max_speakers=10).validate()

    def test_none_speakers_valid(self):
        DiarizationConfig(min_speakers=None, max_speakers=None).validate()


# ---------------------------------------------------------------------------
# PreprocessConfig
# ---------------------------------------------------------------------------

class TestPreprocessConfigValidate:
    def test_valid_defaults(self):
        PreprocessConfig().validate()

    def test_enabled_false_valid(self):
        PreprocessConfig(enabled=False).validate()

    def test_target_peak_dbfs_above_zero_raises(self):
        with pytest.raises(ValueError, match="target_peak_dbfs"):
            PreprocessConfig(target_peak_dbfs=1.0).validate()

    def test_target_peak_dbfs_below_minus_sixty_raises(self):
        with pytest.raises(ValueError, match="target_peak_dbfs"):
            PreprocessConfig(target_peak_dbfs=-61.0).validate()

    def test_target_peak_dbfs_at_boundaries_valid(self):
        PreprocessConfig(target_peak_dbfs=0.0).validate()
        PreprocessConfig(target_peak_dbfs=-60.0).validate()

    def test_all_disabled_valid(self):
        PreprocessConfig(enabled=False, denoise=False, normalize_volume=False).validate()
