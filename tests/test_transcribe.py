"""
Tests for the new functions added to src/transcribe.py:
  - load_openai_whisper_model
  - transcribe_segments
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.exceptions import ModelLoadError, TranscriptionError


# ---------------------------------------------------------------------------
# load_openai_whisper_model
# ---------------------------------------------------------------------------

class TestLoadOpenAIWhisperModel:
    def test_hf_model_detected_by_slash(self):
        """Model names containing '/' should use the HuggingFace pipeline path."""
        # transformers uses lazy loading so patch("transformers.pipeline") does not intercept
        # the local `from transformers import pipeline as hf_pipeline` inside the function.
        # We test routing behaviour only: model name with '/' → _type == "hf".
        with patch("src.transcribe.parse_device", return_value="cpu"), \
             patch("src.transcribe.whisper"):
            from src.transcribe import load_openai_whisper_model
            result = load_openai_whisper_model("mesolitica/malaysian-whisper-base", "cpu")

        assert result["_type"] == "hf"
        assert "_pipe" in result

    def test_standard_model_uses_whisper_load_model(self):
        """Standard model names (no '/') should use whisper.load_model."""
        mock_model = MagicMock()
        with patch("src.transcribe.parse_device", return_value="cpu"), \
             patch("src.transcribe.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = mock_model

            from src.transcribe import load_openai_whisper_model
            result = load_openai_whisper_model("base", "cpu")

        assert result["_type"] == "whisper"
        assert result["_model"] is mock_model

    def test_model_load_failure_raises_model_load_error(self):
        with patch("src.transcribe.parse_device", return_value="cpu"), \
             patch("src.transcribe.whisper") as mock_whisper:
            mock_whisper.load_model.side_effect = RuntimeError("model not found")

            from src.transcribe import load_openai_whisper_model
            with pytest.raises(ModelLoadError):
                load_openai_whisper_model("base", "cpu")


# ---------------------------------------------------------------------------
# transcribe_segments
# ---------------------------------------------------------------------------

class TestTranscribeSegments:
    def test_empty_segment_list_returns_empty(self):
        model = {"_type": "whisper", "_model": MagicMock()}
        from src.transcribe import transcribe_segments
        result = transcribe_segments(model, "base", [])
        assert result == []

    def test_timestamp_offset_applied_correctly(self, tmp_path):
        """Sub-segment timestamps must be offset by original_start."""
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.5, "end": 2.5, "text": "hello"}],
            "language": "en",
        }
        model = {"_type": "whisper", "_model": mock_model}

        from src.transcribe import transcribe_segments
        # original_start=20.0 means absolute start should be 20.0 + 0.5 = 20.5
        segment_clips = [(clip, 20.0, 25.0, "SPEAKER_00")]
        results = transcribe_segments(model, "base", segment_clips, language="en")

        assert len(results) == 1
        assert results[0]["start"] == pytest.approx(20.5)
        assert results[0]["end"] == pytest.approx(22.5)
        assert results[0]["speaker"] == "SPEAKER_00"

    def test_empty_speech_segment_skipped_gracefully(self, tmp_path):
        """If model returns no segments for a clip, skip without raising."""
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": [], "language": "en"}
        model = {"_type": "whisper", "_model": mock_model}

        from src.transcribe import transcribe_segments
        segment_clips = [(clip, 5.0, 10.0, "SPEAKER_01")]
        results = transcribe_segments(model, "base", segment_clips, language="en")
        assert results == []

    def test_results_sorted_by_start_time(self, tmp_path):
        clip_a = tmp_path / "clip_a.wav"
        clip_b = tmp_path / "clip_b.wav"
        clip_a.write_bytes(b"fake")
        clip_b.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            {"segments": [{"start": 0.0, "end": 2.0, "text": "second"}], "language": "en"},
            {"segments": [{"start": 0.0, "end": 2.0, "text": "first"}], "language": "en"},
        ]
        model = {"_type": "whisper", "_model": mock_model}

        from src.transcribe import transcribe_segments
        segment_clips = [
            (clip_b, 10.0, 12.0, "SPEAKER_01"),
            (clip_a, 5.0, 7.0, "SPEAKER_00"),
        ]
        results = transcribe_segments(model, "base", segment_clips, language="en")

        starts = [r["start"] for r in results]
        assert starts == sorted(starts), "Results must be sorted by start time"
