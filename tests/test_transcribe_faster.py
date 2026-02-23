"""
Tests for the new functions added to src/transcribe_faster.py:
  - load_faster_whisper_model
  - transcribe_segments_faster
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.exceptions import ModelLoadError, TranscriptionError


# ---------------------------------------------------------------------------
# load_faster_whisper_model
# ---------------------------------------------------------------------------

class TestLoadFasterWhisperModel:
    def test_model_load_failure_raises_model_load_error(self):
        """Any exception from WhisperModel constructor should become ModelLoadError."""
        with patch("src.transcribe_faster.WhisperModel", side_effect=RuntimeError("CUDA OOM")):
            from src.transcribe_faster import load_faster_whisper_model
            with pytest.raises(ModelLoadError, match="Failed to load faster-whisper model"):
                load_faster_whisper_model("base", "cpu", "int8")

    def test_returns_model_on_success(self):
        mock_model = MagicMock()
        with patch("src.transcribe_faster.WhisperModel", return_value=mock_model):
            from src.transcribe_faster import load_faster_whisper_model
            result = load_faster_whisper_model("base", "cpu", "int8")
        assert result is mock_model


# ---------------------------------------------------------------------------
# transcribe_segments_faster
# ---------------------------------------------------------------------------

class TestTranscribeSegmentsFaster:
    def _make_mock_segment(self, start, end, text, avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=1.5):
        seg = MagicMock()
        seg.start = start
        seg.end = end
        seg.text = f" {text}"
        seg.avg_logprob = avg_logprob
        seg.no_speech_prob = no_speech_prob
        seg.compression_ratio = compression_ratio
        return seg

    def test_empty_segment_list_returns_empty(self):
        model = MagicMock()
        from src.transcribe_faster import transcribe_segments_faster
        result = transcribe_segments_faster(model, [])
        assert result == []
        model.transcribe.assert_not_called()

    def test_timestamp_offset_applied_correctly(self, tmp_path):
        """Sub-segment timestamps must be offset by original_start."""
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"fake")

        mock_seg = self._make_mock_segment(start=1.0, end=3.0, text="hello world")
        mock_info = MagicMock()
        model = MagicMock()
        model.transcribe.return_value = ([mock_seg], mock_info)

        from src.transcribe_faster import transcribe_segments_faster
        # original_start=10.0 means absolute start should be 10.0 + 1.0 = 11.0
        segment_clips = [(clip, 10.0, 15.0, "SPEAKER_00")]
        results = transcribe_segments_faster(model, segment_clips, language="en")

        assert len(results) == 1
        assert results[0]["start"] == pytest.approx(11.0)
        assert results[0]["end"] == pytest.approx(13.0)
        assert results[0]["speaker"] == "SPEAKER_00"

    def test_empty_speech_segment_skipped_gracefully(self, tmp_path):
        """If Whisper returns no sub-segments for a clip, skip it without raising."""
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"fake")

        mock_info = MagicMock()
        model = MagicMock()
        model.transcribe.return_value = ([], mock_info)

        from src.transcribe_faster import transcribe_segments_faster
        segment_clips = [(clip, 5.0, 10.0, "SPEAKER_01")]
        results = transcribe_segments_faster(model, segment_clips, language="en")

        assert results == []

    def test_transcription_error_propagates(self, tmp_path):
        """Exception from model.transcribe should become TranscriptionError."""
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"fake")

        model = MagicMock()
        model.transcribe.side_effect = RuntimeError("GPU exploded")

        from src.transcribe_faster import transcribe_segments_faster
        segment_clips = [(clip, 0.0, 5.0, "SPEAKER_00")]
        with pytest.raises(TranscriptionError):
            transcribe_segments_faster(model, segment_clips, language="en")

    def test_results_sorted_by_start_time(self, tmp_path):
        """Results should be sorted by start time even if segments arrive out of order."""
        clip_a = tmp_path / "clip_a.wav"
        clip_b = tmp_path / "clip_b.wav"
        clip_a.write_bytes(b"fake")
        clip_b.write_bytes(b"fake")

        seg_a = self._make_mock_segment(start=0.0, end=2.0, text="first")
        seg_b = self._make_mock_segment(start=0.0, end=2.0, text="second")
        mock_info = MagicMock()

        model = MagicMock()
        # clip_b has earlier original_start; results should still sort correctly
        model.transcribe.side_effect = [([seg_b], mock_info), ([seg_a], mock_info)]

        from src.transcribe_faster import transcribe_segments_faster
        # clip_b original_start=5.0, clip_a original_start=10.0
        segment_clips = [
            (clip_b, 5.0, 7.0, "SPEAKER_01"),
            (clip_a, 10.0, 12.0, "SPEAKER_00"),
        ]
        results = transcribe_segments_faster(model, segment_clips, language="en")

        starts = [r["start"] for r in results]
        assert starts == sorted(starts), "Results must be sorted by start time"
