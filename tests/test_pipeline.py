"""
Integration tests for src/pipeline.py — transcribe-first architecture.

Tests verify the new pipeline order:
    Preprocess → Transcribe (full audio) → Diarize → Align/Merge → Summarize

All external dependencies (models, diarization, summarization) are mocked.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import DiarizationConfig, PreprocessConfig, SummaryConfig, WhisperConfig
from src.diarize import SpeakerSegment, TranscriptSegment
from src.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAKE_TRANSCRIPT_JSON = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 3.0, "text": "Hello world."},
        {"start": 3.0, "end": 6.0, "text": "How are you?"},
    ],
}

FAKE_TRANSCRIPTION_METRICS = {
    "model": "base",
    "audio_duration_s": 6.0,
    "processing_time_s": 1.2,
    "segments": 2,
    "mode": "full_audio",
}

FAKE_SPEAKER_SEGMENTS = [
    SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
    SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
]

FAKE_MERGED_SEGMENTS = [
    TranscriptSegment(start=0.0, end=3.0, text="Hello world.", speaker="Speaker 1"),
    TranscriptSegment(start=3.0, end=6.0, text="How are you?", speaker="Speaker 2"),
]


def _write_fake_transcript(run_dir: Path) -> tuple[Path, Path]:
    """Write a fake transcript JSON and TXT to run_dir, return (json_path, txt_path)."""
    out_base = run_dir / "transcript"
    json_path = out_base.with_suffix(".json")
    txt_path = out_base.with_suffix(".txt")
    json_path.write_text(json.dumps(FAKE_TRANSCRIPT_JSON), encoding="utf-8")
    txt_path.write_text("Hello world.\nHow are you?", encoding="utf-8")
    return json_path, txt_path


# ---------------------------------------------------------------------------
# Test 1: Diarization disabled — full-audio transcription only
# ---------------------------------------------------------------------------

class TestPipelineDiarizationDisabled:
    def test_no_speaker_files_written(self, tmp_path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake_audio")

        with (
            patch("src.pipeline.denoise_audio") as mock_denoise,
            patch("src.pipeline.transcribe_faster") as mock_transcribe,
            patch("src.pipeline.load_transcript") as mock_load_transcript,
            patch("src.pipeline.create_structured_summary") as mock_summary,
        ):
            preprocessed = tmp_path / "preprocessed.wav"
            preprocessed.write_bytes(b"preprocessed")
            mock_denoise.return_value = preprocessed

            def fake_transcribe(**kwargs):
                run_dir = kwargs["out_base"].parent
                return _write_fake_transcript(run_dir) + (FAKE_TRANSCRIPTION_METRICS,)

            mock_transcribe.side_effect = fake_transcribe
            mock_load_transcript.return_value = MagicMock(segments=[])
            mock_summary.return_value = "# Summary"

            result = run_pipeline(
                audio_path=audio,
                output_dir=tmp_path / "outputs",
                whisper_config=WhisperConfig(backend="faster-whisper", model_name="base"),
                summary_config=SummaryConfig(),
                diarization_config=DiarizationConfig(enabled=False),
                preprocess_config=PreprocessConfig(),
            )

        assert result["speaker_transcript_path"] is None
        assert result["speaker_stats"] is None
        assert result["json_path"] is not None
        assert result["txt_path"] is not None


# ---------------------------------------------------------------------------
# Test 2: Diarization enabled and succeeds — speakers.txt must be written
# ---------------------------------------------------------------------------

class TestPipelineDiarizationEnabledWithMerge:
    def test_speakers_txt_written(self, tmp_path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake_audio")

        with (
            patch("src.pipeline.denoise_audio") as mock_denoise,
            patch("src.pipeline.transcribe_faster") as mock_transcribe,
            patch("src.pipeline.diarize_audio") as mock_diarize,
            patch("src.pipeline.save_rttm"),
            patch("src.pipeline.merge_diarization_with_transcript") as mock_merge,
            patch("src.pipeline.format_transcript_with_speakers") as mock_format,
            patch("src.pipeline.get_speaker_statistics") as mock_stats,
            patch("src.pipeline.load_transcript") as mock_load_transcript,
            patch("src.pipeline.create_structured_summary") as mock_summary,
        ):
            preprocessed = tmp_path / "preprocessed.wav"
            preprocessed.write_bytes(b"preprocessed")
            mock_denoise.return_value = preprocessed

            def fake_transcribe(**kwargs):
                run_dir = kwargs["out_base"].parent
                return _write_fake_transcript(run_dir) + (FAKE_TRANSCRIPTION_METRICS,)

            mock_transcribe.side_effect = fake_transcribe
            mock_diarize.return_value = FAKE_SPEAKER_SEGMENTS
            mock_merge.return_value = FAKE_MERGED_SEGMENTS
            mock_format.return_value = "Speaker 1: Hello world.\nSpeaker 2: How are you?"
            mock_stats.return_value = {
                "Speaker 1": {"duration": 3.0, "segment_count": 1, "word_count": 2},
                "Speaker 2": {"duration": 3.0, "segment_count": 1, "word_count": 3},
            }
            mock_load_transcript.return_value = MagicMock(segments=[])
            mock_summary.return_value = "# Summary"

            result = run_pipeline(
                audio_path=audio,
                output_dir=tmp_path / "outputs",
                whisper_config=WhisperConfig(backend="faster-whisper", model_name="base"),
                summary_config=SummaryConfig(),
                diarization_config=DiarizationConfig(enabled=True),
                preprocess_config=PreprocessConfig(),
            )

        assert result["speaker_transcript_path"] is not None
        speakers_txt = Path(result["speaker_transcript_path"])
        assert speakers_txt.exists()
        assert "Speaker 1" in speakers_txt.read_text()

        # Verify merge was called with (diarization_segments, transcript_segments)
        mock_merge.assert_called_once()
        call_args = mock_merge.call_args[0]
        assert call_args[0] == FAKE_SPEAKER_SEGMENTS  # diarization_segments first


# ---------------------------------------------------------------------------
# Test 3: Diarization fails — pipeline continues without speaker attribution
# ---------------------------------------------------------------------------

class TestPipelineGracefulDegradation:
    def test_pipeline_continues_on_diarization_failure(self, tmp_path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake_audio")

        with (
            patch("src.pipeline.denoise_audio") as mock_denoise,
            patch("src.pipeline.transcribe_faster") as mock_transcribe,
            patch("src.pipeline.diarize_audio") as mock_diarize,
            patch("src.pipeline.load_transcript") as mock_load_transcript,
            patch("src.pipeline.create_structured_summary") as mock_summary,
        ):
            preprocessed = tmp_path / "preprocessed.wav"
            preprocessed.write_bytes(b"preprocessed")
            mock_denoise.return_value = preprocessed

            def fake_transcribe(**kwargs):
                run_dir = kwargs["out_base"].parent
                return _write_fake_transcript(run_dir) + (FAKE_TRANSCRIPTION_METRICS,)

            mock_transcribe.side_effect = fake_transcribe
            mock_diarize.side_effect = RuntimeError("pyannote model unavailable")
            mock_load_transcript.return_value = MagicMock(segments=[])
            mock_summary.return_value = "# Summary"

            # Should not raise even though diarization failed
            result = run_pipeline(
                audio_path=audio,
                output_dir=tmp_path / "outputs",
                whisper_config=WhisperConfig(backend="faster-whisper", model_name="base"),
                summary_config=SummaryConfig(),
                diarization_config=DiarizationConfig(enabled=True),
                preprocess_config=PreprocessConfig(),
            )

        # Pipeline completed: transcript present, no speaker files
        assert result["json_path"] is not None
        assert result["txt_path"] is not None
        assert result["speaker_transcript_path"] is None
        assert result["summary_path"] is not None
