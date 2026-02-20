"""
Tests for src/diarize.py

Covers all pure-logic functions and error-path testing for diarize_audio.
Functions requiring pyannote (load_rttm, evaluate_diarization) are skipped
when the library is not installed.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.diarize import (
    DiarizationMetrics,
    SpeakerSegment,
    TranscriptSegment,
    format_transcript_with_speakers,
    get_speaker_statistics,
    merge_diarization_with_transcript,
    save_rttm,
)
from src.exceptions import AudioFileError, DiarizationError


# ---------------------------------------------------------------------------
# SpeakerSegment
# ---------------------------------------------------------------------------

class TestSpeakerSegment:
    def test_repr(self):
        seg = SpeakerSegment(start=0.0, end=5.5, speaker="Speaker 1")
        assert "Speaker 1" in repr(seg)
        assert "0.00s" in repr(seg)
        assert "5.50s" in repr(seg)

    def test_fields(self):
        seg = SpeakerSegment(start=1.0, end=3.0, speaker="Speaker 2")
        assert seg.start == 1.0
        assert seg.end == 3.0
        assert seg.speaker == "Speaker 2"


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------

class TestTranscriptSegment:
    def test_repr_with_speaker(self):
        seg = TranscriptSegment(start=0.0, end=3.0, text="Hello", speaker="Speaker 1")
        assert "Speaker 1" in repr(seg)

    def test_repr_without_speaker(self):
        seg = TranscriptSegment(start=0.0, end=3.0, text="Hello")
        assert "Speaker" not in repr(seg)

    def test_default_speaker_is_none(self):
        seg = TranscriptSegment(start=0.0, end=1.0, text="Hi")
        assert seg.speaker is None


# ---------------------------------------------------------------------------
# merge_diarization_with_transcript
# ---------------------------------------------------------------------------

class TestMergeDiarizationWithTranscript:
    def _make_diar(self, start, end, speaker):
        return SpeakerSegment(start=start, end=end, speaker=speaker)

    def _make_trans(self, start, end, text):
        return {"start": start, "end": end, "text": text}

    def test_single_segment_exact_match(self):
        diar = [self._make_diar(0.0, 5.0, "Speaker 1")]
        trans = [self._make_trans(0.0, 5.0, "Hello world")]
        result = merge_diarization_with_transcript(diar, trans)
        assert len(result) == 1
        assert result[0].speaker == "Speaker 1"
        assert result[0].text == "Hello world"

    def test_no_overlap_gives_no_speaker(self):
        diar = [self._make_diar(10.0, 15.0, "Speaker 1")]
        trans = [self._make_trans(0.0, 5.0, "Hello")]
        result = merge_diarization_with_transcript(diar, trans)
        assert result[0].speaker is None

    def test_picks_speaker_with_most_overlap(self):
        # Speaker 1 covers 0-3, Speaker 2 covers 3-5
        # Transcript segment 0-4 overlaps 3s with Spk1, 1s with Spk2
        diar = [
            self._make_diar(0.0, 3.0, "Speaker 1"),
            self._make_diar(3.0, 5.0, "Speaker 2"),
        ]
        trans = [self._make_trans(0.0, 4.0, "Some text")]
        result = merge_diarization_with_transcript(diar, trans)
        assert result[0].speaker == "Speaker 1"

    def test_multiple_transcript_segments(self):
        diar = [
            self._make_diar(0.0, 5.0, "Speaker 1"),
            self._make_diar(5.0, 10.0, "Speaker 2"),
        ]
        trans = [
            self._make_trans(0.0, 5.0, "First sentence"),
            self._make_trans(5.0, 10.0, "Second sentence"),
        ]
        result = merge_diarization_with_transcript(diar, trans)
        assert result[0].speaker == "Speaker 1"
        assert result[1].speaker == "Speaker 2"

    def test_empty_inputs(self):
        result = merge_diarization_with_transcript([], [])
        assert result == []

    def test_empty_diarization(self):
        trans = [self._make_trans(0.0, 5.0, "Hello")]
        result = merge_diarization_with_transcript([], trans)
        assert result[0].speaker is None

    def test_preserves_timestamps_and_text(self):
        diar = [self._make_diar(0.0, 5.0, "Speaker 1")]
        trans = [self._make_trans(1.0, 4.0, "Test text")]
        result = merge_diarization_with_transcript(diar, trans)
        assert result[0].start == 1.0
        assert result[0].end == 4.0
        assert result[0].text == "Test text"


# ---------------------------------------------------------------------------
# format_transcript_with_speakers
# ---------------------------------------------------------------------------

class TestFormatTranscriptWithSpeakers:
    def _seg(self, start, end, text, speaker=None):
        return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)

    def test_basic_format_with_timestamps(self):
        segs = [self._seg(0.0, 5.0, "Hello", "Speaker 1")]
        output = format_transcript_with_speakers(segs)
        assert "[Speaker 1, 0:00-0:05]: Hello" == output

    def test_format_without_timestamps(self):
        segs = [self._seg(0.0, 5.0, "Hello", "Speaker 1")]
        output = format_transcript_with_speakers(segs, include_timestamps=False)
        assert output == "[Speaker 1]: Hello"

    def test_unknown_speaker_when_none(self):
        segs = [self._seg(0.0, 5.0, "Hello")]
        output = format_transcript_with_speakers(segs)
        assert "[Unknown" in output

    def test_multiple_segments_joined_by_newline(self):
        segs = [
            self._seg(0.0, 5.0, "Hello", "Speaker 1"),
            self._seg(5.0, 10.0, "World", "Speaker 2"),
        ]
        output = format_transcript_with_speakers(segs)
        lines = output.split("\n")
        assert len(lines) == 2
        assert "Speaker 1" in lines[0]
        assert "Speaker 2" in lines[1]

    def test_timestamp_formatting_minutes(self):
        segs = [self._seg(65.0, 125.0, "Text", "Speaker 1")]
        output = format_transcript_with_speakers(segs)
        assert "1:05" in output  # 65s = 1m 5s
        assert "2:05" in output  # 125s = 2m 5s

    def test_empty_segments(self):
        assert format_transcript_with_speakers([]) == ""


# ---------------------------------------------------------------------------
# get_speaker_statistics
# ---------------------------------------------------------------------------

class TestGetSpeakerStatistics:
    def _seg(self, start, end, text, speaker=None):
        return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)

    def test_single_speaker(self):
        segs = [self._seg(0.0, 5.0, "Hello world", "Speaker 1")]
        stats = get_speaker_statistics(segs)
        assert "Speaker 1" in stats
        assert stats["Speaker 1"]["duration"] == pytest.approx(5.0)
        assert stats["Speaker 1"]["segment_count"] == 1
        assert stats["Speaker 1"]["word_count"] == 2

    def test_multiple_speakers(self):
        segs = [
            self._seg(0.0, 5.0, "Hello world", "Speaker 1"),
            self._seg(5.0, 8.0, "Goodbye", "Speaker 2"),
            self._seg(8.0, 12.0, "More text here", "Speaker 1"),
        ]
        stats = get_speaker_statistics(segs)
        assert stats["Speaker 1"]["segment_count"] == 2
        assert stats["Speaker 1"]["duration"] == pytest.approx(9.0)
        assert stats["Speaker 1"]["word_count"] == 5
        assert stats["Speaker 2"]["segment_count"] == 1
        assert stats["Speaker 2"]["duration"] == pytest.approx(3.0)

    def test_none_speaker_becomes_unknown(self):
        segs = [self._seg(0.0, 5.0, "Test")]
        stats = get_speaker_statistics(segs)
        assert "Unknown" in stats

    def test_empty_segments(self):
        stats = get_speaker_statistics([])
        assert stats == {}

    def test_word_count_accuracy(self):
        segs = [self._seg(0.0, 1.0, "one two three four five", "Speaker 1")]
        stats = get_speaker_statistics(segs)
        assert stats["Speaker 1"]["word_count"] == 5


# ---------------------------------------------------------------------------
# save_rttm
# ---------------------------------------------------------------------------

class TestSaveRttm:
    def test_writes_valid_rttm(self):
        segments = [
            SpeakerSegment(start=0.0, end=5.0, speaker="Speaker 1"),
            SpeakerSegment(start=5.0, end=10.0, speaker="Speaker 2"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            rttm_path = Path(tmpdir) / "test.rttm"
            result = save_rttm(segments, rttm_path)
            assert result == rttm_path
            content = rttm_path.read_text()
            lines = [l for l in content.strip().split("\n") if l]
            assert len(lines) == 2
            assert lines[0].startswith("SPEAKER")
            assert "Speaker_1" in lines[0]
            assert "Speaker_2" in lines[1]

    def test_rttm_duration_field(self):
        segments = [SpeakerSegment(start=1.0, end=4.0, speaker="Speaker 1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            rttm_path = Path(tmpdir) / "test.rttm"
            save_rttm(segments, rttm_path)
            content = rttm_path.read_text()
            parts = content.strip().split()
            # SPEAKER <id> <chnl> <tbeg> <tdur> ...
            tbeg = float(parts[3])
            tdur = float(parts[4])
            assert tbeg == pytest.approx(1.0)
            assert tdur == pytest.approx(3.0)

    def test_custom_recording_id(self):
        segments = [SpeakerSegment(start=0.0, end=2.0, speaker="Speaker 1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            rttm_path = Path(tmpdir) / "test.rttm"
            save_rttm(segments, rttm_path, recording_id="meeting_001")
            content = rttm_path.read_text()
            assert "meeting_001" in content

    def test_empty_segments_writes_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rttm_path = Path(tmpdir) / "empty.rttm"
            save_rttm([], rttm_path)
            content = rttm_path.read_text().strip()
            assert content == ""

    def test_speaker_spaces_replaced_with_underscores(self):
        segments = [SpeakerSegment(start=0.0, end=1.0, speaker="Speaker One")]
        with tempfile.TemporaryDirectory() as tmpdir:
            rttm_path = Path(tmpdir) / "test.rttm"
            save_rttm(segments, rttm_path)
            content = rttm_path.read_text()
            assert "Speaker_One" in content
            assert "Speaker One" not in content


# ---------------------------------------------------------------------------
# DiarizationMetrics
# ---------------------------------------------------------------------------

class TestDiarizationMetrics:
    def _make_metrics(self, **overrides):
        defaults = dict(
            der=0.15,
            jer=0.20,
            missed_speech_rate=0.05,
            false_alarm_rate=0.03,
            speaker_confusion_rate=0.07,
            collar=0.25,
            total_reference_duration=120.0,
        )
        defaults.update(overrides)
        return DiarizationMetrics(**defaults)

    def test_to_dict_keys(self):
        m = self._make_metrics()
        d = m.to_dict()
        assert set(d.keys()) == {
            "der", "jer", "missed_speech_rate", "false_alarm_rate",
            "speaker_confusion_rate", "collar_s", "total_reference_duration_s"
        }

    def test_to_dict_values(self):
        m = self._make_metrics(der=0.1, jer=0.2)
        d = m.to_dict()
        assert d["der"] == pytest.approx(0.1)
        assert d["jer"] == pytest.approx(0.2)

    def test_str_contains_percentages(self):
        m = self._make_metrics(der=0.15)
        s = str(m)
        assert "15.00%" in s

    def test_str_contains_collar(self):
        m = self._make_metrics(collar=0.25)
        s = str(m)
        assert "0.25" in s


# ---------------------------------------------------------------------------
# diarize_audio — error paths (no real model needed)
# ---------------------------------------------------------------------------

class TestDiarizeAudioErrorPaths:
    def test_missing_hf_token_raises_diarization_error(self, tmp_path):
        # Create a valid-looking audio file so validate_audio_file passes
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        with patch.dict(os.environ, {}, clear=True):
            # Ensure HF_TOKEN is not set
            os.environ.pop("HF_TOKEN", None)
            from src.diarize import diarize_audio
            with pytest.raises(DiarizationError, match="Hugging Face token"):
                diarize_audio(audio_file, hf_token=None)

    def test_missing_audio_file_raises_audio_file_error(self, tmp_path):
        missing = tmp_path / "nonexistent.wav"
        from src.diarize import diarize_audio
        with pytest.raises(AudioFileError):
            diarize_audio(missing)

    def test_unsupported_audio_format_raises_audio_file_error(self, tmp_path):
        bad_file = tmp_path / "audio.xyz"
        bad_file.write_bytes(b"\x00" * 100)
        from src.diarize import diarize_audio
        with pytest.raises(AudioFileError):
            diarize_audio(bad_file)


# ---------------------------------------------------------------------------
# load_rttm — skip if pyannote not installed
# ---------------------------------------------------------------------------

pyannote_available = False
try:
    from pyannote.core import Annotation, Segment  # noqa: F401
    pyannote_available = True
except ImportError:
    pass


@pytest.mark.skipif(not pyannote_available, reason="pyannote.core not installed")
class TestLoadRttm:
    def test_load_valid_rttm(self, tmp_path):
        from src.diarize import load_rttm
        rttm_content = (
            "SPEAKER rec1 1 0.000 5.000 <NA> <NA> Speaker_1 <NA> <NA>\n"
            "SPEAKER rec1 1 5.000 3.000 <NA> <NA> Speaker_2 <NA> <NA>\n"
        )
        rttm_path = tmp_path / "test.rttm"
        rttm_path.write_text(rttm_content)
        annotation = load_rttm(rttm_path)
        assert annotation is not None
        labels = set(annotation.labels())
        assert "Speaker_1" in labels
        assert "Speaker_2" in labels

    def test_load_empty_rttm(self, tmp_path):
        from src.diarize import load_rttm
        rttm_path = tmp_path / "empty.rttm"
        rttm_path.write_text("")
        annotation = load_rttm(rttm_path)
        assert len(annotation) == 0

    def test_load_rttm_missing_file_raises(self, tmp_path):
        from src.diarize import load_rttm
        with pytest.raises(DiarizationError):
            load_rttm(tmp_path / "missing.rttm")
