"""
Tests for src/preprocess.py

Uses mocking to avoid actual audio I/O and noisereduce dependency in unit tests.
"""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.config import PreprocessConfig
from src.exceptions import AudioFileError, PreprocessingError


# ---------------------------------------------------------------------------
# denoise_audio — preprocessing disabled (format conversion only)
# ---------------------------------------------------------------------------

class TestDenoiseAudioDisabled:
    def test_disabled_calls_ffmpeg_conversion(self, tmp_path):
        """When config.enabled=False, ffmpeg conversion runs but noisereduce does not."""
        src = tmp_path / "input.wav"
        src.write_bytes(b"fake audio")
        dst = tmp_path / "output.wav"
        config = PreprocessConfig(enabled=False)

        with patch("src.preprocess.validate_audio_file"), \
             patch("src.preprocess.ffmpeg") as mock_ffmpeg, \
             patch("src.preprocess._load_audio_as_float32") as mock_load:

            # Set up ffmpeg mock chain
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value = mock_stream

            result = __import__("src.preprocess", fromlist=["denoise_audio"]).denoise_audio(
                src, dst, config
            )

        mock_ffmpeg.input.assert_called_once()
        mock_load.assert_not_called()
        assert result == dst

    def test_disabled_does_not_call_noisereduce(self, tmp_path):
        src = tmp_path / "input.wav"
        src.write_bytes(b"fake audio")
        dst = tmp_path / "output.wav"
        config = PreprocessConfig(enabled=False, denoise=True)

        with patch("src.preprocess.validate_audio_file"), \
             patch("src.preprocess.ffmpeg") as mock_ffmpeg, \
             patch("src.preprocess._load_audio_as_float32") as mock_load:

            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value = mock_stream

            from src.preprocess import denoise_audio
            denoise_audio(src, dst, config)

        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# denoise_audio — preprocessing enabled
# ---------------------------------------------------------------------------

class TestDenoiseAudioEnabled:
    def _make_mock_audio(self):
        return np.zeros(16000, dtype=np.float32), 16000

    def test_denoise_and_normalize_called_when_enabled(self, tmp_path):
        src = tmp_path / "input.wav"
        src.write_bytes(b"fake audio")
        dst = tmp_path / "output.wav"
        config = PreprocessConfig(enabled=True, denoise=True, normalize_volume=True)

        audio, sr = self._make_mock_audio()

        with patch("src.preprocess.validate_audio_file"), \
             patch("src.preprocess._load_audio_as_float32", return_value=(audio, sr)), \
             patch("src.preprocess.sf") as mock_sf, \
             patch("noisereduce.reduce_noise", return_value=audio) as mock_nr:

            from src.preprocess import denoise_audio
            result = denoise_audio(src, dst, config)

        mock_nr.assert_called_once()
        mock_sf.write.assert_called_once()
        assert result == dst

    def test_normalize_volume_clips_to_unity(self, tmp_path):
        """Audio above 0 dBFS should be clipped to [-1.0, 1.0] after normalization."""
        src = tmp_path / "input.wav"
        src.write_bytes(b"fake audio")
        dst = tmp_path / "output.wav"
        config = PreprocessConfig(enabled=True, denoise=False, normalize_volume=True, target_peak_dbfs=0.0)

        # Audio with peak > 1.0 (simulating very loud source)
        loud_audio = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)

        written_audio = []

        def capture_write(path, audio, sr, **kwargs):
            written_audio.append(audio)

        with patch("src.preprocess.validate_audio_file"), \
             patch("src.preprocess._load_audio_as_float32", return_value=(loud_audio, 16000)), \
             patch("src.preprocess.sf") as mock_sf:

            mock_sf.write.side_effect = capture_write

            from src.preprocess import denoise_audio
            denoise_audio(src, dst, config)

        assert written_audio, "sf.write should have been called"
        result_audio = written_audio[0]
        assert np.all(result_audio <= 1.0), "Audio should not exceed +1.0"
        assert np.all(result_audio >= -1.0), "Audio should not go below -1.0"

    def test_invalid_file_raises_audio_file_error(self, tmp_path):
        src = tmp_path / "nonexistent.wav"
        dst = tmp_path / "output.wav"
        config = PreprocessConfig()

        with pytest.raises(AudioFileError):
            from src.preprocess import denoise_audio
            denoise_audio(src, dst, config)

    def test_noisereduce_failure_raises_preprocessing_error(self, tmp_path):
        src = tmp_path / "input.wav"
        src.write_bytes(b"fake audio")
        dst = tmp_path / "output.wav"
        config = PreprocessConfig(enabled=True, denoise=True)
        audio = np.zeros(16000, dtype=np.float32)

        with patch("src.preprocess.validate_audio_file"), \
             patch("src.preprocess._load_audio_as_float32", return_value=(audio, 16000)), \
             patch("noisereduce.reduce_noise", side_effect=RuntimeError("algorithm failed")):

            from src.preprocess import denoise_audio
            with pytest.raises(PreprocessingError):
                denoise_audio(src, dst, config)


# ---------------------------------------------------------------------------
# slice_segment_to_wav
# ---------------------------------------------------------------------------

class TestSliceSegmentToWav:
    def test_slice_calls_ffmpeg_with_padded_timestamps(self, tmp_path):
        src = tmp_path / "preprocessed.wav"
        src.write_bytes(b"fake audio")
        dst = tmp_path / "clip.wav"

        with patch("src.preprocess.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value = mock_stream

            from src.preprocess import slice_segment_to_wav
            result = slice_segment_to_wav(src, 10.0, 20.0, dst)

        # Verify padding applied: ss should be 10.0 - 0.05 = 9.95
        call_kwargs = mock_ffmpeg.input.call_args
        assert call_kwargs[1]["ss"] == pytest.approx(9.95)
        assert call_kwargs[1]["to"] == pytest.approx(20.05)
        assert result == dst

    def test_slice_zero_start_does_not_go_negative(self, tmp_path):
        src = tmp_path / "preprocessed.wav"
        src.write_bytes(b"fake audio")
        dst = tmp_path / "clip.wav"

        with patch("src.preprocess.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value = mock_stream

            from src.preprocess import slice_segment_to_wav
            slice_segment_to_wav(src, 0.0, 5.0, dst)

        call_kwargs = mock_ffmpeg.input.call_args
        assert call_kwargs[1]["ss"] >= 0.0, "ss must not be negative"

    def test_slice_ffmpeg_error_raises_audio_file_error(self, tmp_path):
        src = tmp_path / "preprocessed.wav"
        src.write_bytes(b"fake audio")
        dst = tmp_path / "clip.wav"

        import ffmpeg as ffmpeg_lib

        with patch("src.preprocess.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.Error = ffmpeg_lib.Error
            mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run.side_effect = \
                ffmpeg_lib.Error("ffmpeg", b"", b"some error")

            from src.preprocess import slice_segment_to_wav
            with pytest.raises(AudioFileError):
                slice_segment_to_wav(src, 1.0, 3.0, dst)
