"""
Audio preprocessing module.

Applies noise reduction and volume normalization to audio before diarization
and transcription. Always produces a 16 kHz mono WAV file for consistent
downstream consumption by faster-whisper or openai-whisper.
"""
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import ffmpeg

from .config import PreprocessConfig
from .exceptions import AudioFileError, PreprocessingError
from .logger import get_logger
from .utils import validate_audio_file

logger = get_logger(__name__)


def _load_audio_as_float32(audio_path: Path) -> tuple:
    """
    Load any audio format to a float32 numpy array at 16 kHz mono.

    Uses ffmpeg-python to convert the source to a temporary 16 kHz mono WAV,
    then reads it with soundfile. Handles mp3, m4a, ogg, flac, and all other
    formats Whisper supports without requiring librosa.

    Args:
        audio_path: Path to source audio file (any format)

    Returns:
        Tuple of (float32 numpy array in [-1.0, 1.0], sample_rate)

    Raises:
        AudioFileError: If the file cannot be decoded
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        (
            ffmpeg
            .input(str(audio_path))
            .output(tmp_path, acodec="pcm_s16le", ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )

        audio, sr = sf.read(tmp_path, dtype="float32")
        return audio, sr

    except ffmpeg.Error as e:
        raise AudioFileError(
            f"Failed to decode audio file '{audio_path}': {e.stderr.decode() if e.stderr else str(e)}"
        ) from e
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def denoise_audio(
    audio_path: Path,
    output_path: Path,
    config: PreprocessConfig,
) -> Path:
    """
    Preprocess audio: optionally denoise and normalize volume.

    Always writes a 16 kHz mono PCM WAV to output_path regardless of input
    format. When config.enabled is False, only format conversion is performed.

    Args:
        audio_path: Source audio file (any Whisper-supported format)
        output_path: Destination .wav path
        config: PreprocessConfig controlling which stages run

    Returns:
        Path to the output .wav file (same as output_path)

    Raises:
        AudioFileError: If source file is invalid or decoding fails
        PreprocessingError: If the processing algorithm fails
    """
    validate_audio_file(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if not config.enabled:
        # Format conversion only — no denoising or normalization
        try:
            (
                ffmpeg
                .input(str(audio_path))
                .output(str(output_path), acodec="pcm_s16le", ar=16000, ac=1)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise AudioFileError(
                f"Failed to convert audio file '{audio_path}': "
                f"{e.stderr.decode() if e.stderr else str(e)}"
            ) from e
        elapsed = time.time() - start_time
        logger.info(f"Preprocessing disabled — converted to 16kHz WAV in {elapsed:.2f}s")
        return output_path

    # Load audio as float32
    audio, sr = _load_audio_as_float32(audio_path)

    try:
        # Noise reduction
        if config.denoise:
            import noisereduce as nr
            audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=config.noise_reduce_stationary,
            )

        # Volume normalization
        if config.normalize_volume:
            peak = np.abs(audio).max()
            if peak > 0:
                target_linear = 10 ** (config.target_peak_dbfs / 20.0)
                audio = audio * (target_linear / peak)

        # Safety clip to [-1.0, 1.0]
        audio = np.clip(audio, -1.0, 1.0)

    except Exception as e:
        raise PreprocessingError(
            f"Audio preprocessing failed for '{audio_path}': {e}"
        ) from e

    sf.write(str(output_path), audio, sr, subtype="PCM_16")

    elapsed = time.time() - start_time
    logger.info(
        f"Preprocessing complete in {elapsed:.2f}s "
        f"(denoise={config.denoise}, normalize={config.normalize_volume})"
    )
    return output_path


def slice_segment_to_wav(
    audio_path: Path,
    start: float,
    end: float,
    output_path: Path,
) -> Path:
    """
    Extract a time slice of an audio file to a 16 kHz mono WAV.

    Uses ffmpeg-python (already a project dependency). A 0.05s padding is
    applied on each side to avoid hard cuts at segment boundaries; downstream
    timestamps should still use the original start/end values.

    Args:
        audio_path: Source audio file (preferably the preprocessed WAV)
        start: Segment start time in seconds (original, unpadded)
        end: Segment end time in seconds (original, unpadded)
        output_path: Destination .wav path

    Returns:
        Path to the extracted clip (same as output_path)

    Raises:
        AudioFileError: If extraction fails
    """
    PADDING = 0.05
    safe_start = max(0.0, start - PADDING)
    safe_end = end + PADDING

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        (
            ffmpeg
            .input(str(audio_path), ss=safe_start, to=safe_end)
            .output(str(output_path), acodec="pcm_s16le", ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise AudioFileError(
            f"Failed to slice segment [{start:.2f}s–{end:.2f}s] from '{audio_path}': "
            f"{e.stderr.decode() if e.stderr else str(e)}"
        ) from e

    return output_path
