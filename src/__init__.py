"""
Speech-to-Summary System

A comprehensive system for transcribing audio and generating structured summaries
with ASR evaluation metrics (WER, CER, RTF).

Main components:
- Transcription: Whisper-based audio transcription
- Summarization: Structured summary generation
- ASR Metrics: WER, CER, RTF evaluation
- Diarization: Speaker identification and labeling
- Pipeline: End-to-end workflow

Example usage:
    from src import transcribe, create_structured_summary, WhisperConfig
    
    # Transcribe audio
    json_path, txt_path, metrics = transcribe(
        audio_path=Path("meeting.mp3"),
        out_base=Path("outputs/meeting"),
        model_name="base"
    )
    
    # Create summary
    transcript = load_transcript(json_path)
    summary = create_structured_summary(transcript)
"""

__version__ = "1.0.0"

# Configuration
from .config import ASRMetricsConfig, DiarizationConfig, PreprocessConfig, SummaryConfig, WhisperConfig

# Exceptions
from .exceptions import (
    ASRMetricsError,
    AudioFileError,
    ConfigurationError,
    DiarizationError,
    ModelLoadError,
    PreprocessingError,
    SpeechToSummaryError,
    SummarizationError,
    TranscriptionError,
)

# Logging
from .logger import get_logger, setup_logger

# Core functionality
from .transcribe import transcribe
from .summarize import create_structured_summary, load_transcript

# ASR Metrics
from .evaluation.asr_metrics import (
    ASRMetrics,
    calculate_cer,
    calculate_rtf,
    calculate_wer,
    evaluate_transcription,
    RTFTimer,
)

# Speaker Diarization
from .diarize import (
    SpeakerSegment,
    TranscriptSegment,
    diarize_audio,
    format_transcript_with_speakers,
    get_speaker_statistics,
    merge_diarization_with_transcript,
)

# Utilities
from .utils import (
    format_duration,
    get_file_size_mb,
    normalize_text,
    parse_device,
    sanitize_filename,
    strip_transcript_timestamps,
    validate_audio_file,
)

__all__ = [
    # Version
    "__version__",
    
    # Configuration
    "WhisperConfig",
    "SummaryConfig",
    "ASRMetricsConfig",
    "DiarizationConfig",
    "PreprocessConfig",

    # Exceptions
    "SpeechToSummaryError",
    "AudioFileError",
    "TranscriptionError",
    "SummarizationError",
    "ConfigurationError",
    "ModelLoadError",
    "ASRMetricsError",
    "DiarizationError",
    "PreprocessingError",
    
    # Logging
    "setup_logger",
    "get_logger",
    
    # Core functions
    "transcribe",
    "create_structured_summary",
    "load_transcript",
    
    # ASR Metrics
    "ASRMetrics",
    "calculate_wer",
    "calculate_cer",
    "calculate_rtf",
    "evaluate_transcription",
    "RTFTimer",
    
    # Speaker Diarization
    "SpeakerSegment",
    "TranscriptSegment",
    "diarize_audio",
    "merge_diarization_with_transcript",
    "format_transcript_with_speakers",
    "get_speaker_statistics",
    
    # Utilities
    "validate_audio_file",
    "sanitize_filename",
    "parse_device",
    "get_file_size_mb",
    "normalize_text",
    "format_duration",
    "strip_transcript_timestamps",
]
