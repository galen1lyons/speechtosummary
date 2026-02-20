"""
Custom exceptions for the speech-to-summary system.

Provides specific exception types for better error handling and debugging.
"""


class SpeechToSummaryError(Exception):
    """Base exception for all speech-to-summary errors."""
    pass


class AudioFileError(SpeechToSummaryError):
    """Raised when there's an issue with the audio file."""
    pass


class TranscriptionError(SpeechToSummaryError):
    """Raised when transcription fails."""
    pass


class SummarizationError(SpeechToSummaryError):
    """Raised when summarization fails."""
    pass


class ConfigurationError(SpeechToSummaryError):
    """Raised when configuration is invalid."""
    pass


class ModelLoadError(SpeechToSummaryError):
    """Raised when model loading fails."""
    pass


class ASRMetricsError(SpeechToSummaryError):
    """Raised when ASR metrics calculation fails."""
    pass


class DiarizationError(SpeechToSummaryError):
    """Raised when speaker diarization fails."""
    pass

