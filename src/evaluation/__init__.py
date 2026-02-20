"""
Evaluation subsystems for post-battle scoring.

These modules are optional capabilities used when reference artifacts are
available (e.g., human transcript/RTTM).
"""

from .asr_metrics import (
    ASRMetrics,
    RTFTimer,
    calculate_cer,
    calculate_rtf,
    calculate_wer,
    evaluate_transcription,
)
from .diarization_metrics import DiarizationMetrics, evaluate_diarization

__all__ = [
    "ASRMetrics",
    "RTFTimer",
    "calculate_wer",
    "calculate_cer",
    "calculate_rtf",
    "evaluate_transcription",
    "DiarizationMetrics",
    "evaluate_diarization",
]

