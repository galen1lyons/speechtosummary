"""
Diarization evaluation subsystem.

Provides post-battle scoring metrics:
- DER (Diarization Error Rate)
- JER (Jaccard Error Rate)
"""
from dataclasses import dataclass
from typing import Dict

from ..exceptions import DiarizationError


@dataclass
class DiarizationMetrics:
    """Container for diarization evaluation metrics."""

    der: float
    jer: float
    missed_speech_rate: float
    false_alarm_rate: float
    speaker_confusion_rate: float
    collar: float
    total_reference_duration: float

    def to_dict(self) -> Dict:
        return {
            "der": self.der,
            "jer": self.jer,
            "missed_speech_rate": self.missed_speech_rate,
            "false_alarm_rate": self.false_alarm_rate,
            "speaker_confusion_rate": self.speaker_confusion_rate,
            "collar_s": self.collar,
            "total_reference_duration_s": self.total_reference_duration,
        }

    def __str__(self) -> str:
        return (
            f"Diarization Metrics (collar={self.collar}s):\n"
            f"  DER:               {self.der:.2%}\n"
            f"  JER:               {self.jer:.2%}\n"
            f"  Missed speech:     {self.missed_speech_rate:.2%}\n"
            f"  False alarm:       {self.false_alarm_rate:.2%}\n"
            f"  Speaker confusion: {self.speaker_confusion_rate:.2%}\n"
            f"  Reference duration: {self.total_reference_duration:.1f}s"
        )


def evaluate_diarization(
    reference: "Annotation",
    hypothesis: "Annotation",
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> DiarizationMetrics:
    """Compute DER and JER between reference and hypothesis diarization."""
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    except ImportError as e:
        raise DiarizationError(
            f"pyannote.metrics is required for diarization evaluation: {e}"
        )

    try:
        der_metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        components = der_metric(reference, hypothesis, detailed=True)

        total = components["total"]
        der = components["diarization error rate"]
        missed = components["missed detection"] / total if total > 0 else 0.0
        false_alarm = components["false alarm"] / total if total > 0 else 0.0
        confusion = components["confusion"] / total if total > 0 else 0.0

        jer_metric = JaccardErrorRate(collar=collar)
        jer = float(jer_metric(reference, hypothesis))
    except Exception as e:
        raise DiarizationError(f"Diarization evaluation failed: {e}")

    return DiarizationMetrics(
        der=float(der),
        jer=jer,
        missed_speech_rate=float(missed),
        false_alarm_rate=float(false_alarm),
        speaker_confusion_rate=float(confusion),
        collar=collar,
        total_reference_duration=float(total),
    )

