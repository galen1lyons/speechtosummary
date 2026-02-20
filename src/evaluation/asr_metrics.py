"""
ASR evaluation subsystem.

Provides post-battle scoring metrics:
- WER (Word Error Rate)
- CER (Character Error Rate)
- RTF (Real Time Factor)
"""
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..exceptions import ASRMetricsError
from ..logger import get_logger
from ..utils import normalize_text

logger = get_logger(__name__)


@dataclass
class ASRMetrics:
    """Container for ASR evaluation metrics."""

    wer: Optional[float] = None
    word_errors: Optional[int] = None
    word_total: Optional[int] = None
    word_substitutions: Optional[int] = None
    word_deletions: Optional[int] = None
    word_insertions: Optional[int] = None

    cer: Optional[float] = None
    char_errors: Optional[int] = None
    char_total: Optional[int] = None
    char_substitutions: Optional[int] = None
    char_deletions: Optional[int] = None
    char_insertions: Optional[int] = None

    rtf: Optional[float] = None
    audio_duration: Optional[float] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "wer": {
                "rate": self.wer,
                "errors": self.word_errors,
                "total_words": self.word_total,
                "substitutions": self.word_substitutions,
                "deletions": self.word_deletions,
                "insertions": self.word_insertions,
            },
            "cer": {
                "rate": self.cer,
                "errors": self.char_errors,
                "total_chars": self.char_total,
                "substitutions": self.char_substitutions,
                "deletions": self.char_deletions,
                "insertions": self.char_insertions,
            },
            "rtf": {
                "factor": self.rtf,
                "audio_duration_s": self.audio_duration,
                "processing_time_s": self.processing_time,
            },
        }

    def __str__(self) -> str:
        lines = ["ASR Metrics:"]

        if self.wer is not None:
            lines.append(f"  WER: {self.wer:.2%} ({self.word_errors}/{self.word_total} errors)")
            lines.append(f"    Substitutions: {self.word_substitutions}")
            lines.append(f"    Deletions: {self.word_deletions}")
            lines.append(f"    Insertions: {self.word_insertions}")

        if self.cer is not None:
            lines.append(f"  CER: {self.cer:.2%} ({self.char_errors}/{self.char_total} errors)")
            lines.append(f"    Substitutions: {self.char_substitutions}")
            lines.append(f"    Deletions: {self.char_deletions}")
            lines.append(f"    Insertions: {self.char_insertions}")

        if self.rtf is not None:
            lines.append(f"  RTF: {self.rtf:.2f}x realtime")
            lines.append(f"    Audio duration: {self.audio_duration:.1f}s")
            lines.append(f"    Processing time: {self.processing_time:.1f}s")

        return "\n".join(lines)


def levenshtein_distance(ref: list, hyp: list) -> Tuple[int, int, int, int]:
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]

    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    i, j = len(ref), len(hyp)
    substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if i == 0:
            insertions += j
            break
        if j == 0:
            deletions += i
            break
        if ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif d[i][j] == d[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    total_errors = substitutions + deletions + insertions
    return total_errors, substitutions, deletions, insertions


def calculate_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
    remove_punctuation: bool = True,
) -> Tuple[float, int, int, int, int, int]:
    if normalize:
        reference = normalize_text(reference, remove_punctuation=remove_punctuation)
        hypothesis = normalize_text(hypothesis, remove_punctuation=remove_punctuation)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        raise ASRMetricsError("Reference transcript is empty")

    errors, subs, dels, ins = levenshtein_distance(ref_words, hyp_words)
    wer = errors / len(ref_words)
    return wer, errors, len(ref_words), subs, dels, ins


def calculate_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> Tuple[float, int, int, int, int, int]:
    if normalize:
        reference = normalize_text(reference, remove_punctuation=False)
        hypothesis = normalize_text(hypothesis, remove_punctuation=False)

    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if len(ref_chars) == 0:
        raise ASRMetricsError("Reference transcript is empty")

    errors, subs, dels, ins = levenshtein_distance(ref_chars, hyp_chars)
    cer = errors / len(ref_chars)
    return cer, errors, len(ref_chars), subs, dels, ins


def calculate_rtf(audio_duration: float, processing_time: float) -> float:
    if audio_duration <= 0:
        raise ASRMetricsError(f"Invalid audio duration: {audio_duration}")
    return processing_time / audio_duration


def evaluate_transcription(
    reference: str,
    hypothesis: str,
    audio_duration: Optional[float] = None,
    processing_time: Optional[float] = None,
    calculate_wer_metric: bool = True,
    calculate_cer_metric: bool = True,
    normalize: bool = True,
    remove_punctuation: bool = True,
) -> ASRMetrics:
    metrics = ASRMetrics()

    if calculate_wer_metric:
        try:
            wer, w_errors, w_total, w_subs, w_dels, w_ins = calculate_wer(
                reference, hypothesis, normalize, remove_punctuation
            )
            metrics.wer = wer
            metrics.word_errors = w_errors
            metrics.word_total = w_total
            metrics.word_substitutions = w_subs
            metrics.word_deletions = w_dels
            metrics.word_insertions = w_ins
            logger.info(f"WER: {wer:.2%} ({w_errors}/{w_total} errors)")
        except Exception as e:
            logger.warning(f"Failed to calculate WER: {e}")

    if calculate_cer_metric:
        try:
            cer, c_errors, c_total, c_subs, c_dels, c_ins = calculate_cer(
                reference, hypothesis, normalize
            )
            metrics.cer = cer
            metrics.char_errors = c_errors
            metrics.char_total = c_total
            metrics.char_substitutions = c_subs
            metrics.char_deletions = c_dels
            metrics.char_insertions = c_ins
            logger.info(f"CER: {cer:.2%} ({c_errors}/{c_total} errors)")
        except Exception as e:
            logger.warning(f"Failed to calculate CER: {e}")

    if audio_duration is not None and processing_time is not None:
        try:
            rtf = calculate_rtf(audio_duration, processing_time)
            metrics.rtf = rtf
            metrics.audio_duration = audio_duration
            metrics.processing_time = processing_time
            logger.info(f"RTF: {rtf:.2f}x realtime ({processing_time:.1f}s / {audio_duration:.1f}s)")
        except Exception as e:
            logger.warning(f"Failed to calculate RTF: {e}")

    return metrics


class RTFTimer:
    """Context manager for measuring Real Time Factor."""

    def __init__(self, audio_duration: float):
        self.audio_duration = audio_duration
        self.start_time = None
        self.end_time = None
        self.processing_time = None
        self.rtf = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
        self.rtf = calculate_rtf(self.audio_duration, self.processing_time)
        logger.info(
            f"Processing completed in {self.processing_time:.2f}s "
            f"(RTF: {self.rtf:.2f}x realtime)"
        )

