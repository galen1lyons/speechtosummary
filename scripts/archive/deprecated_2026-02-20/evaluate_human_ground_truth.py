#!/usr/bin/env python3
"""
Comprehensive evaluation against human ground truth.

Metrics computed:
  - WER  (Word Error Rate)        — transcription word accuracy
  - CER  (Character Error Rate)   — transcription character accuracy
  - DER  (Diarization Error Rate) — speaker diarization accuracy
  - JER  (Jaccard Error Rate)     — per-speaker Jaccard overlap
  - RTF  (Real-Time Factor)       — processing speed (requires processing_time)

Usage:
    python scripts/evaluate_human_ground_truth.py

Inputs (hardcoded for this evaluation):
    Reference transcript : outputs/studio_sembang/studio_sembang_human_transcribe.txt
    Reference diarization: outputs/studio_sembang/studio_sembang_human_diarization.txt
    Model transcript     : outputs/Studio_Sembang_Slice_-_Wan_in_a_Million_ft.txt
    Model RTTM           : outputs/Studio_Sembang_Slice_-_Wan_in_a_Million_ft.rttm
    Model JSON           : outputs/Studio_Sembang_Slice_-_Wan_in_a_Million_ft.json
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.asr_metrics import evaluate_transcription


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_human_transcript(file_path: Path) -> str:
    """
    Extract plain text from human timestamped transcript.

    Format: [12.91 - 15.59] Hey guys It's Amy dan...
    Skips lines containing [unintelligible].
    """
    lines = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "]" in line:
                text = line.split("]", 1)[1].strip()
                if text and "[unintelligible]" not in text.lower():
                    lines.append(text)
    return " ".join(lines)


def extract_text_from_model_transcript(file_path: Path) -> str:
    """
    Extract plain text from model timestamped transcript.

    Format: [12.91 - 15.59] Maksud Amy dan...
    """
    lines = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "]" in line:
                text = line.split("]", 1)[1].strip()
                if text:
                    lines.append(text)
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Diarization parsing helpers
# ---------------------------------------------------------------------------

def parse_human_diarization(file_path: Path):
    """
    Parse human diarization file into a pyannote Annotation.

    Accepted format:
        [SPEAKER_00 12.91 - 15.59] Hey guys...

    Lines without a SPEAKER_XX label are silently skipped.
    """
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    # Match [SPEAKER_XX start - end] where XX are digits
    pattern = re.compile(r"\[(SPEAKER_\d+)\s+([\d.]+)\s*-\s*([\d.]+)\]")

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = pattern.search(line)
            if match:
                speaker, start, end = match.groups()
                start, end = float(start), float(end)
                if end > start:
                    annotation[Segment(start, end)] = speaker

    return annotation


def parse_rttm(file_path: Path):
    """
    Parse RTTM file into a pyannote Annotation.

    RTTM format (fields):
        SPEAKER <file_id> <channel> <onset> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    from pyannote.core import Annotation, Segment

    annotation = Annotation()

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                onset = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                end = onset + duration
                if end > onset:
                    annotation[Segment(onset, end)] = speaker

    return annotation


# ---------------------------------------------------------------------------
# DER / JER computation
# ---------------------------------------------------------------------------

def compute_diarization_metrics(reference, hypothesis, collar: float = 0.25):
    """
    Compute DER and JER using pyannote.metrics.

    Returns:
        dict with der, jer, missed_speech, false_alarm, confusion (all as floats 0-1)
    """
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    der_metric = DiarizationErrorRate(collar=collar)
    jer_metric = JaccardErrorRate(collar=collar)

    der = float(der_metric(reference, hypothesis))
    jer = float(jer_metric(reference, hypothesis))

    components = der_metric.compute_components(reference, hypothesis)
    total_ref = components.get("total", 1.0) or 1.0

    return {
        "der": der,
        "jer": jer,
        "missed_speech_rate": float(components.get("missed detection", 0.0)) / total_ref,
        "false_alarm_rate": float(components.get("false alarm", 0.0)) / total_ref,
        "speaker_confusion_rate": float(components.get("confusion", 0.0)) / total_ref,
        "total_reference_duration_s": float(total_ref),
        "collar_s": collar,
        "reference_speakers": list(reference.labels()),
        "hypothesis_speakers": list(hypothesis.labels()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def quality_label(rate: float, thresholds: tuple) -> str:
    """Return a quality label based on error rate and threshold tuple."""
    if rate < thresholds[0]:
        return "Excellent"
    if rate < thresholds[1]:
        return "Very Good"
    if rate < thresholds[2]:
        return "Good"
    if rate < thresholds[3]:
        return "Fair"
    return "Poor"


def main():
    base = PROJECT_ROOT / "outputs"

    paths = {
        "ref_transcript":   base / "studio_sembang" / "studio_sembang_human_transcribe.txt",
        "ref_diarization":  base / "studio_sembang" / "studio_sembang_human_diarization.txt",
        "hyp_transcript":   base / "Studio_Sembang_Slice_-_Wan_in_a_Million_ft.txt",
        "hyp_rttm":         base / "Studio_Sembang_Slice_-_Wan_in_a_Million_ft.rttm",
        "hyp_json":         base / "Studio_Sembang_Slice_-_Wan_in_a_Million_ft.json",
        "output":           base / "studio_sembang" / "human_eval_metrics.json",
    }

    for key, p in paths.items():
        if key != "output" and not p.exists():
            print(f"ERROR: Missing file: {p}")
            sys.exit(1)

    print("=" * 65)
    print("  COMPREHENSIVE HUMAN GROUND TRUTH EVALUATION")
    print("  FasterWhisper (ideal config) vs Human Reference")
    print("=" * 65)

    # ------------------------------------------------------------------
    # WER / CER
    # ------------------------------------------------------------------
    print("\n[1/3] Transcription metrics (WER / CER)")
    print("-" * 40)

    ref_text = extract_text_from_human_transcript(paths["ref_transcript"])
    hyp_text = extract_text_from_model_transcript(paths["hyp_transcript"])

    print(f"  Reference : {len(ref_text.split()):,} words  |  {len(ref_text):,} chars")
    print(f"  Hypothesis: {len(hyp_text.split()):,} words  |  {len(hyp_text):,} chars")

    asr = evaluate_transcription(
        reference=ref_text,
        hypothesis=hyp_text,
        normalize=True,
        remove_punctuation=True,
    )

    # ------------------------------------------------------------------
    # DER / JER
    # ------------------------------------------------------------------
    print("\n[2/3] Diarization metrics (DER / JER)")
    print("-" * 40)

    ref_annot = parse_human_diarization(paths["ref_diarization"])
    hyp_annot = parse_rttm(paths["hyp_rttm"])

    print(f"  Reference speakers : {sorted(ref_annot.labels())}")
    print(f"  Hypothesis speakers: {sorted(hyp_annot.labels())}")

    diar = compute_diarization_metrics(ref_annot, hyp_annot, collar=0.25)

    # ------------------------------------------------------------------
    # RTF
    # ------------------------------------------------------------------
    print("\n[3/3] Real-Time Factor (RTF)")
    print("-" * 40)

    with open(paths["hyp_json"], encoding="utf-8") as f:
        model_json = json.load(f)

    audio_duration = model_json.get("duration")
    processing_time = None  # not stored in pipeline output

    print(f"  Audio duration   : {audio_duration:.1f}s  ({audio_duration / 60:.1f} min)")
    print("  Processing time  : not captured — RTF unavailable")
    print("  (Re-run pipeline with timing to get RTF)")

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    wer_pct = asr.wer * 100
    cer_pct = asr.cer * 100
    der_pct = diar["der"] * 100
    jer_pct = diar["jer"] * 100

    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  Metric   Value       Quality          Details           │
  ├─────────────────────────────────────────────────────────┤
  │  WER    {wer_pct:6.2f}%    {quality_label(asr.wer,  (0.05,0.10,0.15,0.25)):<16}  {asr.word_errors}/{asr.word_total} words  │
  │  CER    {cer_pct:6.2f}%    {quality_label(asr.cer,  (0.02,0.05,0.10,0.15)):<16}  {asr.char_errors}/{asr.char_total} chars  │
  │  DER    {der_pct:6.2f}%    {quality_label(diar['der'], (0.05,0.10,0.20,0.30)):<16}  collar=0.25s      │
  │  JER    {jer_pct:6.2f}%    {quality_label(diar['jer'], (0.05,0.10,0.20,0.30)):<16}  collar=0.25s      │
  │  RTF    N/A          —                process_time=null │
  └─────────────────────────────────────────────────────────┘""")

    print(f"""
  WER breakdown:  S={asr.word_substitutions}  D={asr.word_deletions}  I={asr.word_insertions}
  CER breakdown:  S={asr.char_substitutions}  D={asr.char_deletions}  I={asr.char_insertions}

  DER breakdown:
    Missed speech     : {diar['missed_speech_rate']:.2%}
    False alarm       : {diar['false_alarm_rate']:.2%}
    Speaker confusion : {diar['speaker_confusion_rate']:.2%}
    Total ref duration: {diar['total_reference_duration_s']:.1f}s
""")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    result = {
        "reference_transcript": str(paths["ref_transcript"]),
        "reference_diarization": str(paths["ref_diarization"]),
        "hypothesis_transcript": str(paths["hyp_transcript"]),
        "hypothesis_rttm": str(paths["hyp_rttm"]),
        "wer": asr.to_dict()["wer"],
        "cer": asr.to_dict()["cer"],
        "diarization": diar,
        "rtf": {
            "factor": None,
            "audio_duration_s": audio_duration,
            "processing_time_s": processing_time,
            "note": "processing_time not captured during pipeline run",
        },
    }

    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["output"], "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved to: {paths['output']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
