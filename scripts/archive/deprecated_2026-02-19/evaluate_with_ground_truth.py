#!/usr/bin/env python3
"""
Evaluate ASR output against ground truth transcript.

This script calculates proper ASR metrics (WER/CER) by comparing:
- Ground truth: Manually transcribed reference
- Hypothesis: Model output (faster-whisper FW5)

Usage:
    python scripts/evaluate_with_ground_truth.py \\
        --reference data/ground_truth.txt \\
        --hypothesis outputs/faster_whisper_final/FW5_combined_best.txt
"""
import argparse
import json
from pathlib import Path

from src.asr_metrics import evaluate_transcription, ASRMetrics


def extract_text_from_timestamped(file_path: Path) -> str:
    """
    Extract plain text from timestamped transcript.

    Converts format like:
        [0.56 - 2.56] Oh
        [2.56 - 4.56] today's episode is

    To:
        Oh today's episode is
    """
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract text after timestamp
            if ']' in line:
                text = line.split(']', 1)[1].strip()
                if text:
                    lines.append(text)

    return ' '.join(lines)


def load_transcript(file_path: Path, is_timestamped: bool = False) -> str:
    """
    Load transcript from file.

    Args:
        file_path: Path to transcript file
        is_timestamped: If True, extract text from [timestamp] format

    Returns:
        Plain text transcript
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Transcript not found: {file_path}")

    if is_timestamped:
        return extract_text_from_timestamped(file_path)
    else:
        return file_path.read_text(encoding='utf-8').strip()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR output against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate_with_ground_truth.py \\
      --reference data/ground_truth.txt \\
      --hypothesis outputs/faster_whisper_final/FW5_combined_best.txt

  # With custom output
  python scripts/evaluate_with_ground_truth.py \\
      --reference data/ground_truth.txt \\
      --hypothesis outputs/faster_whisper_final/FW5_combined_best.txt \\
      --output results/evaluation.json

  # Keep punctuation for WER
  python scripts/evaluate_with_ground_truth.py \\
      --reference data/ground_truth.txt \\
      --hypothesis outputs/faster_whisper_final/FW5_combined_best.txt \\
      --keep-punctuation

Creating Ground Truth:
  1. Listen to the audio file carefully
  2. Transcribe exactly what you hear (not what model generated)
  3. Save as plain text file: data/ground_truth.txt
  4. Run this script to calculate WER/CER
        """
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference/ground truth transcript (plain text or timestamped)"
    )
    parser.add_argument(
        "--hypothesis",
        required=True,
        help="Path to hypothesis/model output transcript"
    )
    parser.add_argument(
        "--output",
        help="Path to save metrics JSON (optional)"
    )
    parser.add_argument(
        "--reference-is-timestamped",
        action="store_true",
        help="Reference file has [timestamp] format"
    )
    parser.add_argument(
        "--hypothesis-is-timestamped",
        action="store_true",
        default=True,
        help="Hypothesis file has [timestamp] format (default: True for model output)"
    )
    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help="Keep punctuation when calculating WER (default: remove)"
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Show first 500 characters of each transcript for comparison"
    )

    args = parser.parse_args()

    # Load transcripts
    print("="*60)
    print("ASR EVALUATION WITH GROUND TRUTH")
    print("="*60)
    print()

    print(f"Loading reference: {args.reference}")
    reference = load_transcript(
        Path(args.reference),
        is_timestamped=args.reference_is_timestamped
    )
    print(f"  ✓ {len(reference.split())} words, {len(reference)} characters")

    print(f"Loading hypothesis: {args.hypothesis}")
    hypothesis = load_transcript(
        Path(args.hypothesis),
        is_timestamped=args.hypothesis_is_timestamped
    )
    print(f"  ✓ {len(hypothesis.split())} words, {len(hypothesis)} characters")
    print()

    # Show preview if requested
    if args.show_diff:
        print("="*60)
        print("TRANSCRIPT PREVIEW (first 500 chars)")
        print("="*60)
        print("Reference:")
        print(f"  {reference[:500]}...")
        print()
        print("Hypothesis:")
        print(f"  {hypothesis[:500]}...")
        print()

    # Calculate metrics
    print("="*60)
    print("CALCULATING METRICS")
    print("="*60)
    print()

    metrics = evaluate_transcription(
        reference=reference,
        hypothesis=hypothesis,
        normalize=True,
        remove_punctuation=not args.keep_punctuation
    )

    # Display results
    print(metrics)
    print()

    # Interpretation
    print("="*60)
    print("INTERPRETATION")
    print("="*60)

    if metrics.wer is not None:
        wer_pct = metrics.wer * 100
        if wer_pct < 5:
            wer_quality = "Excellent"
        elif wer_pct < 10:
            wer_quality = "Very Good"
        elif wer_pct < 15:
            wer_quality = "Good"
        elif wer_pct < 25:
            wer_quality = "Fair"
        else:
            wer_quality = "Poor"

        print(f"WER: {wer_pct:.1f}% - {wer_quality}")
        print(f"  → Model got {100 - wer_pct:.1f}% of words correct")

    if metrics.cer is not None:
        cer_pct = metrics.cer * 100
        if cer_pct < 2:
            cer_quality = "Excellent"
        elif cer_pct < 5:
            cer_quality = "Very Good"
        elif cer_pct < 10:
            cer_quality = "Good"
        elif cer_pct < 15:
            cer_quality = "Fair"
        else:
            cer_quality = "Poor"

        print(f"CER: {cer_pct:.1f}% - {cer_quality}")
        print(f"  → Model got {100 - cer_pct:.1f}% of characters correct")

    print()
    print("Error Breakdown:")
    if metrics.word_substitutions is not None:
        print(f"  Substitutions: {metrics.word_substitutions} words replaced incorrectly")
    if metrics.word_deletions is not None:
        print(f"  Deletions: {metrics.word_deletions} words missing from output")
    if metrics.word_insertions is not None:
        print(f"  Insertions: {metrics.word_insertions} extra words added")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "reference_file": str(args.reference),
            "hypothesis_file": str(args.hypothesis),
            "reference_word_count": len(reference.split()),
            "hypothesis_word_count": len(hypothesis.split()),
            "metrics": metrics.to_dict(),
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print()
        print(f"✅ Metrics saved to: {output_path}")

    print()
    print("="*60)
    print("NEXT STEPS")
    print("="*60)
    print("Based on the metrics:")
    print()

    if metrics.wer and metrics.wer < 0.15:  # Less than 15% WER
        print("✅ Model performance is good! Consider:")
        print("   - Testing on more diverse audio samples")
        print("   - Documenting this as the baseline quality")
    else:
        print("⚠️  Model needs improvement. Investigate:")
        print("   - Which types of errors are most common?")
        print("   - Are errors in specific sections (start/middle/end)?")
        print("   - Is the audio quality affecting results?")
        print("   - Try larger model (medium/large) for better accuracy")

    print()
    print("To create more ground truth files:")
    print("   1. Listen to the audio carefully")
    print("   2. Transcribe word-for-word what you hear")
    print("   3. Save as plain text file")
    print("   4. Run this script again")


if __name__ == "__main__":
    main()
