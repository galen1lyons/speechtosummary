#!/usr/bin/env python3
"""
Validate optimal faster-whisper settings across multiple audio files.
Tests the FW5 configuration (strict VAD + beam 7) on various audio samples.
"""
import json
import time
from pathlib import Path
from faster_whisper import WhisperModel


def count_hallucinations(segments):
    """Quick hallucination detector - counts repetitive words."""
    hallucination_count = 0
    for seg in segments:
        text = seg.text
        words = text.split()

        # Check for excessive repetition (same word 5+ times in a row)
        if len(words) >= 5:
            for i in range(len(words) - 4):
                if all(words[i] == words[i+j] for j in range(5)):
                    hallucination_count += 1

    return hallucination_count


def test_audio_file(audio_path, model):
    """Test a single audio file with optimal settings."""

    print(f"\n{'='*60}")
    print(f"Testing: {audio_path.name}")
    print(f"{'='*60}")

    # Get file size
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # Transcribe
    transcribe_start = time.time()

    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=7,  # Optimal
        vad_filter=True,
        vad_parameters={
            "threshold": 0.7,  # Strict
            "min_speech_duration_ms": 500,
            "min_silence_duration_ms": 3000
        }
    )

    segments_list = list(segments)
    transcribe_time = time.time() - transcribe_start

    # Calculate metrics
    if segments_list:
        audio_duration = segments_list[-1].end
        rtf = transcribe_time / audio_duration if audio_duration > 0 else 0
    else:
        audio_duration = 0
        rtf = 0

    hallucinations = count_hallucinations(segments_list)

    # Results
    print(f"✅ Segments: {len(segments_list)}")
    print(f"✅ Duration: {audio_duration:.1f}s ({audio_duration/60:.1f} min)")
    print(f"✅ Processing time: {transcribe_time:.1f}s")
    print(f"✅ RTF: {rtf:.2f}x", end="")
    if rtf < 1:
        print(f" ({1/rtf:.1f}x faster than realtime)")
    else:
        print()
    print(f"✅ Hallucinations: {hallucinations}")
    print(f"✅ Language: {info.language} ({info.language_probability:.1%})")

    return {
        "filename": audio_path.name,
        "file_size_mb": file_size_mb,
        "segments": len(segments_list),
        "audio_duration_s": audio_duration,
        "processing_time_s": transcribe_time,
        "rtf": rtf,
        "hallucinations": hallucinations,
        "language": info.language,
        "language_probability": info.language_probability
    }


def main():
    # Audio files to test
    data_dir = Path("data")

    # Get all MP3 files except the one we already tested extensively
    audio_files = [
        f for f in data_dir.glob("*.mp3")
        if f.name != "mamak session scam.mp3"  # Already tested
    ]

    if not audio_files:
        print("❌ No audio files found in data/ directory")
        return

    print("="*60)
    print("VALIDATION: Optimal faster-whisper Settings")
    print("="*60)
    print(f"Testing {len(audio_files)} audio files")
    print()
    print("Configuration:")
    print("  - Model: base")
    print("  - Beam size: 7")
    print("  - VAD threshold: 0.7 (strict)")
    print("  - Min speech duration: 500ms")
    print("  - Min silence duration: 3000ms")
    print()

    # Load model once
    print("Loading faster-whisper model...")
    model_start = time.time()
    model = WhisperModel("base", device="cpu", compute_type="int8")
    model_load_time = time.time() - model_start
    print(f"✅ Model loaded in {model_load_time:.2f}s")

    # Test each file
    results = []

    for audio_path in sorted(audio_files):
        try:
            result = test_audio_file(audio_path, model)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {audio_path.name}: {e}")
            continue

    # Save results
    output_dir = Path("outputs/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "total_files_tested": len(results),
            "model_load_time_s": model_load_time,
            "results": results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Files tested: {len(results)}")
    print()
    print(f"{'File':<35} {'Halluc':<8} {'Segments':<10} {'RTF':<10}")
    print("-"*60)

    total_hallucinations = 0
    for r in results:
        print(f"{r['filename']:<35} {r['hallucinations']:<8} {r['segments']:<10} {r['rtf']:<10.2f}")
        total_hallucinations += r['hallucinations']

    print()
    avg_hallucinations = total_hallucinations / len(results) if results else 0
    print(f"📊 Average hallucinations per file: {avg_hallucinations:.1f}")
    print(f"📊 Total hallucinations: {total_hallucinations}")
    print()
    print(f"✅ Summary saved: {summary_path}")

    if avg_hallucinations <= 2:
        print("\n🎉 Excellent! Optimal settings work consistently across files!")
    elif avg_hallucinations <= 5:
        print("\n✅ Good! Settings perform well across different audio files.")
    else:
        print("\n⚠️  Some files have higher hallucination counts - may need per-file tuning.")


if __name__ == "__main__":
    main()
