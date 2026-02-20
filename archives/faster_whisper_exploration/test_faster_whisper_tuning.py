#!/usr/bin/env python3
"""
Practical faster-whisper parameter tuning.
Run focused tests to improve accuracy without extreme performance costs.
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


def run_test(audio_path, test_id, description, model_size="base", beam_size=5,
             temperature=None, vad_params=None, device="cpu", compute_type="int8"):
    """
    Run a single faster-whisper test with specific parameters.
    """
    print("\n" + "="*60)
    print(f"TEST: {test_id} - {description}")
    print("="*60)
    print(f"Model: {model_size}")
    print(f"Beam size: {beam_size}")
    print(f"Temperature: {temperature if temperature is not None else 'default (fallback array)'}")
    if vad_params:
        print(f"VAD threshold: {vad_params.get('threshold', 0.5)}")
        print(f"Min speech duration: {vad_params.get('min_speech_duration_ms', 250)}ms")
        print(f"Min silence duration: {vad_params.get('min_silence_duration_ms', 2000)}ms")
    print()

    # Load model
    print(f"Loading faster-whisper model ({model_size})...")
    model_start = time.time()
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )
    model_load_time = time.time() - model_start
    print(f"Model loaded in {model_load_time:.2f}s")

    # Transcribe
    print("Transcribing...")
    transcribe_start = time.time()

    # Build transcribe kwargs
    transcribe_kwargs = {
        "language": "en",
        "beam_size": beam_size,
        "vad_filter": True,
    }

    # Only add temperature if explicitly specified
    if temperature is not None:
        transcribe_kwargs["temperature"] = temperature

    if vad_params:
        transcribe_kwargs["vad_parameters"] = vad_params
    else:
        # Default VAD params
        transcribe_kwargs["vad_parameters"] = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 2000
        }

    segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    # Convert generator to list
    segments_list = list(segments)

    transcribe_time = time.time() - transcribe_start
    print(f"Transcription completed in {transcribe_time:.2f}s")

    # Calculate metrics
    if segments_list:
        audio_duration = segments_list[-1].end
        rtf = transcribe_time / audio_duration if audio_duration > 0 else 0
    else:
        audio_duration = 0
        rtf = 0

    hallucinations = count_hallucinations(segments_list)

    # Print results
    print()
    print("📊 RESULTS:")
    print(f"  Segments: {len(segments_list)}")
    print(f"  Hallucinations detected: {hallucinations}")
    print(f"  Processing time: {transcribe_time:.1f}s")
    print(f"  RTF: {rtf:.2f}x")
    if rtf < 1:
        print(f"    → {1/rtf:.1f}x faster than realtime")

    # Save output
    output_dir = Path("outputs/faster_whisper_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{test_id}.json"
    txt_path = output_dir / f"{test_id}.txt"

    # Convert segments to JSON-serializable format
    segments_json = []
    for seg in segments_list:
        segments_json.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "compression_ratio": seg.compression_ratio
        })

    result = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segments_json
    }

    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Save text with timestamps
    lines = []
    for seg in segments_list:
        text = seg.text.strip()
        lines.append(f"[{seg.start:.2f} - {seg.end:.2f}] {text}")

    txt_path.write_text("\n".join(lines), encoding='utf-8')

    # Summary
    summary = {
        "test_id": test_id,
        "description": description,
        "model": model_size,
        "beam_size": beam_size,
        "temperature": temperature if temperature is not None else "default",
        "vad_parameters": vad_params or transcribe_kwargs["vad_parameters"],
        "segments": len(segments_list),
        "hallucinations": hallucinations,
        "audio_duration_s": audio_duration,
        "processing_time_s": transcribe_time,
        "model_load_time_s": model_load_time,
        "rtf": rtf,
        "json_path": str(json_path),
        "txt_path": str(txt_path),
    }

    print(f"  Output: {txt_path}")
    return summary


def main():
    audio_path = Path("data/mamak session scam.mp3")

    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return

    # Test configurations - FIXED: Don't force temperature=0.0!
    tests = [
        {
            "test_id": "FW2_strict_vad",
            "description": "Stricter VAD filtering (fixed)",
            "model_size": "base",
            "beam_size": 5,
            "temperature": None,  # Use default fallback array
            "vad_params": {
                "threshold": 0.7,  # Stricter (was 0.5)
                "min_speech_duration_ms": 500,  # Longer (was 250)
                "min_silence_duration_ms": 3000  # Longer (was 2000)
            }
        },
        {
            "test_id": "FW3_default_verify",
            "description": "Default params (verify baseline match)",
            "model_size": "base",
            "beam_size": 5,
            "temperature": None,  # Use default fallback array
            "vad_params": {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 2000
            }
        },
        {
            "test_id": "FW4_beam7",
            "description": "Modest beam increase (5 → 7) (fixed)",
            "model_size": "base",
            "beam_size": 7,  # Increased from 5
            "temperature": None,  # Use default fallback array
            "vad_params": {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 2000
            }
        }
    ]

    results = []

    print("="*60)
    print("FASTER-WHISPER PARAMETER TUNING")
    print("="*60)
    print(f"Audio: {audio_path}")
    print(f"Running {len(tests)} tests...")
    print()
    print("Baseline (FW1): 119 hallucinations, 191 segments, 441s")
    print()

    try:
        for test_config in tests:
            result = run_test(audio_path, **test_config)
            results.append(result)
            time.sleep(1)  # Brief pause between tests

        # Save summary
        output_dir = Path("outputs/faster_whisper_tuning")
        summary_path = output_dir / "summary.json"

        # Add baseline for comparison
        baseline = {
            "test_id": "FW1_baseline",
            "description": "Baseline (from initial test)",
            "model": "base",
            "beam_size": 5,
            "temperature": "default",
            "vad_parameters": {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 2000
            },
            "segments": 191,
            "hallucinations": 119,
            "processing_time_s": 441.18,
            "rtf": 0.48
        }

        all_results = [baseline] + results

        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Summary saved: {summary_path}")

        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON WITH BASELINE")
        print("="*60)
        print(f"{'Test':<20} {'Halluc':<10} {'Segments':<10} {'Time':<10} {'RTF':<10}")
        print("-"*60)
        print(f"{'FW1_baseline':<20} {baseline['hallucinations']:<10} {baseline['segments']:<10} {baseline['processing_time_s']:<10.1f} {baseline['rtf']:<10.2f}")

        for r in results:
            print(f"{r['test_id']:<20} {r['hallucinations']:<10} {r['segments']:<10} {r['processing_time_s']:<10.1f} {r['rtf']:<10.2f}")

        print()
        print("🔍 ANALYSIS:")
        best_halluc = min(results, key=lambda x: x['hallucinations'])
        print(f"  Best hallucination count: {best_halluc['test_id']} ({best_halluc['hallucinations']} hallucinations)")

        if best_halluc['hallucinations'] < baseline['hallucinations']:
            improvement = ((baseline['hallucinations'] - best_halluc['hallucinations']) / baseline['hallucinations']) * 100
            print(f"  Improvement: {improvement:.1f}% reduction vs baseline")

        print()
        print("✅ First 3 tests complete. Ready for diagnosis and next steps.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
