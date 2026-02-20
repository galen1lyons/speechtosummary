#!/usr/bin/env python3
"""
Test faster-whisper on the mamak session scam audio.

faster-whisper:
- 4x faster than OpenAI Whisper
- Uses CTranslate2 for efficient inference
- Same quality, better performance
- More stable than WhisperX
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


def test_faster_whisper(audio_path, model_size="base", device="cpu", compute_type="int8"):
    """
    Test faster-whisper transcription.

    Args:
        audio_path: Path to audio file
        model_size: tiny, base, small, medium, large
        device: cpu or cuda
        compute_type: int8, float16, float32
    """
    print("="*60)
    print("faster-whisper Transcription Test")
    print("="*60)
    print(f"Audio: {audio_path}")
    print(f"Model: {model_size}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")
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

    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=True,  # Enable VAD filtering
        vad_parameters={
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 2000
        }
    )

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
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2%})")
    print(f"Segments: {len(segments_list)}")
    print(f"Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} min)")
    print(f"Processing time: {transcribe_time:.1f}s")
    print(f"RTF: {rtf:.2f}x")
    if rtf < 1:
        print(f"  → Processing is {1/rtf:.1f}x faster than realtime")
    else:
        print(f"  → Processing is {rtf:.1f}x slower than realtime")
    print(f"Hallucinations detected: {hallucinations}")
    print()

    # Save output
    output_dir = Path("outputs/faster_whisper_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "faster_whisper_result.json"
    txt_path = output_dir / "faster_whisper_result.txt"

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
    print(f"✅ JSON saved: {json_path}")

    # Save text with timestamps
    lines = []
    for seg in segments_list:
        text = seg.text.strip()
        lines.append(f"[{seg.start:.2f} - {seg.end:.2f}] {text}")

    txt_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"✅ Text saved: {txt_path}")

    # Summary
    summary = {
        "model": f"faster-whisper-{model_size}",
        "segments": len(segments_list),
        "audio_duration_s": audio_duration,
        "processing_time_s": transcribe_time,
        "model_load_time_s": model_load_time,
        "rtf": rtf,
        "hallucinations": hallucinations,
        "language": info.language,
        "language_probability": info.language_probability,
        "json_path": str(json_path),
        "txt_path": str(txt_path),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {summary_path}")

    return summary


def main():
    audio_path = Path("data/mamak session scam.mp3")

    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return

    try:
        result = test_faster_whisper(
            audio_path,
            model_size="base",
            device="cpu",
            compute_type="int8"  # Optimized for CPU
        )

        print()
        print("="*60)
        print("COMPARISON WITH BASELINE")
        print("="*60)
        print("Original Whisper baseline:")
        print("  - Hallucinations: 217")
        print("  - Segments: 35")
        print("  - Time: 269s (4.5 min)")
        print("  - RTF: 0.29x")
        print()
        print("faster-whisper:")
        print(f"  - Hallucinations: {result['hallucinations']}")
        print(f"  - Segments: {result['segments']}")
        print(f"  - Time: {result['processing_time_s']:.1f}s ({result['processing_time_s']/60:.1f} min)")
        print(f"  - RTF: {result['rtf']:.2f}x")
        print()

        # Calculate improvements
        time_improvement = ((269 - result['processing_time_s']) / 269) * 100
        if time_improvement > 0:
            print(f"⚡ faster-whisper is {time_improvement:.1f}% faster!")

        if result['hallucinations'] < 217:
            halluc_improvement = ((217 - result['hallucinations']) / 217) * 100
            print(f"🎉 Hallucinations reduced by {halluc_improvement:.1f}%!")
        elif result['hallucinations'] > 217:
            print(f"⚠️ More hallucinations than baseline (+{result['hallucinations'] - 217})")
        else:
            print("📊 Same hallucination count as baseline")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
