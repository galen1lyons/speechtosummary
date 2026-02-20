#!/usr/bin/env python3
"""
Final test: Combine best parameters (Strict VAD + Beam 7)
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


def test_combined(audio_path):
    """Test combined best settings: Strict VAD + Beam 7"""

    print("="*60)
    print("FINAL TEST: Strict VAD + Beam 7")
    print("="*60)
    print(f"Audio: {audio_path}")
    print("Model: base")
    print("Beam size: 7")
    print("Temperature: default (fallback array)")
    print("VAD threshold: 0.7 (strict)")
    print("Min speech duration: 500ms")
    print("Min silence duration: 3000ms")
    print()

    # Load model
    print("Loading faster-whisper model (base)...")
    model_start = time.time()
    model = WhisperModel("base", device="cpu", compute_type="int8")
    model_load_time = time.time() - model_start
    print(f"Model loaded in {model_load_time:.2f}s")

    # Transcribe with combined best settings
    print("Transcribing...")
    transcribe_start = time.time()

    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=7,  # Best from FW4
        vad_filter=True,
        vad_parameters={
            "threshold": 0.7,  # Strict from FW2
            "min_speech_duration_ms": 500,
            "min_silence_duration_ms": 3000
        }
        # No temperature specified = uses default fallback
    )

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
    print(f"Hallucinations detected: {hallucinations}")
    print()

    # Save output
    output_dir = Path("outputs/faster_whisper_final")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "FW5_combined_best.json"
    txt_path = output_dir / "FW5_combined_best.txt"

    # Convert segments to JSON
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

    # Save text
    lines = []
    for seg in segments_list:
        text = seg.text.strip()
        lines.append(f"[{seg.start:.2f} - {seg.end:.2f}] {text}")

    txt_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"✅ Text saved: {txt_path}")

    # Summary
    summary = {
        "test_id": "FW5_combined_best",
        "description": "Strict VAD + Beam 7",
        "model": "base",
        "beam_size": 7,
        "temperature": "default",
        "vad_parameters": {
            "threshold": 0.7,
            "min_speech_duration_ms": 500,
            "min_silence_duration_ms": 3000
        },
        "segments": len(segments_list),
        "hallucinations": hallucinations,
        "audio_duration_s": audio_duration,
        "processing_time_s": transcribe_time,
        "model_load_time_s": model_load_time,
        "rtf": rtf,
        "language": info.language,
        "language_probability": info.language_probability,
        "json_path": str(json_path),
        "txt_path": str(txt_path),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {summary_path}")

    # Print comparison
    print()
    print("="*60)
    print("COMPARISON WITH PREVIOUS TESTS")
    print("="*60)
    print(f"{'Test':<25} {'Halluc':<10} {'Segments':<10} {'Time':<10} {'RTF':<10}")
    print("-"*60)
    print(f"{'FW1_baseline':<25} {119:<10} {191:<10} {441.2:<10.1f} {0.48:<10.2f}")
    print(f"{'FW2_strict_vad':<25} {0:<10} {381:<10} {398.8:<10.1f} {0.43:<10.2f}")
    print(f"{'FW4_beam7':<25} {5:<10} {210:<10} {445.6:<10.1f} {0.49:<10.2f}")
    print(f"{'FW5_combined_best':<25} {hallucinations:<10} {len(segments_list):<10} {transcribe_time:<10.1f} {rtf:<10.2f}")
    print()

    if hallucinations < 5:
        print(f"🎉 Best result yet! Only {hallucinations} hallucinations!")
    elif hallucinations == 5:
        print(f"📊 Matched FW4 beam7 ({hallucinations} hallucinations)")
    else:
        print(f"📊 Result: {hallucinations} hallucinations")

    return summary


def main():
    audio_path = Path("data/mamak session scam.mp3")

    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return

    try:
        test_combined(audio_path)
        print()
        print("✅ Final test complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
