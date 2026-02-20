#!/usr/bin/env python3
"""
Quick verification that parameters are now being passed to HuggingFace models.
Runs 2 tests with different beam sizes to confirm different outputs.
"""

import sys
from pathlib import Path

# Add repository root to path so package imports work
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.transcribe import transcribe
import json

def main():
    audio_path = Path("data/mamak session scam.mp3")

    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return 1

    print("🔍 Verification Test: Checking if parameters affect output\n")
    print("=" * 80)

    # Test 1: Low beam size
    print("\n📊 Test 1: beam_size=1 (fast, lower quality)")
    print("-" * 80)
    out1 = Path("outputs/verify_test/beam1")
    out1.parent.mkdir(parents=True, exist_ok=True)

    json1, txt1, metrics1 = transcribe(
        audio_path=audio_path,
        out_base=out1,
        model_name="mesolitica/malaysian-whisper-base",
        language="auto",
        beam_size=1,  # Very low
        temperature=0.0,
    )

    # Test 2: High beam size
    print("\n📊 Test 2: beam_size=10 (slower, higher quality)")
    print("-" * 80)
    out2 = Path("outputs/verify_test/beam10")

    json2, txt2, metrics2 = transcribe(
        audio_path=audio_path,
        out_base=out2,
        model_name="mesolitica/malaysian-whisper-base",
        language="auto",
        beam_size=10,  # High
        temperature=0.0,
    )

    # Compare results
    print("\n" + "=" * 80)
    print("📈 COMPARISON RESULTS")
    print("=" * 80)

    with open(json1, 'r') as f:
        result1 = json.load(f)
    with open(json2, 'r') as f:
        result2 = json.load(f)

    segments1 = len(result1['segments'])
    segments2 = len(result2['segments'])
    text1 = result1['text']
    text2 = result2['text']

    print(f"\nTest 1 (beam=1):")
    print(f"  - Segments: {segments1}")
    print(f"  - Processing time: {metrics1['processing_time_s']:.2f}s")
    print(f"  - Text length: {len(text1)} chars")
    print(f"  - First 100 chars: {text1[:100]}...")

    print(f"\nTest 2 (beam=10):")
    print(f"  - Segments: {segments2}")
    print(f"  - Processing time: {metrics2['processing_time_s']:.2f}s")
    print(f"  - Text length: {len(text2)} chars")
    print(f"  - First 100 chars: {text2[:100]}...")

    # Check if outputs are different
    print("\n" + "=" * 80)
    if text1 == text2:
        print("❌ FAILED: Outputs are IDENTICAL - parameters still not working!")
        print("   The beam_size parameter is not affecting the output.")
        return 1
    else:
        print("✅ SUCCESS: Outputs are DIFFERENT - parameters are now working!")
        print(f"   Text diff: {abs(len(text1) - len(text2))} chars")
        print(f"   Segment diff: {abs(segments1 - segments2)} segments")
        print(f"   Time diff: {abs(metrics1['processing_time_s'] - metrics2['processing_time_s']):.2f}s")
        return 0

if __name__ == "__main__":
    sys.exit(main())
