"""
Model comparison utilities.

Provides functions to compare different transcription models side-by-side.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from faster_whisper import WhisperModel

from .evaluation.asr_metrics import calculate_wer, calculate_cer, calculate_rtf
from .logger import get_logger
from .utils import format_duration

logger = get_logger(__name__)


def compare_models(
    audio_path: Path,
    model1_name: str = "base",
    model2_name: str = "base",
    model1_backend: str = "faster-whisper",
    model2_backend: str = "faster-whisper",
    device: str = "cpu",
    reference_text: Optional[str] = None,
) -> Dict:
    """
    Compare two transcription models on the same audio.

    Args:
        audio_path: Path to audio file
        model1_name: First model name (e.g., "base", "medium", "large-v3")
        model2_name: Second model name
        model1_backend: Backend for model 1 ("faster-whisper" or "openai-whisper")
        model2_backend: Backend for model 2
        device: Device to use ("cpu" or "cuda")
        reference_text: Optional reference transcript for WER/CER

    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing models on: {audio_path.name}")
    logger.info(f"  Model 1: {model1_backend}/{model1_name}")
    logger.info(f"  Model 2: {model2_backend}/{model2_name}")

    results = {
        "audio_file": str(audio_path),
        "model1": {
            "backend": model1_backend,
            "model": model1_name,
        },
        "model2": {
            "backend": model2_backend,
            "model": model2_name,
        }
    }

    # Transcribe with model 1
    logger.info(f"\n📝 Transcribing with Model 1...")
    model1_result = _transcribe_with_model(
        audio_path, model1_name, model1_backend, device
    )
    results["model1"].update(model1_result)

    # Transcribe with model 2
    logger.info(f"\n📝 Transcribing with Model 2...")
    model2_result = _transcribe_with_model(
        audio_path, model2_name, model2_backend, device
    )
    results["model2"].update(model2_result)

    # Calculate WER/CER if reference provided
    if reference_text:
        logger.info(f"\n📊 Calculating WER/CER against reference...")

        model1_wer = calculate_wer(reference_text, model1_result["text"])
        model1_cer = calculate_cer(reference_text, model1_result["text"])

        model2_wer = calculate_wer(reference_text, model2_result["text"])
        model2_cer = calculate_cer(reference_text, model2_result["text"])

        # Store scalar rates for table rendering and winner logic.
        results["model1"]["wer"] = model1_wer[0]
        results["model1"]["cer"] = model1_cer[0]
        results["model2"]["wer"] = model2_wer[0]
        results["model2"]["cer"] = model2_cer[0]

        # Preserve full metric breakdowns in results payload.
        results["model1"]["wer_details"] = {
            "errors": model1_wer[1],
            "total_words": model1_wer[2],
            "substitutions": model1_wer[3],
            "deletions": model1_wer[4],
            "insertions": model1_wer[5],
        }
        results["model1"]["cer_details"] = {
            "errors": model1_cer[1],
            "total_chars": model1_cer[2],
            "substitutions": model1_cer[3],
            "deletions": model1_cer[4],
            "insertions": model1_cer[5],
        }
        results["model2"]["wer_details"] = {
            "errors": model2_wer[1],
            "total_words": model2_wer[2],
            "substitutions": model2_wer[3],
            "deletions": model2_wer[4],
            "insertions": model2_wer[5],
        }
        results["model2"]["cer_details"] = {
            "errors": model2_cer[1],
            "total_chars": model2_cer[2],
            "substitutions": model2_cer[3],
            "deletions": model2_cer[4],
            "insertions": model2_cer[5],
        }

    # Print comparison table
    _print_comparison_table(results)

    return results


def _transcribe_with_model(
    audio_path: Path,
    model_name: str,
    backend: str,
    device: str
) -> Dict:
    """Transcribe audio with specified model and backend."""

    if backend == "faster-whisper":
        return _transcribe_faster_whisper(audio_path, model_name, device)
    elif backend == "openai-whisper":
        return _transcribe_openai_whisper(audio_path, model_name, device)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _transcribe_faster_whisper(
    audio_path: Path,
    model_name: str,
    device: str
) -> Dict:
    """Transcribe using faster-whisper."""
    from .transcribe_faster import transcribe_faster

    # Use a temporary output path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out_base = Path(tmpdir) / "temp"

        start_time = time.time()
        json_path, txt_path, metrics = transcribe_faster(
            audio_path=audio_path,
            out_base=out_base,
            model_name=model_name,
            device=device,
        )
        processing_time = time.time() - start_time

        # Read transcript text
        text = txt_path.read_text(encoding='utf-8')

        # Read JSON for segments
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            "text": text,
            "segments": len(data.get("segments", [])),
            "processing_time": processing_time,
            "audio_duration": metrics.get("audio_duration_s", 0),
            "rtf": metrics.get("rtf", 0),
            "language": data.get("language", "unknown"),
        }


def _transcribe_openai_whisper(
    audio_path: Path,
    model_name: str,
    device: str
) -> Dict:
    """Transcribe using OpenAI Whisper."""
    from .transcribe import transcribe

    # Use a temporary output path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out_base = Path(tmpdir) / "temp"

        start_time = time.time()
        json_path, txt_path, metrics = transcribe(
            audio_path=audio_path,
            out_base=out_base,
            model_name=model_name,
            device=device,
        )
        processing_time = time.time() - start_time

        # Read transcript text
        text = txt_path.read_text(encoding='utf-8')

        # Read JSON for segments
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            "text": text,
            "segments": len(data.get("segments", [])),
            "processing_time": processing_time,
            "audio_duration": metrics.get("audio_duration_s", 0),
            "rtf": metrics.get("rtf", 0),
            "language": data.get("language", "unknown"),
        }


def _print_comparison_table(results: Dict):
    """Print a clean comparison table."""

    m1 = results["model1"]
    m2 = results["model2"]

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Header
    print(f"{'Metric':<30} {'Model 1':<25} {'Model 2':<25}")
    print("-" * 80)

    # Backend and model
    print(f"{'Backend':<30} {m1['backend']:<25} {m2['backend']:<25}")
    print(f"{'Model':<30} {m1['model']:<25} {m2['model']:<25}")
    print()

    # Performance metrics
    print(f"{'Segments':<30} {m1['segments']:<25} {m2['segments']:<25}")
    print(f"{'Audio Duration':<30} {format_duration(m1['audio_duration']):<25} {format_duration(m2['audio_duration']):<25}")
    print(f"{'Processing Time':<30} {m1['processing_time']:.1f}s{'':<20} {m2['processing_time']:.1f}s")
    print(f"{'RTF (Real-Time Factor)':<30} {m1['rtf']:.2f}x{'':<21} {m2['rtf']:.2f}x")
    print(f"{'Language Detected':<30} {m1['language']:<25} {m2['language']:<25}")

    # WER/CER if available
    if "wer" in m1:
        print()
        print("Accuracy (vs reference):")
        print(f"{'WER (Word Error Rate)':<30} {m1['wer']:.2%}{'':<20} {m2['wer']:.2%}")
        print(f"{'CER (Character Error Rate)':<30} {m1['cer']:.2%}{'':<20} {m2['cer']:.2%}")

        # Winner
        if m1['wer'] < m2['wer']:
            winner = "Model 1"
        elif m2['wer'] < m1['wer']:
            winner = "Model 2"
        else:
            winner = "Tie"
        print(f"\n🏆 Winner (lower WER): {winner}")

    print("=" * 80)
