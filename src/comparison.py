"""
Model comparison utilities.

Provides functions to compare different transcription models side-by-side.
Supports pairwise (2-model) and multi-model comparison with per-model
device/compute_type settings.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .evaluation.asr_metrics import calculate_wer, calculate_cer, calculate_rtf
from .logger import get_logger
from .utils import format_duration, strip_transcript_timestamps

logger = get_logger(__name__)


@dataclass
class ModelSpec:
    """Specification for a single model in a comparison run."""
    model_name: str
    device: str = "cpu"
    compute_type: str = "int8"
    backend: str = "faster-whisper"


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
    logger.info(f"\nTranscribing with Model 1...")
    model1_result = _transcribe_with_model(
        audio_path, model1_name, model1_backend, device
    )
    results["model1"].update(model1_result)

    # Transcribe with model 2
    logger.info(f"\nTranscribing with Model 2...")
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
    device: str,
    compute_type: str = "int8",
) -> Dict:
    """Transcribe audio with specified model and backend."""

    if backend == "faster-whisper":
        return _transcribe_faster_whisper(audio_path, model_name, device, compute_type)
    elif backend == "openai-whisper":
        return _transcribe_openai_whisper(audio_path, model_name, device)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _transcribe_faster_whisper(
    audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str = "int8",
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
            compute_type=compute_type,
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
        print(f"\nBest WER: {winner}")

    print("=" * 80)


def compare_multiple_models(
    audio_path: Path,
    models: List[ModelSpec],
    reference_text: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Compare N transcription models on the same audio file.

    Args:
        audio_path: Path to audio file
        models: List of ModelSpec configurations to compare
        reference_text: Optional reference transcript for WER/CER
        output_path: Optional path to save JSON results

    Returns:
        Dictionary with all model results and comparison table
    """
    logger.info(f"Multi-model comparison on: {audio_path.name}")
    logger.info(f"  Models: {', '.join(m.model_name for m in models)}")

    results = {
        "audio_file": str(audio_path),
        "timestamp": datetime.now().isoformat(),
        "models": [],
    }

    for i, spec in enumerate(models, 1):
        label = f"{spec.backend}/{spec.model_name} ({spec.device}, {spec.compute_type})"
        logger.info(f"\n[{i}/{len(models)}] Transcribing with {label}...")

        try:
            model_result = _transcribe_with_model(
                audio_path,
                spec.model_name,
                spec.backend,
                spec.device,
                spec.compute_type,
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            model_result = {"error": str(e)}

        entry = {
            "model": spec.model_name,
            "backend": spec.backend,
            "device": spec.device,
            "compute_type": spec.compute_type,
        }
        entry.update(model_result)

        if reference_text and "text" in model_result:
            wer = calculate_wer(reference_text, model_result["text"])
            cer = calculate_cer(reference_text, model_result["text"])
            entry["wer"] = wer[0]
            entry["cer"] = cer[0]
            entry["wer_details"] = {
                "errors": wer[1], "total_words": wer[2],
                "substitutions": wer[3], "deletions": wer[4], "insertions": wer[5],
            }
            entry["cer_details"] = {
                "errors": cer[1], "total_chars": cer[2],
                "substitutions": cer[3], "deletions": cer[4], "insertions": cer[5],
            }

        results["models"].append(entry)

    _print_multi_comparison_table(results)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"\nResults saved to {output_path}")

    return results


def _print_multi_comparison_table(results: Dict) -> None:
    """Print a comparison table for N models."""
    models = results["models"]
    if not models:
        return

    col_width = 20
    label_width = 28
    n = len(models)

    print("\n" + "=" * (label_width + col_width * n + 4))
    print("MODEL SIZE COMPARISON")
    print("=" * (label_width + col_width * n + 4))

    # Header row
    header = f"{'Metric':<{label_width}}"
    for m in models:
        col_label = f"{m['model']} ({m['device']})"
        header += f"  {col_label:<{col_width}}"
    print(header)
    print("-" * (label_width + col_width * n + 4))

    # Compute type
    row = f"{'Compute Type':<{label_width}}"
    for m in models:
        row += f"  {m.get('compute_type', 'n/a'):<{col_width}}"
    print(row)

    # Segments
    row = f"{'Segments':<{label_width}}"
    for m in models:
        val = str(m.get("segments", "ERR"))
        row += f"  {val:<{col_width}}"
    print(row)

    # Audio duration
    row = f"{'Audio Duration':<{label_width}}"
    for m in models:
        val = format_duration(m["audio_duration"]) if "audio_duration" in m else "ERR"
        row += f"  {val:<{col_width}}"
    print(row)

    # Processing time (inference time)
    row = f"{'Inference Time':<{label_width}}"
    for m in models:
        val = f"{m['processing_time']:.1f}s" if "processing_time" in m else "ERR"
        row += f"  {val:<{col_width}}"
    print(row)

    # RTF
    row = f"{'RTF':<{label_width}}"
    for m in models:
        val = f"{m['rtf']:.3f}x" if "rtf" in m else "ERR"
        row += f"  {val:<{col_width}}"
    print(row)

    # WER/CER if available
    if any("wer" in m for m in models):
        print()
        print("Accuracy (vs reference):")

        row = f"{'WER':<{label_width}}"
        for m in models:
            val = f"{m['wer']:.2%}" if "wer" in m else "n/a"
            row += f"  {val:<{col_width}}"
        print(row)

        row = f"{'CER':<{label_width}}"
        for m in models:
            val = f"{m['cer']:.2%}" if "cer" in m else "n/a"
            row += f"  {val:<{col_width}}"
        print(row)

        # Find winner
        scored = [(m["model"], m["wer"]) for m in models if "wer" in m]
        if scored:
            best = min(scored, key=lambda x: x[1])
            print(f"\nBest WER: {best[0]} ({best[1]:.2%})")

    print("=" * (label_width + col_width * n + 4))


def build_comparison_parser() -> argparse.ArgumentParser:
    """Build CLI parser for multi-model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare multiple Whisper model sizes on the same audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare base, small, medium on CPU
  python -m src.comparison --audio data/meeting.mp3 \\
    --models base small medium \\
    --reference outputs/reference/human/dedm_meeting/dedm_meeting_human_plain.txt

  # Include large on GPU
  python -m src.comparison --audio data/meeting.mp3 \\
    --models base small medium large \\
    --devices cpu cpu cpu cuda \\
    --compute-types int8 int8 int8 float16 \\
    --reference outputs/reference/human/dedm_meeting/dedm_meeting_human_plain.txt
        """,
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Whisper model sizes to compare (e.g., base small medium large)",
    )
    parser.add_argument(
        "--devices", nargs="+", default=None,
        help="Device per model (e.g., cpu cpu cpu cuda). Defaults to all cpu.",
    )
    parser.add_argument(
        "--compute-types", nargs="+", default=None,
        help="Compute type per model (e.g., int8 int8 int8 float16). Defaults to all int8.",
    )
    parser.add_argument(
        "--backend", default="faster-whisper",
        choices=["faster-whisper", "openai-whisper"],
        help="Transcription backend (default: faster-whisper)",
    )
    parser.add_argument(
        "--reference", default=None,
        help="Path to reference transcript for WER/CER evaluation",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (default: outputs/comparisons/<timestamp>.json)",
    )
    return parser


def main() -> None:
    """CLI entry point for multi-model comparison."""
    parser = build_comparison_parser()
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    n_models = len(args.models)

    devices = args.devices or ["cpu"] * n_models
    compute_types = args.compute_types or ["int8"] * n_models

    if len(devices) != n_models:
        parser.error(f"--devices must have {n_models} values (one per model), got {len(devices)}")
    if len(compute_types) != n_models:
        parser.error(f"--compute-types must have {n_models} values (one per model), got {len(compute_types)}")

    model_specs = [
        ModelSpec(
            model_name=name,
            device=dev,
            compute_type=ct,
            backend=args.backend,
        )
        for name, dev, ct in zip(args.models, devices, compute_types)
    ]

    reference_text = None
    if args.reference:
        ref_path = Path(args.reference).expanduser().resolve()
        reference_text = strip_transcript_timestamps(
            ref_path.read_text(encoding="utf-8")
        )

    output_path = None
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = "_vs_".join(args.models)
        output_path = Path(f"outputs/comparisons/{ts}__{model_slug}.json")

    compare_multiple_models(
        audio_path=audio_path,
        models=model_specs,
        reference_text=reference_text,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
