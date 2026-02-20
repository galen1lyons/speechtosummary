#!/usr/bin/env python3
"""
Unified transcription CLI with subcommands.

This is the main entry point for all transcription tasks.
Replaces multiple single-purpose scripts with one modular tool.

Usage:
    # Transcribe audio
    python scripts/transcribe.py transcribe --audio file.mp3

    # Evaluate against reference
    python scripts/transcribe.py evaluate --hypothesis result.txt --reference human.txt

    # Compare two models
    python scripts/transcribe.py compare --audio file.mp3 --model1 base --model2 large-v3

    # Batch transcribe
    python scripts/transcribe.py batch --input-dir data/ --output-dir outputs/

    # Validate optimal settings
    python scripts/transcribe.py validate --config fw5_optimal
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcribe_faster import transcribe_faster
from src.transcribe import transcribe as transcribe_openai
from src.evaluation.asr_metrics import evaluate_transcription, calculate_wer, calculate_cer
from src.comparison import compare_models
from src.diarize import load_rttm
from src.evaluation.diarization_metrics import evaluate_diarization
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

CONFIGS = {
    'fw5_optimal': {
        'description': 'FW5: Optimal FasterWhisper (beam=7, strict VAD)',
        'model_name': 'base',
        'backend': 'faster-whisper',
        'device': 'cpu',
        'compute_type': 'int8',
        'beam_size': 7,
        'use_optimal_vad': True,
        'vad_threshold': 0.7,
        'min_speech_duration_ms': 500,
        'min_silence_duration_ms': 3000,
    },
    'baseline': {
        'description': 'Baseline FasterWhisper (default settings)',
        'model_name': 'base',
        'backend': 'faster-whisper',
        'device': 'cpu',
        'compute_type': 'int8',
        'beam_size': 5,
        'use_optimal_vad': False,
    },
    'large': {
        'description': 'FasterWhisper large-v3 (multilingual)',
        'model_name': 'large-v3',
        'backend': 'faster-whisper',
        'device': 'cpu',
        'compute_type': 'int8',
        'beam_size': 5,
        'use_optimal_vad': True,
        'vad_threshold': 0.5,
    },
}


# ============================================================================
# SUBCOMMAND: transcribe
# ============================================================================

def cmd_transcribe(args):
    """Transcribe audio file."""
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1

    # Determine output path
    if args.output:
        out_base = Path(args.output).expanduser().resolve()
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_base = output_dir / audio_path.stem

    # Load config if specified
    config = {}
    if args.config:
        if args.config not in CONFIGS:
            logger.error(f"Unknown config: {args.config}")
            logger.info(f"Available configs: {', '.join(CONFIGS.keys())}")
            return 1
        config = CONFIGS[args.config]
        logger.info(f"Using config: {args.config} - {config['description']}")

    # Override with command-line arguments
    backend = args.backend or config.get('backend', 'faster-whisper')
    model_name = args.model or config.get('model_name', 'base')
    device = args.device or config.get('device', 'cpu')
    language = args.language or 'en'

    logger.info(f"Transcribing: {audio_path.name}")
    logger.info(f"Backend: {backend}, Model: {model_name}, Device: {device}")

    # Transcribe based on backend
    if backend == 'faster-whisper':
        json_path, txt_path, metrics = transcribe_faster(
            audio_path=audio_path,
            out_base=out_base,
            model_name=model_name,
            language=language,
            device=device,
            compute_type=config.get('compute_type', 'int8'),
            beam_size=config.get('beam_size', 7),
            use_optimal_vad=config.get('use_optimal_vad', True),
            vad_threshold=config.get('vad_threshold', 0.7),
            min_speech_duration_ms=config.get('min_speech_duration_ms', 500),
            min_silence_duration_ms=config.get('min_silence_duration_ms', 3000),
        )
    elif backend == 'openai-whisper':
        json_path, txt_path, metrics = transcribe_openai(
            audio_path=audio_path,
            out_base=out_base,
            model_name=model_name,
            language=language,
            device=device,
        )
    else:
        logger.error(f"Unknown backend: {backend}")
        return 1

    logger.info(f"\n✅ Transcription complete!")
    logger.info(f"   JSON: {json_path}")
    logger.info(f"   Text: {txt_path}")
    logger.info(f"\n📊 Metrics:")
    logger.info(f"   Duration: {metrics['audio_duration_s']:.1f}s")
    logger.info(f"   Processing time: {metrics['processing_time_s']:.1f}s")
    logger.info(f"   RTF: {metrics['rtf']:.2f}x")

    return 0


# ============================================================================
# SUBCOMMAND: evaluate
# ============================================================================

def cmd_evaluate(args):
    """Evaluate transcription against reference."""
    hypothesis_path = Path(args.hypothesis).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()

    if not hypothesis_path.exists():
        logger.error(f"Hypothesis file not found: {hypothesis_path}")
        return 1

    if not reference_path.exists():
        logger.error(f"Reference file not found: {reference_path}")
        return 1

    # Read files
    hypothesis = hypothesis_path.read_text(encoding='utf-8')
    reference = reference_path.read_text(encoding='utf-8')

    logger.info(f"Evaluating: {hypothesis_path.name}")
    logger.info(f"Reference: {reference_path.name}")

    # Calculate metrics
    metrics = evaluate_transcription(
        reference=reference,
        hypothesis=hypothesis,
    )

    logger.info(f"\n{metrics}")

    # Save metrics if output specified
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"\n✅ Metrics saved: {output_path}")

    return 0


# ============================================================================
# SUBCOMMAND: compare
# ============================================================================

def cmd_compare(args):
    """Compare two models side-by-side."""
    audio_path = Path(args.audio).expanduser().resolve()

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1

    # Read reference if provided
    reference_text = None
    if args.reference:
        reference_path = Path(args.reference).expanduser().resolve()
        if reference_path.exists():
            reference_text = reference_path.read_text(encoding='utf-8')
        else:
            logger.warning(f"Reference file not found: {reference_path}")

    # Compare models
    results = compare_models(
        audio_path=audio_path,
        model1_name=args.model1,
        model2_name=args.model2,
        model1_backend=args.backend1,
        model2_backend=args.backend2,
        device=args.device,
        reference_text=reference_text,
    )

    # Save results if output specified
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ Comparison saved: {output_path}")

    return 0


# ============================================================================
# SUBCOMMAND: batch
# ============================================================================

def cmd_batch(args):
    """Batch transcribe multiple files."""
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f'*{ext}'))

    if not audio_files:
        logger.error(f"No audio files found in: {input_dir}")
        return 1

    logger.info(f"Found {len(audio_files)} audio files")

    # Load config if specified
    config = {}
    if args.config:
        if args.config not in CONFIGS:
            logger.error(f"Unknown config: {args.config}")
            return 1
        config = CONFIGS[args.config]

    backend = args.backend or config.get('backend', 'faster-whisper')
    model_name = args.model or config.get('model_name', 'base')
    device = args.device or config.get('device', 'cpu')

    # Transcribe each file
    for i, audio_path in enumerate(audio_files, 1):
        logger.info(f"\n[{i}/{len(audio_files)}] Processing: {audio_path.name}")

        out_base = output_dir / audio_path.stem

        try:
            if backend == 'faster-whisper':
                transcribe_faster(
                    audio_path=audio_path,
                    out_base=out_base,
                    model_name=model_name,
                    device=device,
                    compute_type=config.get('compute_type', 'int8'),
                    beam_size=config.get('beam_size', 7),
                    use_optimal_vad=config.get('use_optimal_vad', True),
                    vad_threshold=config.get('vad_threshold', 0.7),
                )
            elif backend == 'openai-whisper':
                transcribe_openai(
                    audio_path=audio_path,
                    out_base=out_base,
                    model_name=model_name,
                    device=device,
                )
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_path.name}: {e}")
            continue

    logger.info(f"\n✅ Batch transcription complete!")
    logger.info(f"   Output directory: {output_dir}")

    return 0


# ============================================================================
# SUBCOMMAND: validate
# ============================================================================

def cmd_validate(args):
    """Validate configuration on test files."""
    # This is similar to validate_optimal_settings.py
    # For now, just show available configs

    logger.info("Available configurations:")
    for name, config in CONFIGS.items():
        logger.info(f"\n  {name}:")
        logger.info(f"    {config['description']}")
        logger.info(f"    Backend: {config['backend']}")
        logger.info(f"    Model: {config['model_name']}")
        if 'beam_size' in config:
            logger.info(f"    Beam size: {config['beam_size']}")
        if 'vad_threshold' in config:
            logger.info(f"    VAD threshold: {config['vad_threshold']}")

    logger.info(f"\nTo use a config:")
    logger.info(f"  python scripts/transcribe.py transcribe --config {list(CONFIGS.keys())[0]} --audio file.mp3")

    return 0


# ============================================================================
# SUBCOMMAND: diarize-evaluate
# ============================================================================

def cmd_diarize_evaluate(args):
    """Evaluate diarization output against a reference RTTM."""
    hypothesis_path = Path(args.hypothesis).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()

    if not hypothesis_path.exists():
        logger.error(f"Hypothesis RTTM not found: {hypothesis_path}")
        return 1

    if not reference_path.exists():
        logger.error(f"Reference RTTM not found: {reference_path}")
        return 1

    logger.info(f"Hypothesis: {hypothesis_path.name}")
    logger.info(f"Reference:  {reference_path.name}")
    logger.info(f"Collar:     {args.collar}s")
    logger.info(f"Skip overlap: {args.skip_overlap}")

    reference = load_rttm(reference_path)
    hypothesis = load_rttm(hypothesis_path)

    metrics = evaluate_diarization(
        reference=reference,
        hypothesis=hypothesis,
        collar=args.collar,
        skip_overlap=args.skip_overlap,
    )

    logger.info(f"\n{metrics}")

    if args.output:
        import json
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics.to_dict(), indent=2),
            encoding="utf-8"
        )
        logger.info(f"\n✅ Metrics saved: {output_path}")

    return 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified transcription CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Subcommand to run')

    # ========== SUBCOMMAND: transcribe ==========
    transcribe_parser = subparsers.add_parser(
        'transcribe',
        help='Transcribe audio file'
    )
    transcribe_parser.add_argument('--audio', required=True, help='Audio file path')
    transcribe_parser.add_argument('--output', help='Output base path (without extension)')
    transcribe_parser.add_argument('--output-dir', default='outputs', help='Output directory (default: outputs)')
    transcribe_parser.add_argument('--config', choices=list(CONFIGS.keys()), help='Preset configuration')
    transcribe_parser.add_argument('--backend', choices=['faster-whisper', 'openai-whisper'], help='Backend to use')
    transcribe_parser.add_argument('--model', help='Model name (base, medium, large-v3, etc.)')
    transcribe_parser.add_argument('--language', default='en', help='Language code (default: en)')
    transcribe_parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use')

    # ========== SUBCOMMAND: evaluate ==========
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate transcription against reference'
    )
    evaluate_parser.add_argument('--hypothesis', required=True, help='Hypothesis transcript file')
    evaluate_parser.add_argument('--reference', required=True, help='Reference transcript file')
    evaluate_parser.add_argument('--output', help='Output metrics file (JSON)')

    # ========== SUBCOMMAND: compare ==========
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare two models side-by-side'
    )
    compare_parser.add_argument('--audio', required=True, help='Audio file path')
    compare_parser.add_argument('--model1', default='base', help='First model name (default: base)')
    compare_parser.add_argument('--model2', default='large-v3', help='Second model name (default: large-v3)')
    compare_parser.add_argument('--backend1', default='faster-whisper', help='Backend for model 1')
    compare_parser.add_argument('--backend2', default='faster-whisper', help='Backend for model 2')
    compare_parser.add_argument('--reference', help='Reference transcript for WER/CER')
    compare_parser.add_argument('--device', default='cpu', help='Device to use (default: cpu)')
    compare_parser.add_argument('--output', help='Output comparison file (JSON)')

    # ========== SUBCOMMAND: batch ==========
    batch_parser = subparsers.add_parser(
        'batch',
        help='Batch transcribe multiple files'
    )
    batch_parser.add_argument('--input-dir', required=True, help='Input directory with audio files')
    batch_parser.add_argument('--output-dir', required=True, help='Output directory for transcripts')
    batch_parser.add_argument('--config', choices=list(CONFIGS.keys()), help='Preset configuration')
    batch_parser.add_argument('--backend', choices=['faster-whisper', 'openai-whisper'], help='Backend to use')
    batch_parser.add_argument('--model', help='Model name')
    batch_parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use')

    # ========== SUBCOMMAND: validate ==========
    validate_parser = subparsers.add_parser(
        'validate',
        help='Show available configurations'
    )

    # ========== SUBCOMMAND: diarize-evaluate ==========
    diarize_eval_parser = subparsers.add_parser(
        'diarize-evaluate',
        help='Evaluate diarization output against a reference RTTM (DER/JER)'
    )
    diarize_eval_parser.add_argument(
        '--hypothesis', required=True,
        help='Path to hypothesis RTTM file (system output)'
    )
    diarize_eval_parser.add_argument(
        '--reference', required=True,
        help='Path to reference RTTM file (ground truth)'
    )
    diarize_eval_parser.add_argument(
        '--collar', type=float, default=0.25,
        help='Forgiveness collar in seconds around speaker boundaries (default: 0.25)'
    )
    diarize_eval_parser.add_argument(
        '--skip-overlap', action='store_true',
        help='Exclude overlapping speech regions from scoring'
    )
    diarize_eval_parser.add_argument(
        '--output',
        help='Save metrics to this JSON file'
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to subcommand handler
    if args.command == 'transcribe':
        return cmd_transcribe(args)
    elif args.command == 'evaluate':
        return cmd_evaluate(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'batch':
        return cmd_batch(args)
    elif args.command == 'validate':
        return cmd_validate(args)
    elif args.command == 'diarize-evaluate':
        return cmd_diarize_evaluate(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
