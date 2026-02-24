"""
End-to-end pipeline for transcription and summarization.

Combines transcription, summarization, and ASR metrics into a single workflow.
"""
import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .evaluation.asr_metrics import evaluate_transcription
from .config import DiarizationConfig, PreprocessConfig, SummaryConfig, WhisperConfig
from .diarize import (
    TranscriptSegment,
    diarize_audio,
    format_transcript_with_speakers,
    get_speaker_statistics,
    load_rttm,
    merge_diarization_with_transcript,
    save_rttm,
)
from .evaluation.diarization_metrics import evaluate_diarization
from .logger import get_logger
from .preprocess import denoise_audio, slice_segment_to_wav
from .summarize import create_structured_summary, load_transcript
from .transcribe import load_openai_whisper_model, transcribe as transcribe_audio, transcribe_segments
from .transcribe_faster import load_faster_whisper_model, transcribe_faster, transcribe_segments_faster
from .utils import parse_device, sanitize_filename, strip_transcript_timestamps

logger = get_logger(__name__)

# Minimum speaker segment duration to attempt transcription (seconds)
MIN_SEGMENT_DURATION_S = 0.3


def _resolve_battle_root(output_dir: Path, evaluation_enabled: bool) -> Path:
    """Resolve canonical output root for production vs evaluation battles."""
    if output_dir.name in {"runs", "eval"}:
        return output_dir
    if evaluation_enabled:
        return output_dir / "campaigns" / "eval"
    return output_dir / "runs"


def _build_loadout_slug(whisper_config: WhisperConfig, summary_config: SummaryConfig) -> str:
    """Build a concise slug representing key runtime loadout settings."""
    backend_short = "fw" if whisper_config.backend == "faster-whisper" else "ow"
    model_slug = sanitize_filename(whisper_config.model_name).lower()
    compute_slug = sanitize_filename(whisper_config.compute_type).lower()
    content_slug = sanitize_filename(summary_config.content_type).lower()
    return f"{backend_short}-{model_slug}-{compute_slug}-{content_slug}"


def _create_run_dir(
    audio_path: Path,
    output_dir: Path,
    whisper_config: WhisperConfig,
    summary_config: SummaryConfig,
    evaluation_enabled: bool,
    run_name: Optional[str] = None,
) -> tuple[str, Path]:
    """Create a deterministic run directory for all pipeline artifacts."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audio_slug = sanitize_filename(run_name or audio_path.stem).lower()
    loadout_slug = _build_loadout_slug(whisper_config, summary_config)
    run_folder = f"{run_id}__{audio_slug}__{loadout_slug}"
    run_dir = _resolve_battle_root(output_dir, evaluation_enabled=evaluation_enabled) / run_folder
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _to_jsonable_config(config_obj) -> Dict:
    """Convert dataclass-based config object to a JSON-safe dict."""
    return dict(vars(config_obj))


def run_pipeline(
    audio_path: Path,
    output_dir: Path = Path("outputs"),
    whisper_config: Optional[WhisperConfig] = None,
    summary_config: Optional[SummaryConfig] = None,
    diarization_config: Optional[DiarizationConfig] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    reference_transcript: Optional[str] = None,
    reference_rttm_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    cli_command: Optional[str] = None,
) -> Dict:
    """
    Run complete transcription and summarization pipeline.

    Pipeline order (primary path, diarization enabled):
        Preprocess → Diarize → Transcribe per segment → Merge → Summarize

    Fallback path (diarization disabled):
        Preprocess → Transcribe full audio → Summarize

    Args:
        audio_path: Path to input audio file
        output_dir: Directory for outputs
        whisper_config: Whisper transcription configuration
        summary_config: Summary generation configuration
        diarization_config: Speaker diarization configuration
        preprocess_config: Audio preprocessing configuration
        reference_transcript: Optional reference transcript for ASR metrics
        reference_rttm_path: Optional reference RTTM for diarization metrics
        run_name: Optional custom run name for the output folder slug
        cli_command: CLI command string recorded in manifest

    Returns:
        Dictionary with all output paths and metrics
    """
    import time as _time

    logger.info("=" * 70)
    logger.info("MEETING TRANSCRIPTION PIPELINE")
    logger.info("=" * 70)

    # Use default configs if not provided
    if whisper_config is None:
        whisper_config = WhisperConfig()
    if summary_config is None:
        summary_config = SummaryConfig()
    if diarization_config is None:
        diarization_config = DiarizationConfig()
    if preprocess_config is None:
        preprocess_config = PreprocessConfig()

    # Validate all configs
    whisper_config.validate()
    summary_config.validate()
    diarization_config.validate()
    preprocess_config.validate()

    evaluation_enabled = bool(reference_transcript or reference_rttm_path)
    asr_evaluation_enabled = bool(reference_transcript)
    diarization_evaluation_enabled = bool(reference_rttm_path)

    # Prepare output directory and run folder
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = _create_run_dir(
        audio_path=audio_path,
        output_dir=output_dir,
        whisper_config=whisper_config,
        summary_config=summary_config,
        evaluation_enabled=evaluation_enabled,
        run_name=run_name,
    )
    out_base = run_dir / "transcript"
    safe_stem = sanitize_filename(run_name or audio_path.stem)

    # Initialize artifact paths
    diar_metrics_path = None
    rttm_path = None
    stats_path = None
    asr_metrics_path = None
    speaker_transcript_path = None
    speaker_stats = None
    segment_clips_dir = None
    preprocessed_path = run_dir / "preprocessed.wav"

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Battle class: {'evaluation' if evaluation_enabled else 'production'}")

    # ------------------------------------------------------------------ #
    # Step 1: Preprocessing                                                #
    # ------------------------------------------------------------------ #
    logger.info("\n Step 1: Preprocessing")
    logger.info("-" * 70)
    preprocessed_path = denoise_audio(audio_path, preprocessed_path, preprocess_config)
    logger.info(f"Preprocessed audio: {preprocessed_path}")

    # ------------------------------------------------------------------ #
    # PRIMARY PATH: Diarize first, then transcribe per speaker segment    #
    # ------------------------------------------------------------------ #
    if diarization_config.enabled:

        # ---------------------------------------------------------- #
        # Step 2: Diarization                                          #
        # ---------------------------------------------------------- #
        logger.info("\n Step 2: Speaker Diarization")
        logger.info("-" * 70)

        try:
            speaker_segments = diarize_audio(
                audio_path=preprocessed_path,
                min_speakers=diarization_config.min_speakers,
                max_speakers=diarization_config.max_speakers,
                hf_token=diarization_config.hf_token,
            )

            rttm_path = run_dir / "diarization.rttm"
            save_rttm(speaker_segments, rttm_path, recording_id=safe_stem)
            logger.info(f"Diarization complete: {len(speaker_segments)} segments, RTTM saved: {rttm_path}")

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            logger.info("Falling back to full-audio transcription...")
            diarization_config = DiarizationConfig(enabled=False)
            speaker_segments = []

        # ---------------------------------------------------------- #
        # Step 3: Per-segment transcription                            #
        # ---------------------------------------------------------- #
        if diarization_config.enabled and speaker_segments:
            logger.info("\n Step 3: Transcription (per speaker segment)")
            logger.info("-" * 70)

            segment_clips_dir = run_dir / "clips"
            segment_clips_dir.mkdir(exist_ok=True)

            # Build clip list, filtering out very short segments
            segment_clips = []
            skipped = 0
            for i, seg in enumerate(speaker_segments):
                if (seg.end - seg.start) < MIN_SEGMENT_DURATION_S:
                    skipped += 1
                    continue
                clip_path = segment_clips_dir / f"segment_{i:04d}.wav"
                slice_segment_to_wav(preprocessed_path, seg.start, seg.end, clip_path)
                segment_clips.append((clip_path, seg.start, seg.end, seg.speaker))

            if skipped:
                logger.info(f"Skipped {skipped} segments shorter than {MIN_SEGMENT_DURATION_S}s")
            logger.info(f"Transcribing {len(segment_clips)} segments...")

            # Load model once, transcribe all segments
            transcription_start = _time.time()
            device = parse_device(whisper_config.device)
            language = whisper_config.language if whisper_config.language != "auto" else "auto"

            if whisper_config.backend == "faster-whisper":
                model = load_faster_whisper_model(
                    model_name=whisper_config.model_name,
                    device=device,
                    compute_type=whisper_config.compute_type,
                )
                raw_segments = transcribe_segments_faster(
                    model=model,
                    segment_clips=segment_clips,
                    language=language,
                    beam_size=whisper_config.beam_size,
                    use_optimal_vad=whisper_config.use_optimal_vad,
                    vad_threshold=whisper_config.vad_threshold,
                    min_speech_duration_ms=whisper_config.min_speech_duration_ms,
                    min_silence_duration_ms=whisper_config.min_silence_duration_ms,
                )
            else:
                model = load_openai_whisper_model(
                    model_name=whisper_config.model_name,
                    device=whisper_config.device,
                )
                raw_segments = transcribe_segments(
                    model=model,
                    model_name=whisper_config.model_name,
                    segment_clips=segment_clips,
                    language=language,
                    beam_size=whisper_config.beam_size,
                    temperature=whisper_config.temperature,
                    initial_prompt=whisper_config.initial_prompt,
                )

            transcription_time = _time.time() - transcription_start
            audio_duration = raw_segments[-1]["end"] if raw_segments else 0.0
            transcription_metrics = {
                "model": whisper_config.model_name,
                "audio_duration_s": audio_duration,
                "processing_time_s": transcription_time,
                "segments": len(raw_segments),
                "mode": "per_segment_diarized",
            }
            logger.info(
                f"Transcription complete: {len(raw_segments)} sub-segments in {transcription_time:.2f}s"
            )

            # -------------------------------------------------- #
            # Step 4: Merge and save transcript                    #
            # -------------------------------------------------- #
            logger.info("\n Step 4: Merging transcript")
            logger.info("-" * 70)

            # Build transcript JSON compatible with load_transcript() / summarize
            result = {
                "language": whisper_config.language,
                "segments": raw_segments,
            }
            json_path = out_base.with_suffix(".json")
            txt_path = out_base.with_suffix(".txt")

            json_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Timestamped text file
            lines = [
                f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}"
                for seg in raw_segments
            ]
            txt_path.write_text("\n".join(lines), encoding="utf-8")

            # Build TranscriptSegment list and format speaker transcript
            merged_segments = [
                TranscriptSegment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    speaker=s["speaker"],
                )
                for s in raw_segments
            ]
            speaker_transcript = format_transcript_with_speakers(merged_segments)
            speaker_transcript_path = run_dir / "speakers.txt"
            speaker_transcript_path.write_text(speaker_transcript, encoding="utf-8")
            logger.info(f"Speaker transcript saved: {speaker_transcript_path}")

            # Speaker statistics
            speaker_stats = get_speaker_statistics(merged_segments)
            logger.info("\nSpeaker Statistics:")
            for speaker, stats in sorted(speaker_stats.items()):
                logger.info(
                    f"  {speaker}: {stats['duration']:.1f}s, "
                    f"{stats['segment_count']} segments, "
                    f"{stats['word_count']} words"
                )
            stats_path = run_dir / "speaker_stats.json"
            stats_path.write_text(json.dumps(speaker_stats, indent=2), encoding="utf-8")
            logger.info(f"Speaker stats saved: {stats_path}")

            # Diarization evaluation (if reference RTTM provided)
            if reference_rttm_path is not None:
                logger.info("\n Diarization Evaluation")
                logger.info("-" * 70)
                try:
                    reference_annotation = load_rttm(reference_rttm_path)
                    hypothesis_annotation = load_rttm(rttm_path)
                    diarization_metrics = evaluate_diarization(
                        reference=reference_annotation,
                        hypothesis=hypothesis_annotation,
                    )
                    logger.info(f"\n{diarization_metrics}")
                    diar_metrics_path = run_dir / "diarization_metrics.json"
                    diar_metrics_path.write_text(
                        json.dumps(diarization_metrics.to_dict(), indent=2),
                        encoding="utf-8",
                    )
                    logger.info(f"Diarization metrics saved: {diar_metrics_path}")
                except Exception as e:
                    logger.error(f"Diarization evaluation failed: {e}")

        else:
            # Diarization failed or returned no segments — fall through to fallback
            diarization_config = DiarizationConfig(enabled=False)

    # ------------------------------------------------------------------ #
    # FALLBACK PATH: Full-audio transcription (no diarization)            #
    # ------------------------------------------------------------------ #
    if not diarization_config.enabled:
        logger.info("\n Step 2: Transcription (full audio)")
        logger.info("-" * 70)
        logger.info(f"Backend: {whisper_config.backend}")

        if whisper_config.backend == "faster-whisper":
            json_path, txt_path, transcription_metrics = transcribe_faster(
                audio_path=preprocessed_path,
                out_base=out_base,
                model_name=whisper_config.model_name,
                language=whisper_config.language if whisper_config.language != "auto" else "auto",
                device=parse_device(whisper_config.device),
                compute_type=whisper_config.compute_type,
                beam_size=whisper_config.beam_size,
                use_optimal_vad=whisper_config.use_optimal_vad,
                vad_threshold=whisper_config.vad_threshold,
                min_speech_duration_ms=whisper_config.min_speech_duration_ms,
                min_silence_duration_ms=whisper_config.min_silence_duration_ms,
            )
        else:
            json_path, txt_path, transcription_metrics = transcribe_audio(
                audio_path=preprocessed_path,
                out_base=out_base,
                model_name=whisper_config.model_name,
                language=whisper_config.language,
                device=whisper_config.device,
                beam_size=whisper_config.beam_size,
                temperature=whisper_config.temperature,
                initial_prompt=whisper_config.initial_prompt,
            )

    # ------------------------------------------------------------------ #
    # Step N: Summarization                                                #
    # ------------------------------------------------------------------ #
    step_num = 5 if (rttm_path and speaker_transcript_path) else 3
    logger.info(f"\n Step {step_num}: Summarization")
    logger.info("-" * 70)

    transcript = load_transcript(json_path)
    summary = create_structured_summary(transcript, summary_config, content_type=summary_config.content_type)

    summary_path = run_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info(f"Summary saved: {summary_path}")

    # ------------------------------------------------------------------ #
    # Step N+1: ASR Metrics (optional)                                     #
    # ------------------------------------------------------------------ #
    asr_metrics = None
    if reference_transcript:
        step_num_asr = step_num + 1
        logger.info(f"\n Step {step_num_asr}: ASR Metrics")
        logger.info("-" * 70)

        from .summarize import extract_full_text
        hypothesis = extract_full_text(transcript)

        asr_metrics = evaluate_transcription(
            reference=reference_transcript,
            hypothesis=hypothesis,
            audio_duration=transcription_metrics.get("audio_duration_s"),
            processing_time=transcription_metrics.get("processing_time_s"),
        )

        logger.info(f"\n{asr_metrics}")
        asr_metrics_path = run_dir / "asr_metrics.json"
        asr_metrics_path.write_text(
            json.dumps(asr_metrics.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info(f"ASR metrics saved: {asr_metrics_path}")

    # ------------------------------------------------------------------ #
    # Manifest                                                             #
    # ------------------------------------------------------------------ #
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": "2.0",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battle_class": "evaluation" if evaluation_enabled else "production",
        "battlefield": {
            "audio_path": str(audio_path),
            "audio_filename": audio_path.name,
        },
        "commander": {
            "command": cli_command,
            "entrypoint": "src.pipeline.run_pipeline",
        },
        "loadout": {
            "preprocessing": _to_jsonable_config(preprocess_config),
            "transcription": _to_jsonable_config(whisper_config),
            "summarization": _to_jsonable_config(summary_config),
            "diarization": _to_jsonable_config(diarization_config),
        },
        "evaluation": {
            "evaluation_enabled": evaluation_enabled,
            "asr_evaluation_enabled": asr_evaluation_enabled,
            "diarization_evaluation_enabled": diarization_evaluation_enabled,
            "asr_evaluation_executed": bool(asr_metrics_path),
            "diarization_evaluation_executed": bool(diar_metrics_path),
        },
        "references": {
            "reference_transcript_provided": bool(reference_transcript),
            "reference_rttm_path": str(reference_rttm_path) if reference_rttm_path else None,
        },
        "artifacts": {
            "preprocessed_wav": str(preprocessed_path),
            "segment_clips_dir": str(segment_clips_dir) if segment_clips_dir else None,
            "transcript_json": str(json_path),
            "transcript_txt": str(txt_path),
            "summary_md": str(summary_path),
            "speakers_txt": str(speaker_transcript_path) if speaker_transcript_path else None,
            "speaker_stats_json": str(stats_path) if stats_path else None,
            "diarization_rttm": str(rttm_path) if rttm_path else None,
            "asr_metrics_json": str(asr_metrics_path) if asr_metrics_path else None,
            "diarization_metrics_json": str(diar_metrics_path) if diar_metrics_path else None,
        },
        "metrics": {
            "transcription_metrics": transcription_metrics,
            "asr_metrics": asr_metrics.to_dict() if asr_metrics else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Manifest saved: {manifest_path}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Run Directory: {run_dir}")
    logger.info(f"Transcript: {txt_path}")
    if speaker_transcript_path:
        logger.info(f"Speaker Transcript: {speaker_transcript_path}")
    logger.info(f"Summary: {summary_path}")
    if asr_metrics_path:
        logger.info(f"ASR Metrics: {asr_metrics_path}")
    logger.info("=" * 70)

    return {
        "run_id": run_id,
        "battle_class": "evaluation" if evaluation_enabled else "production",
        "run_dir": str(run_dir),
        "json_path": str(json_path),
        "txt_path": str(txt_path),
        "speaker_transcript_path": str(speaker_transcript_path) if speaker_transcript_path else None,
        "speaker_stats": speaker_stats,
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "transcription_metrics": transcription_metrics,
        "asr_metrics": asr_metrics.to_dict() if asr_metrics else None,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="End-to-end meeting transcription and summarization pipeline"
    )
    
    # Input/Output
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output root directory (production: outputs/runs, evaluation: outputs/campaigns/eval)",
    )
    parser.add_argument(
        "--run-name",
        help="Optional custom run name used in the run folder slug",
    )
    
    # Whisper options
    parser.add_argument(
        "--backend",
        default="faster-whisper",
        choices=["faster-whisper", "openai-whisper"],
        help="Transcription backend: faster-whisper (default, recommended) or openai-whisper (required for HuggingFace models)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model: tiny, base, small, medium, large (or HuggingFace model ID with --backend openai-whisper)",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code or 'auto'",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type: int8 (default), float16, float32",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=7,
        help="Beam size for decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Initial prompt for Whisper (openai-whisper backend only)",
    )
    
    # Summary options
    parser.add_argument(
        "--max-summary-length",
        type=int,
        default=500,
        help="Maximum executive summary length",
    )
    parser.add_argument(
        "--content-type",
        default="general",
        choices=["meeting", "interview", "podcast", "general"],
        help="Content type for summarization (default: general)",
    )
    parser.add_argument(
        "--summary-model-path",
        default=None,
        help=(
            "Path to a local Mistral 7B GGUF file for summarization. "
            "If not provided, auto-downloads Q4_K_M (~4.4GB) from HuggingFace Hub on first run."
        ),
    )

    # Diarization options
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable speaker diarization",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers",
    )
    
    # ASR metrics
    parser.add_argument(
        "--reference-transcript",
        help="Path to reference transcript for ASR metrics",
    )

    # Diarization metrics
    parser.add_argument(
        "--reference-rttm",
        help="Path to reference RTTM file for DER/JER evaluation (requires --enable-diarization)",
    )

    # Preprocessing options
    parser.add_argument(
        "--disable-preprocessing",
        action="store_true",
        help="Skip denoising and normalization (format conversion to 16kHz WAV still runs)",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip spectral noise reduction (normalization still runs)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip volume normalization (denoising still runs)",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    # Prepare paths
    audio_path = Path(args.audio).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    
    # Prepare configs
    whisper_config = WhisperConfig(
        backend=args.backend,
        model_name=args.whisper_model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        temperature=args.temperature,
        initial_prompt=args.initial_prompt,
    )
    
    summary_config = SummaryConfig(
        max_summary_length=args.max_summary_length,
        content_type=args.content_type,
        llm_model_path=args.summary_model_path,
    )
    
    diarization_config = DiarizationConfig(
        enabled=args.enable_diarization,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    preprocess_config = PreprocessConfig(
        enabled=not args.disable_preprocessing,
        denoise=not args.no_denoise,
        normalize_volume=not args.no_normalize,
    )

    # Load reference transcript if provided
    reference_transcript = None
    if args.reference_transcript:
        ref_path = Path(args.reference_transcript).expanduser().resolve()
        reference_transcript = strip_transcript_timestamps(
            ref_path.read_text(encoding="utf-8")
        )

    # Resolve reference RTTM path if provided
    reference_rttm_path = None
    if args.reference_rttm:
        reference_rttm_path = Path(args.reference_rttm).expanduser().resolve()

    # Run pipeline
    cli_command = "python -m src.pipeline " + " ".join(shlex.quote(arg) for arg in sys.argv[1:])
    run_pipeline(
        audio_path=audio_path,
        output_dir=output_dir,
        whisper_config=whisper_config,
        summary_config=summary_config,
        diarization_config=diarization_config,
        preprocess_config=preprocess_config,
        reference_transcript=reference_transcript,
        reference_rttm_path=reference_rttm_path,
        run_name=args.run_name,
        cli_command=cli_command,
    )

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
