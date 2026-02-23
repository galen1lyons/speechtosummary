"""
Improved transcription module with comprehensive logging, error handling, and ASR metrics.

This module provides Whisper-based transcription with:
- Detailed progress logging
- ASR metrics (WER, CER, RTF)
- Robust error handling
- Performance measurement
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import whisper

from .evaluation.asr_metrics import RTFTimer, calculate_rtf
from .config import WhisperConfig
from .exceptions import AudioFileError, ModelLoadError, TranscriptionError
from .logger import get_logger
from .utils import (
    format_duration,
    get_file_size_mb,
    parse_device,
    sanitize_filename,
    validate_audio_file,
)

logger = get_logger(__name__)


def save_text_with_timestamps(segments: list, txt_path: Path) -> None:
    """
    Save transcript with timestamps to text file.
    
    Args:
        segments: List of segment dictionaries from Whisper
        txt_path: Path to output text file
    """
    lines = []
    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        lines.append(f"[{start:0.2f} - {end:0.2f}] {text}")
    
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug(f"Saved timestamped transcript: {txt_path}")


def resolve_out_base(
    audio_path: Path,
    out_arg: Optional[str],
    out_dir_arg: Optional[str]
) -> Path:
    """
    Resolve output base path from arguments.
    
    Args:
        audio_path: Input audio file path
        out_arg: Explicit output base path (without extension)
        out_dir_arg: Output directory (uses audio filename)
        
    Returns:
        Resolved output base path
    """
    if out_arg:
        out_base = Path(out_arg).expanduser().resolve()
        if out_base.suffix:
            out_base = out_base.with_suffix("")
        out_base.parent.mkdir(parents=True, exist_ok=True)
        return out_base
    
    out_dir = Path(out_dir_arg or "outputs").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = sanitize_filename(audio_path.stem)
    return out_dir / safe_stem


def get_audio_duration(result: Dict) -> float:
    """
    Extract audio duration from Whisper result.
    
    Args:
        result: Whisper transcription result
        
    Returns:
        Audio duration in seconds
    """
    segments = result.get("segments", [])
    if segments:
        return segments[-1].get("end", 0.0)
    return 0.0


def transcribe(
    audio_path: Path,
    out_base: Path,
    model_name: str = "base",
    language: str = "auto",
    device: str = "auto",
    beam_size: int = 5,
    temperature: float = 0.0,
    initial_prompt: Optional[str] = None,
    verbose: bool = False,
    compression_ratio_threshold: Optional[float] = None,
    logprob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
) -> Tuple[Path, Path, Dict]:
    """
    Transcribe audio file using Whisper with comprehensive logging and metrics.

    Args:
        audio_path: Path to input audio file
        out_base: Output base path (without extension)
        model_name: Whisper model size (tiny, base, small, medium, large)
        language: Language code or 'auto' for detection
        device: Device to use ('auto', 'cpu', 'cuda', 'cuda:0')
        beam_size: Beam size for decoding (higher = more accurate, slower)
        temperature: Sampling temperature (0.0 = deterministic)
        initial_prompt: Optional prompt to guide transcription
        verbose: Whether to enable verbose logging
        compression_ratio_threshold: Hallucination detection threshold (default: 2.4)
        logprob_threshold: Log probability threshold (default: -1.0)
        no_speech_threshold: Silence detection threshold (default: 0.6)

    Returns:
        Tuple of (json_path, txt_path, metrics_dict)

    Raises:
        AudioFileError: If audio file is invalid
        ModelLoadError: If model loading fails
        TranscriptionError: If transcription fails
    """
    # Validate input
    validate_audio_file(audio_path)
    
    file_size = get_file_size_mb(audio_path)
    logger.info(f"Transcribing: {audio_path.name} ({file_size:.2f} MB)")
    logger.info(f"Model: {model_name}, Language: {language}, Device: {device}")
    
    # Parse device
    try:
        device = parse_device(device)
        logger.debug(f"Using device: {device}")
    except Exception as e:
        raise ModelLoadError(f"Failed to parse device '{device}': {e}")
    
    # Load model
    model_load_start = time.time()
    is_hf_model = "/" in model_name

    try:
        if is_hf_model:
            logger.info(f"Loading HuggingFace model '{model_name}' via pipeline...")
            from transformers import pipeline
            import torch
            
            # Determine device ID for pipeline
            device_id = -1 # default cpu
            if device and "cuda" in device.lower():
                try:
                    device_id = int(device.split(":")[-1]) if ":" in device else 0
                except:
                    device_id = 0
            
            # Create pipe with long-form support
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=device_id,
                chunk_length_s=30,  # Enable long-form
            )
        else:
            logger.info(f"Loading OpenAI Whisper model '{model_name}'...")
            if device and device.lower() != "auto":
                model = whisper.load_model(model_name, device=device)
            else:
                model = whisper.load_model(model_name)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model '{model_name}': {e}")
    
    model_load_time = time.time() - model_load_start
    logger.info(f"Model loaded in {model_load_time:.2f}s")
    
    # Transcribe with timing
    logger.info("Starting transcription...")
    transcription_start = time.time()
    
    try:
        if is_hf_model:
            # Run pipeline with generation parameters
            generate_kwargs = {
                "return_timestamps": True,
                "num_beams": beam_size,
            }

            # Language setting
            if language and language.lower() != "auto":
                generate_kwargs["language"] = language

            # Temperature and sampling
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["do_sample"] = True

            # Initial prompt (if supported by the model)
            if initial_prompt:
                generate_kwargs["prompt"] = initial_prompt
                logger.info(f"Using initial prompt: {initial_prompt[:50]}...")

            # Advanced Whisper parameters (may not be supported by all HF models)
            # Log a warning that these might be ignored
            if compression_ratio_threshold is not None or logprob_threshold is not None or no_speech_threshold is not None:
                logger.warning(
                    "Advanced Whisper parameters (compression_ratio_threshold, logprob_threshold, "
                    "no_speech_threshold) may not be supported by HuggingFace pipeline models"
                )

            logger.info(f"HF Pipeline params: beam_size={beam_size}, temperature={temperature}")

            # Transcribe (pipeline handles librosa/loading internally if given path)
            pipe_result = pipe(str(audio_path), return_timestamps=True, generate_kwargs=generate_kwargs)
            
            # Normalize to Whisper segments
            raw_segments = pipe_result.get("chunks", [])
            segments = []
            for seg in raw_segments:
                ts = seg.get("timestamp", (0.0, 0.0))
                segments.append({
                    "start": ts[0] if ts[0] is not None else 0.0,
                    "end": ts[1] if ts[1] is not None else (ts[0] if ts[0] is not None else 0.0),
                    "text": seg.get("text", "").strip(),
                })
            
            result = {
                "text": pipe_result.get("text", "").strip(),
                "segments": segments,
                "language": language if language and language != "auto" else "ms" # Malaysian Whisper is primarily ms
            }
        else:
            # OpenAI Whisper
            options = {
                "task": "transcribe",
                "beam_size": beam_size,
                "temperature": temperature,
                "verbose": verbose,
            }
            if language and language.lower() != "auto":
                options["language"] = language
            if initial_prompt:
                options["initial_prompt"] = initial_prompt

            # Add advanced parameters if provided
            if compression_ratio_threshold is not None:
                options["compression_ratio_threshold"] = compression_ratio_threshold
            if logprob_threshold is not None:
                options["logprob_threshold"] = logprob_threshold
            if no_speech_threshold is not None:
                options["no_speech_threshold"] = no_speech_threshold

            result = model.transcribe(str(audio_path), **options)
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")
    
    transcription_time = time.time() - transcription_start
    
    # Extract metrics
    audio_duration = get_audio_duration(result)
    rtf = calculate_rtf(audio_duration, transcription_time) if audio_duration > 0 else None
    
    logger.info(f"Transcription completed in {transcription_time:.2f}s")
    logger.info(f"Detected language: {result.get('language', 'unknown')}")
    logger.info(f"Segments: {len(result.get('segments', []))}")
    logger.info(f"Audio duration: {format_duration(audio_duration)}")
    
    if rtf:
        logger.info(f"RTF: {rtf:.2f}x realtime")
        if rtf < 1.0:
            logger.info(f"  → Processing is {1/rtf:.1f}x faster than realtime")
        else:
            logger.info(f"  → Processing is {rtf:.1f}x slower than realtime")
    
    # Save outputs
    json_path = out_base.with_suffix(".json")
    txt_path = out_base.with_suffix(".txt")
    
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.debug(f"Saved JSON: {json_path}")
    
    save_text_with_timestamps(result.get("segments", []), txt_path)
    
    # Prepare metrics
    metrics = {
        "audio_duration_s": audio_duration,
        "processing_time_s": transcription_time,
        "model_load_time_s": model_load_time,
        "rtf": rtf,
        "file_size_mb": file_size,
        "model": model_name,
        "language_detected": result.get("language"),
        "num_segments": len(result.get("segments", [])),
    }
    
    logger.info(f"✅ Transcription complete!")
    logger.info(f"   JSON: {json_path}")
    logger.info(f"   Text: {txt_path}")
    
    return json_path, txt_path, metrics


def load_openai_whisper_model(
    model_name: str = "base",
    device: str = "auto",
) -> object:
    """
    Load an OpenAI Whisper or HuggingFace model and return it.

    Handles both standard Whisper model names and HuggingFace model IDs
    (those containing '/') via the transformers pipeline. Returns an opaque
    model object to be passed to transcribe_segments().

    Args:
        model_name: Whisper model name or HuggingFace model ID (containing '/')
        device: Device string ("auto", "cpu", "cuda", "cuda:0")

    Returns:
        Loaded model object (whisper.Model or transformers Pipeline wrapped in a dict)

    Raises:
        ModelLoadError: If model loading fails
    """
    load_start = time.time()
    is_hf_model = "/" in model_name

    try:
        resolved_device = parse_device(device)
    except Exception as e:
        raise ModelLoadError(f"Failed to parse device '{device}': {e}") from e

    try:
        if is_hf_model:
            logger.info(f"Loading HuggingFace model '{model_name}' via pipeline...")
            from transformers import pipeline as hf_pipeline
            device_id = -1
            if resolved_device and "cuda" in resolved_device.lower():
                try:
                    device_id = int(resolved_device.split(":")[-1]) if ":" in resolved_device else 0
                except ValueError:
                    device_id = 0
            model_obj = {
                "_type": "hf",
                "_pipe": hf_pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=device_id,
                    chunk_length_s=30,
                ),
            }
        else:
            logger.info(f"Loading OpenAI Whisper model '{model_name}'...")
            if resolved_device and resolved_device.lower() != "auto":
                model_obj = {"_type": "whisper", "_model": whisper.load_model(model_name, device=resolved_device)}
            else:
                model_obj = {"_type": "whisper", "_model": whisper.load_model(model_name)}
    except Exception as e:
        raise ModelLoadError(f"Failed to load model '{model_name}': {e}") from e

    logger.info(f"Model loaded in {time.time() - load_start:.2f}s")
    return model_obj


def transcribe_segments(
    model: object,
    model_name: str,
    segment_clips: list,
    language: str = "auto",
    beam_size: int = 5,
    temperature: float = 0.0,
    initial_prompt: Optional[str] = None,
) -> list:
    """
    Transcribe multiple pre-sliced audio clips using a pre-loaded model.

    Timestamps in returned segments are absolute (relative to the full
    original audio), not relative to each clip.

    Args:
        model: Pre-loaded model object from load_openai_whisper_model()
        model_name: Model name (used for logging; HF models detected via '/')
        segment_clips: List of (clip_path, original_start, original_end, speaker) tuples
        language: Language code or "auto"
        beam_size: Beam size for decoding
        temperature: Sampling temperature
        initial_prompt: Optional prompt to guide transcription

    Returns:
        List of segment dicts with keys: start, end, text, speaker.
        Timestamps are absolute. List is sorted by start time.

    Raises:
        TranscriptionError: If transcription fails for any segment
    """
    if not segment_clips:
        return []

    is_hf_model = isinstance(model, dict) and model.get("_type") == "hf"
    all_segments = []

    for clip_path, original_start, original_end, speaker in segment_clips:
        try:
            if is_hf_model:
                pipe = model["_pipe"]
                generate_kwargs: Dict = {"return_timestamps": True, "num_beams": beam_size}
                if language and language.lower() != "auto":
                    generate_kwargs["language"] = language
                if temperature > 0:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["do_sample"] = True
                if initial_prompt:
                    generate_kwargs["prompt"] = initial_prompt
                pipe_result = pipe(str(clip_path), return_timestamps=True, generate_kwargs=generate_kwargs)
                raw_chunks = pipe_result.get("chunks", [])
                sub_segs = []
                for chunk in raw_chunks:
                    ts = chunk.get("timestamp", (0.0, 0.0))
                    text = chunk.get("text", "").strip()
                    if text:
                        sub_segs.append({
                            "start": (ts[0] or 0.0),
                            "end": (ts[1] or ts[0] or 0.0),
                            "text": text,
                        })
            else:
                whisper_model = model["_model"]
                options: Dict = {
                    "task": "transcribe",
                    "beam_size": beam_size,
                    "temperature": temperature,
                }
                if language and language.lower() != "auto":
                    options["language"] = language
                if initial_prompt:
                    options["initial_prompt"] = initial_prompt
                result = whisper_model.transcribe(str(clip_path), **options)
                sub_segs = [
                    {"start": s.get("start", 0.0), "end": s.get("end", 0.0), "text": s.get("text", "").strip()}
                    for s in result.get("segments", [])
                    if s.get("text", "").strip()
                ]
        except Exception as e:
            raise TranscriptionError(
                f"Transcription failed for segment [{original_start:.2f}s–{original_end:.2f}s] "
                f"speaker={speaker}: {e}"
            ) from e

        if not sub_segs:
            logger.debug(
                f"No speech detected in segment [{original_start:.2f}s–{original_end:.2f}s] "
                f"speaker={speaker} — skipping"
            )
            continue

        for seg in sub_segs:
            all_segments.append({
                "start": original_start + seg["start"],
                "end": original_start + seg["end"],
                "text": seg["text"],
                "speaker": speaker,
            })

    all_segments.sort(key=lambda s: s["start"])
    return all_segments


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio to text with timestamps using Whisper"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--out", help="Output base path without extension")
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for outputs when --out is not provided",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size: tiny, base, small, medium, large",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code (e.g., en, ms, zh, ja) or 'auto'",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (higher = more accurate but slower)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0.0 = deterministic)",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Initial prompt to guide transcription (helps with accents)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Whisper output",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    audio_path = Path(args.audio).expanduser().resolve()
    out_base = resolve_out_base(audio_path, args.out, args.out_dir)
    
    try:
        json_path, txt_path, metrics = transcribe(
            audio_path,
            out_base,
            args.model,
            args.language,
            args.device,
            args.beam_size,
            args.temperature,
            args.initial_prompt,
            args.verbose,
        )
        
        print(f"\n✅ Success!")
        print(f"Transcript JSON: {json_path}")
        print(f"Transcript text: {txt_path}")
        print(f"\nMetrics:")
        print(f"  Audio duration: {metrics['audio_duration_s']:.1f}s")
        print(f"  Processing time: {metrics['processing_time_s']:.1f}s")
        rtf = metrics['rtf']
        rtf_str = f"{rtf:.2f}x realtime" if rtf is not None else "N/A (no audio)"
        print(f"  RTF: {rtf_str}")
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise


if __name__ == "__main__":
    main()
