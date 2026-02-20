"""
Utility functions for the speech-to-summary system.

Provides common operations like file validation, sanitization, and device detection.
"""
import re
import string
from pathlib import Path
from typing import Optional

import torch

from .exceptions import AudioFileError


def validate_audio_file(audio_path: Path) -> None:
    """
    Validate that the audio file exists and has a supported extension.
    
    Args:
        audio_path: Path to audio file
        
    Raises:
        AudioFileError: If file doesn't exist or has unsupported extension
    """
    if not audio_path.exists():
        raise AudioFileError(f"Audio file not found: {audio_path}")
    
    if not audio_path.is_file():
        raise AudioFileError(f"Path is not a file: {audio_path}")
    
    # Supported audio formats by Whisper/FFmpeg
    supported_extensions = {
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
        ".mp4", ".avi", ".mov", ".mkv", ".webm"
    }
    
    if audio_path.suffix.lower() not in supported_extensions:
        raise AudioFileError(
            f"Unsupported audio format: {audio_path.suffix}. "
            f"Supported formats: {', '.join(sorted(supported_extensions))}"
        )


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Sanitize a filename by replacing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove or replace characters that are problematic in filenames
    # Keep alphanumeric, dots, hyphens, underscores
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", replacement, filename)
    
    # Remove leading/trailing replacement characters
    sanitized = sanitized.strip(replacement)
    
    # Ensure we have at least something
    return sanitized or "output"


def parse_device(device_str: str) -> str:
    """
    Parse device string and return appropriate device for PyTorch/Whisper.
    
    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)
        
    Returns:
        Device string suitable for model loading
    """
    device_str = device_str.lower().strip()
    
    if device_str == "auto":
        # Auto-detect: use CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Install CUDA or use 'cpu'.")
        return device_str
    
    return device_str


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    return file_path.stat().st_size / (1024 * 1024)


def normalize_text(text: str, remove_punctuation: bool = False) -> str:
    """
    Normalize text for comparison in ASR metrics.
    
    Args:
        text: Input text
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Optionally remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())  # Clean up whitespace again
    
    return text


def strip_transcript_timestamps(text: str) -> str:
    """
    Strip timestamp markers from a transcript file before WER/CER evaluation.

    Handles the project's standard timestamped format:
        [12.91 - 15.59] Some spoken text here.
    Also strips inline bracket annotations such as [unintelligible].

    Args:
        text: Raw transcript file contents (may contain timestamp markers)

    Returns:
        Plain text with timestamps and bracket annotations removed,
        ready for ASR metric evaluation.
    """
    lines = []
    for line in text.splitlines():
        # Remove leading segment timestamp  [start - end]
        line = re.sub(r"^\[\d+\.?\d*\s*-\s*\d+\.?\d*\]\s*", "", line)
        # Remove any remaining bracket annotations e.g. [unintelligible]
        line = re.sub(r"\[[^\]]*\]", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2m 30s" or "45.2s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"
