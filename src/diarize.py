"""
Speaker Diarization Module

This module provides speaker diarization functionality using pyannote.audio.
It identifies different speakers in audio recordings and assigns speaker labels
to transcript segments.

Key Features:
- Automatic speaker detection
- Configurable min/max speaker constraints
- Integration with Whisper transcripts
- Timestamp-based speaker assignment

Author: Meeting Transcription System
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from .exceptions import DiarizationError
from .logger import get_logger
from .utils import validate_audio_file

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


@dataclass
class SpeakerSegment:
    """Represents a segment of audio attributed to a specific speaker."""
    
    start: float  # Start time in seconds
    end: float    # End time in seconds
    speaker: str  # Speaker label (e.g., "Speaker 1")
    
    def __repr__(self) -> str:
        return f"SpeakerSegment({self.speaker}, {self.start:.2f}s-{self.end:.2f}s)"


@dataclass
class TranscriptSegment:
    """Represents a transcript segment with optional speaker attribution."""
    
    start: float           # Start time in seconds
    end: float             # End time in seconds
    text: str              # Transcript text
    speaker: Optional[str] = None  # Speaker label
    
    def __repr__(self) -> str:
        speaker_str = f", {self.speaker}" if self.speaker else ""
        return f"TranscriptSegment({self.start:.2f}s-{self.end:.2f}s{speaker_str})"


def diarize_audio(
    audio_path: Path,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[SpeakerSegment]:
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_path: Path to audio file
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        hf_token: Hugging Face token (optional, will use HF_TOKEN env var if not provided)
    
    Returns:
        List of SpeakerSegment objects with speaker labels and timestamps
    
    Raises:
        AudioFileError: If audio file is invalid
        DiarizationError: If diarization fails
    
    Example:
        >>> segments = diarize_audio(Path("meeting.mp3"), min_speakers=2, max_speakers=5)
        >>> for seg in segments:
        ...     print(f"{seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
    """
    logger.info(f"Starting speaker diarization: {audio_path.name}")
    
    # Validate audio file
    validate_audio_file(audio_path)
    
    # Get HF token
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise DiarizationError(
            "Hugging Face token not found. Please set HF_TOKEN environment variable "
            "or provide it as an argument. See HF_TOKEN_SETUP.md for instructions."
        )
    
    try:
        # Compatibility fix for newer torchaudio versions and older pyannote.audio
        import torchaudio
        if not hasattr(torchaudio, "set_audio_backend"):
            logger.debug("Monkey-patching torchaudio.set_audio_backend for compatibility")
            torchaudio.set_audio_backend = lambda x: None
            
        # Import pyannote here to give better error messages
        from pyannote.audio import Pipeline
        
        logger.info("Loading pyannote.audio diarization pipeline...")
        
        # Load pre-trained pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token
        )
        
        logger.info("Pipeline loaded successfully")
        
        # Configure speaker constraints
        diarization_kwargs = {}
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = min_speakers
            logger.info(f"Min speakers: {min_speakers}")
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = max_speakers
            logger.info(f"Max speakers: {max_speakers}")
        
        # Pre-load audio with torchaudio to bypass torchcodec compatibility issues
        import torchaudio
        logger.info("Loading audio with torchaudio...")
        waveform, sample_rate = torchaudio.load(str(audio_path))
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        # Run diarization
        logger.info("Running diarization (this may take a while)...")
        output = pipeline(audio_input, **diarization_kwargs)

        # pyannote.audio 4.x returns DiarizeOutput; extract the Annotation
        if hasattr(output, "speaker_diarization"):
            diarization = output.speaker_diarization
        else:
            diarization = output

        # Convert to SpeakerSegment objects
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                start=turn.start,
                end=turn.end,
                speaker=f"Speaker {speaker}"
            )
            segments.append(segment)
        
        logger.info(f"Diarization complete: {len(segments)} speaker segments detected")
        
        # Log speaker summary
        unique_speakers = set(seg.speaker for seg in segments)
        logger.info(f"Unique speakers: {len(unique_speakers)} ({', '.join(sorted(unique_speakers))})")
        
        return segments
        
    except ImportError as e:
        raise DiarizationError(
            f"Failed to import pyannote.audio. Please install it: pip install pyannote.audio\n"
            f"Error: {e}"
        )
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}")


def merge_diarization_with_transcript(
    diarization_segments: List[SpeakerSegment],
    transcript_segments: List[Dict],
) -> List[TranscriptSegment]:
    """
    Merge speaker diarization results with Whisper transcript segments.
    
    Assigns speaker labels to transcript segments based on timestamp overlap.
    Uses the speaker with the most overlap for each transcript segment.
    
    Args:
        diarization_segments: List of SpeakerSegment objects from diarization
        transcript_segments: List of Whisper transcript segments (dicts with 'start', 'end', 'text')
    
    Returns:
        List of TranscriptSegment objects with speaker labels
    
    Example:
        >>> diar_segs = [SpeakerSegment(0.0, 5.0, "Speaker 1")]
        >>> trans_segs = [{"start": 0.0, "end": 3.0, "text": "Hello"}]
        >>> merged = merge_diarization_with_transcript(diar_segs, trans_segs)
        >>> print(merged[0].speaker)
        Speaker 1
    """
    logger.info(f"Merging {len(diarization_segments)} diarization segments with {len(transcript_segments)} transcript segments")
    
    merged_segments = []
    
    for trans_seg in transcript_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_text = trans_seg["text"]
        
        # Find speaker with most overlap
        best_speaker = None
        max_overlap = 0.0
        
        for diar_seg in diarization_segments:
            # Calculate overlap
            overlap_start = max(trans_start, diar_seg.start)
            overlap_end = min(trans_end, diar_seg.end)
            overlap = max(0.0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg.speaker
        
        # Create merged segment
        merged_seg = TranscriptSegment(
            start=trans_start,
            end=trans_end,
            text=trans_text,
            speaker=best_speaker
        )
        merged_segments.append(merged_seg)
    
    # Log speaker distribution
    speaker_counts = {}
    for seg in merged_segments:
        if seg.speaker:
            speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1
    
    logger.info("Speaker distribution in transcript:")
    for speaker, count in sorted(speaker_counts.items()):
        logger.info(f"  {speaker}: {count} segments")
    
    return merged_segments


def format_transcript_with_speakers(
    segments: List[TranscriptSegment],
    include_timestamps: bool = True
) -> str:
    """
    Format transcript segments with speaker labels.
    
    Args:
        segments: List of TranscriptSegment objects
        include_timestamps: Whether to include timestamps in output
    
    Returns:
        Formatted transcript string
    
    Example:
        >>> segments = [TranscriptSegment(0.0, 5.0, "Hello", "Speaker 1")]
        >>> print(format_transcript_with_speakers(segments))
        [Speaker 1, 0:00-0:05]: Hello
    """
    lines = []
    
    for seg in segments:
        if include_timestamps:
            start_time = f"{int(seg.start // 60)}:{int(seg.start % 60):02d}"
            end_time = f"{int(seg.end // 60)}:{int(seg.end % 60):02d}"
            speaker = seg.speaker or "Unknown"
            line = f"[{speaker}, {start_time}-{end_time}]: {seg.text}"
        else:
            speaker = seg.speaker or "Unknown"
            line = f"[{speaker}]: {seg.text}"
        
        lines.append(line)
    
    return "\n".join(lines)


def get_speaker_statistics(segments: List[TranscriptSegment]) -> Dict[str, Dict]:
    """
    Calculate statistics about speaker participation.
    
    Args:
        segments: List of TranscriptSegment objects
    
    Returns:
        Dictionary mapping speaker names to statistics (duration, segment_count, word_count)
    
    Example:
        >>> segments = [TranscriptSegment(0.0, 5.0, "Hello world", "Speaker 1")]
        >>> stats = get_speaker_statistics(segments)
        >>> print(stats["Speaker 1"]["duration"])
        5.0
    """
    stats = {}
    
    for seg in segments:
        speaker = seg.speaker or "Unknown"

        if speaker not in stats:
            stats[speaker] = {
                "duration": 0.0,
                "segment_count": 0,
                "word_count": 0
            }

        duration = seg.end - seg.start
        word_count = len(seg.text.split())

        stats[speaker]["duration"] += duration
        stats[speaker]["segment_count"] += 1
        stats[speaker]["word_count"] += word_count

    return stats


# =============================================================================
# RTTM I/O
# =============================================================================

def save_rttm(
    segments: List[SpeakerSegment],
    rttm_path: Path,
    recording_id: str = "recording",
) -> Path:
    """
    Serialise speaker segments to RTTM format.

    RTTM line format:
        SPEAKER <file> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>

    Args:
        segments: List of SpeakerSegment objects from diarization
        rttm_path: Output path for the RTTM file
        recording_id: Recording identifier used in the RTTM file_id field

    Returns:
        Path to the written RTTM file
    """
    lines = []
    for seg in segments:
        duration = seg.end - seg.start
        speaker = seg.speaker.replace(" ", "_")
        lines.append(
            f"SPEAKER {recording_id} 1 {seg.start:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )
    rttm_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.debug(f"Saved RTTM: {rttm_path}")
    return rttm_path


def load_rttm(rttm_path: Path) -> "Annotation":
    """
    Load an RTTM file into a pyannote Annotation object.

    Args:
        rttm_path: Path to RTTM file

    Returns:
        pyannote.core.Annotation with speaker timeline

    Raises:
        DiarizationError: If file cannot be parsed
    """
    try:
        from pyannote.core import Annotation, Segment
    except ImportError as e:
        raise DiarizationError(
            f"pyannote.core is required to load RTTM files: {e}"
        )

    annotation = Annotation()
    try:
        with open(rttm_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                parts = line.split()
                if parts[0] != "SPEAKER" or len(parts) < 8:
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                annotation[Segment(start, start + duration)] = speaker
    except Exception as e:
        raise DiarizationError(f"Failed to parse RTTM file '{rttm_path}': {e}")

    logger.debug(f"Loaded RTTM: {rttm_path} ({len(annotation)} segments)")
    return annotation


# =============================================================================
# DIARIZATION EVALUATION SUBSYSTEM (compatibility re-export)
# =============================================================================

from .evaluation.diarization_metrics import DiarizationMetrics, evaluate_diarization
