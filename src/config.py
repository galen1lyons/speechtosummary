"""
Configuration management for the speech-to-summary system.

Provides centralized, validated configuration using dataclasses.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class WhisperConfig:
    """Configuration for Whisper transcription."""

    # Backend selection: "faster-whisper" or "openai-whisper"
    # faster-whisper is default — empirically better hallucination control and speed
    backend: str = "faster-whisper"

    model_name: str = "base"
    language: str = "auto"
    device: str = "auto"
    beam_size: int = 7
    temperature: float = 0.0
    initial_prompt: Optional[str] = None

    # faster-whisper specific provisions
    compute_type: str = "int8"
    use_optimal_vad: bool = True
    vad_threshold: float = 0.7
    min_speech_duration_ms: int = 500
    min_silence_duration_ms: int = 3000

    # openai-whisper advanced parameters
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    suppress_tokens: str = "-1"

    # Fallback temperatures for difficult audio (openai-whisper)
    temperature_increment_on_fallback: float = 0.2
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    def validate(self) -> None:
        """Validate configuration parameters."""
        valid_backends = ["faster-whisper", "openai-whisper"]
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.backend}. Must be one of {valid_backends}")

        if self.backend == "faster-whisper" and "/" in self.model_name:
            raise ValueError(
                f"HuggingFace model '{self.model_name}' requires backend='openai-whisper'. "
                f"faster-whisper only supports standard model sizes (tiny, base, small, medium, large)."
            )

        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if self.model_name not in valid_models and "/" not in self.model_name:
            raise ValueError(f"Invalid model: {self.model_name}. Must be one of {valid_models} or a Hugging Face model ID (containing '/')")

        if self.beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {self.beam_size}")

        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError(f"temperature must be in [0.0, 1.0], got {self.temperature}")


@dataclass
class SummaryConfig:
    """Configuration for summary generation."""

    # Content type: controls output sections and title
    content_type: str = "general"  # "meeting" | "interview" | "podcast" | "general"

    # Legacy parameters (for backward compatibility)
    max_summary_length: int = 500
    min_summary_length: int = 50

    # AI summarization parameters
    max_length: int = 200
    min_length: int = 50
    
    # Extraction limits
    max_action_items: int = 10
    max_decisions: int = 5
    max_key_points: int = 7
    
    # Keywords for action item detection
    action_keywords: List[str] = field(default_factory=lambda: [
        "need to", "should", "must", "will", "todo", "action item",
        "follow up", "deadline", "by", "complete", "finish"
    ])
    
    # Keywords for decision detection
    decision_keywords: List[str] = field(default_factory=lambda: [
        "decided", "agreed", "conclusion", "resolution", "determined",
        "settled on", "chose", "selected"
    ])
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        valid_types = {"meeting", "interview", "podcast", "general"}
        if self.content_type not in valid_types:
            raise ValueError(f"content_type must be one of {valid_types}")

        if self.max_summary_length < self.min_summary_length:
            raise ValueError(
                f"max_summary_length ({self.max_summary_length}) must be >= "
                f"min_summary_length ({self.min_summary_length})"
            )

        if self.max_length < self.min_length:
            raise ValueError(
                f"max_length ({self.max_length}) must be >= "
                f"min_length ({self.min_length})"
            )


@dataclass
class ASRMetricsConfig:
    """Configuration for ASR evaluation metrics."""
    
    # Reference transcript for WER/CER calculation
    reference_transcript: Optional[str] = None
    
    # Whether to normalize text before comparison
    normalize_text: bool = True
    
    # Whether to remove punctuation for WER calculation
    remove_punctuation: bool = True
    
    # Whether to calculate character-level metrics
    calculate_cer: bool = True
    
    # Whether to calculate word-level metrics
    calculate_wer: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.calculate_cer and not self.calculate_wer:
            raise ValueError("At least one of calculate_cer or calculate_wer must be True")


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization."""
    
    # Whether to enable speaker diarization
    enabled: bool = False
    
    # Minimum number of speakers (optional constraint)
    min_speakers: Optional[int] = None
    
    # Maximum number of speakers (optional constraint)
    max_speakers: Optional[int] = None
    
    # Hugging Face token (will use HF_TOKEN env var if not provided)
    hf_token: Optional[str] = None
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.min_speakers is not None and self.min_speakers < 1:
            raise ValueError(f"min_speakers must be >= 1, got {self.min_speakers}")
        
        if self.max_speakers is not None and self.max_speakers < 1:
            raise ValueError(f"max_speakers must be >= 1, got {self.max_speakers}")
        
        if (self.min_speakers is not None and self.max_speakers is not None
            and self.min_speakers > self.max_speakers):
            raise ValueError(
                f"min_speakers ({self.min_speakers}) must be <= "
                f"max_speakers ({self.max_speakers})"
            )


@dataclass
class PreprocessConfig:
    """Configuration for audio preprocessing before diarization and transcription."""

    # Master switch: if False, skip denoising and normalization (format conversion still runs)
    enabled: bool = True

    # Whether to apply spectral noise reduction via noisereduce
    denoise: bool = True

    # Whether to normalize output volume to a target peak dBFS
    normalize_volume: bool = True

    # Target peak level in dBFS for normalization (-3.0 is a safe default)
    target_peak_dbfs: float = -3.0

    # Use stationary noise estimation (conservative; better for speech with music)
    noise_reduce_stationary: bool = True

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not (-60.0 <= self.target_peak_dbfs <= 0.0):
            raise ValueError(
                f"target_peak_dbfs must be in [-60.0, 0.0], got {self.target_peak_dbfs}"
            )
