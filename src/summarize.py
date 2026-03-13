"""
Summarization module using Mistral 7B Instruct via llama-cpp-python.

Uses Mistral-7B-Instruct-v0.3 (GGUF, Q4_K_M quantization) for grounded summarization.
The model is prompted to use ONLY information present in the transcript, preventing the
news-article confabulation seen with mT5_multilingual_XLSum (XL-Sum training mismatch).
Falls back to extractive summarization if llama-cpp-python is not installed or model
download fails.

Key features:
- Local summarization via llama-cpp-python (CPU-efficient, no API key needed)
- Auto-downloads Mistral 7B Q4_K_M (~4.4GB) from HuggingFace Hub on first run
- Grounding instruction prompt prevents hallucination
- Hierarchical chunking for long transcripts (4000 words/chunk, 8192-token context)
- Content-type-aware output sections (meeting, interview, podcast, general)
- Malay/Manglish keyword support for action items and decisions
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SummaryConfig
from .exceptions import SummarizationError
from .logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Mistral 7B constants
# ---------------------------------------------------------------------------

_DEFAULT_GGUF_REPO = "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
_DEFAULT_GGUF_FILENAME = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

_MISTRAL_PROMPT_TEMPLATE = (
    "<s>[INST] You are summarizing a corporate meeting transcript about robotics and factory automation. "
    "Key terminology you may encounter:\n"
    "- AMR: Autonomous Mobile Robot\n"
    "- AGV: Automated Guided Vehicle\n"
    "- ROS: Robot Operating System\n"
    "- QGV: QR-code Guided Vehicle\n"
    "- ASRS: Automated Storage and Retrieval System\n"
    "- MES: Manufacturing Execution System\n"
    "- WMS: Warehouse Management System\n"
    "- Fleet management: Software to coordinate multiple robots\n\n"
    "Summarize ONLY what is explicitly said in the transcript below. "
    "Do not add any information, names, or events not present in the transcript. "
    "Do not invent acronyms or rename concepts. "
    "Write 3-5 concise sentences.\n\n"
    "Transcript:\n{chunk}\n\n"
    "Summary: [/INST]"
)


def load_transcript(json_path: Path) -> Dict:
    """
    Load transcript from JSON file.
    
    Args:
        json_path: Path to transcript JSON
        
    Returns:
        Transcript dictionary
        
    Raises:
        SummarizationError: If loading fails
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise SummarizationError(f"Failed to load transcript: {e}")


def extract_full_text(transcript: Dict) -> str:
    """
    Extract full text from transcript.
    
    Args:
        transcript: Transcript dictionary from Whisper
        
    Returns:
        Full transcript text
    """
    # Try to get text from 'text' field first
    if "text" in transcript:
        return transcript["text"].strip()
    
    # Otherwise, concatenate segments
    segments = transcript.get("segments", [])
    texts = [seg.get("text", "").strip() for seg in segments]
    return " ".join(texts)


def chunk_text(text: str, max_words: int = 4000) -> List[str]:
    """
    Split text into chunks for summarization.
    
    Args:
        text: Full text to chunk
        max_words: Maximum words per chunk
        
    Returns:
        List of text chunks
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    
    return chunks


def _try_load_mistral(config: SummaryConfig) -> Optional[Any]:
    """
    Attempt to load Mistral 7B via llama-cpp-python.

    Resolution order for the GGUF file:
      1. config.llm_model_path if provided and the file exists on disk
      2. Auto-download from bartowski/Mistral-7B-Instruct-v0.3-GGUF via huggingface_hub

    Returns:
        Llama instance on success, None if llama_cpp unavailable or load fails.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.warning("llama-cpp-python not installed; falling back to extractive summary")
        return None

    model_path: Optional[str] = config.llm_model_path

    if model_path and not Path(model_path).exists():
        logger.warning(f"Specified llm_model_path not found: {model_path}; will auto-download")
        model_path = None

    if model_path is None:
        try:
            from huggingface_hub import hf_hub_download
            logger.info(f"Downloading {_DEFAULT_GGUF_FILENAME} from {_DEFAULT_GGUF_REPO} ...")
            model_path = hf_hub_download(
                repo_id=_DEFAULT_GGUF_REPO,
                filename=_DEFAULT_GGUF_FILENAME,
            )
            logger.info(f"Model cached at: {model_path}")
        except Exception as e:
            logger.warning(f"Auto-download failed: {e}; falling back to extractive summary")
            return None

    try:
        logger.info(f"Loading Mistral GGUF: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=config.llm_n_ctx,
            n_threads=config.llm_n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        return llm
    except Exception as e:
        logger.warning(f"Failed to load Mistral model: {e}; falling back to extractive summary")
        return None


def generate_mistral_summary(text: str, config: SummaryConfig) -> str:
    """
    Generate summary using Mistral 7B Instruct via llama-cpp-python.

    Chunks text at max_words=4000, summarizes each chunk with a grounding
    instruction prompt, then runs a hierarchical pass if multiple chunks exist.

    Args:
        text: Full transcript text
        config: SummaryConfig with llm_* fields

    Returns:
        Generated summary string, or extractive fallback on failure.
    """
    if not text:
        return ""

    llm = _try_load_mistral(config)
    if llm is None:
        logger.info("Mistral unavailable; using extractive summary")
        return generate_extractive_summary(text)

    chunks = chunk_text(text, max_words=4000)
    logger.info(f"Summarizing {len(chunks)} chunk(s) with Mistral")

    chunk_summaries: List[str] = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"Summarizing chunk {i + 1}/{len(chunks)}")
        prompt = _MISTRAL_PROMPT_TEMPLATE.format(chunk=chunk)
        try:
            output = llm(
                prompt,
                max_tokens=config.llm_max_tokens,
                temperature=config.llm_temperature,
                stop=["</s>", "[INST]"],
                echo=False,
            )
            summary_text = output["choices"][0]["text"].strip()
            if summary_text:
                chunk_summaries.append(summary_text)
        except Exception as e:
            logger.warning(f"Chunk {i + 1} summarization failed: {e}")

    if not chunk_summaries:
        logger.warning("All chunks failed; using extractive summary")
        return generate_extractive_summary(text)

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    logger.info("Running hierarchical summarization pass over chunk summaries")
    combined_intermediate = " ".join(chunk_summaries)
    final_prompt = _MISTRAL_PROMPT_TEMPLATE.format(chunk=combined_intermediate)
    try:
        output = llm(
            final_prompt,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
            stop=["</s>", "[INST]"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        logger.warning(f"Hierarchical pass failed: {e}; returning joined chunk summaries")
        return combined_intermediate


def generate_extractive_summary(text: str, num_sentences: int = 5) -> str:
    """
    Fallback: Generate summary by extracting key sentences.
    
    Args:
        text: Full text
        num_sentences: Number of sentences to extract
        
    Returns:
        Extractive summary
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]
    
    if not sentences:
        return "No summary available."
    
    # Take sentences evenly distributed through the text
    step = max(1, len(sentences) // num_sentences)
    selected = []
    
    for i in range(0, min(len(sentences), num_sentences * step), step):
        if i < len(sentences):
            selected.append(sentences[i])
    
    return ". ".join(selected[:num_sentences]) + "."


def extract_action_items(
    full_text: str,
    keywords: Optional[List[str]] = None,
    max_items: int = 10
) -> List[str]:
    """
    Extract action items from transcript using keyword patterns.
    
    Args:
        full_text: Full transcript text
        keywords: Keywords that indicate action items
        max_items: Maximum number of items to extract
        
    Returns:
        List of action items
    """
    if keywords is None:
        keywords = [
            "need to", "should", "must", "will", "going to",
            "have to", "plan to", "action item", "todo", "task",
            "follow up", "deadline", "by", "complete", "finish", "assign"
        ]
    
    sentences = re.split(r'[.!?]+', full_text)
    action_items = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 15:
            continue
        
        # Check if sentence contains action keywords
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in keywords):
            # Clean up the sentence
            cleaned = sentence.strip()
            if cleaned and cleaned not in action_items:
                action_items.append(cleaned)
    
    return action_items[:max_items]


def extract_decisions(
    full_text: str,
    keywords: Optional[List[str]] = None,
    max_decisions: int = 5
) -> List[str]:
    """
    Extract decisions from transcript using keyword patterns.
    
    Args:
        full_text: Full transcript text
        keywords: Keywords that indicate decisions
        max_decisions: Maximum number of decisions to extract
        
    Returns:
        List of decisions
    """
    if keywords is None:
        keywords = [
            "decided", "agreed", "conclusion", "resolution", "determined",
            "settled on", "chose", "selected", "approved", "confirmed"
        ]
    
    sentences = re.split(r'[.!?]+', full_text)
    decisions = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 15:
            continue
        
        # Check if sentence contains decision keywords
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in keywords):
            # Clean up the sentence
            cleaned = sentence.strip()
            if cleaned and cleaned not in decisions:
                decisions.append(cleaned)
    
    return decisions[:max_decisions]


def extract_key_points(full_text: str, num_points: int = 5) -> List[str]:
    """
    Extract key discussion points from text.
    
    Args:
        full_text: Full transcript text
        num_points: Number of points to extract
        
    Returns:
        List of key points
    """
    sentences = re.split(r'[.!?]+', full_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 30]
    
    if not sentences:
        return ["No key points identified."]
    
    # Look for sentences with important keywords
    important_sentences = []
    importance_keywords = [
        'important', 'key', 'main', 'significant', 'critical', 'essential',
        'discussed', 'mentioned', 'noted', 'highlighted', 'emphasized',
        'focus', 'priority', 'concern', 'issue', 'challenge', 'opportunity'
    ]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in importance_keywords):
            important_sentences.append(sentence)
    
    # If we found enough, use those
    if len(important_sentences) >= num_points:
        return important_sentences[:num_points]
    
    # Otherwise, take evenly distributed sentences
    step = max(1, len(sentences) // num_points)
    key_points = []
    
    for i in range(0, min(len(sentences), num_points * step), step):
        if i < len(sentences):
            key_points.append(sentences[i])
    
    return key_points[:num_points]


_CONTENT_TYPE_TITLES = {
    "meeting":   "# Meeting Summary",
    "interview": "# Interview Summary",
    "podcast":   "# Episode Summary",
    "general":   "# Content Summary",
}


def _build_meeting_sections(full_text: str, config: SummaryConfig) -> List[str]:
    """Build meeting-specific sections: discussion points, action items, decisions."""
    action_keywords = list(config.action_keywords) + ["perlu", "harus", "akan", "rancang"]
    decision_keywords = list(config.decision_keywords) + ["putus", "setuju", "kesimpulan"]

    key_points = extract_key_points(full_text, config.max_key_points)
    action_items = extract_action_items(full_text, action_keywords, config.max_action_items)
    decisions = extract_decisions(full_text, decision_keywords, config.max_decisions)

    lines = ["## Key Discussion Points", ""]
    for point in key_points:
        lines.append(f"- {point}")

    lines.extend(["", "## Action Items", ""])
    if action_items:
        for item in action_items:
            lines.append(f"- [ ] {item}")
    else:
        lines.append("- [ ] No specific action items identified")

    lines.extend(["", "## Decisions Made", ""])
    if decisions:
        for decision in decisions:
            lines.append(f"- {decision}")
    else:
        lines.append("- No specific decisions identified")

    return lines


def _build_interview_sections(full_text: str, config: SummaryConfig) -> List[str]:
    """Build interview-specific sections: main topics, notable insights, key takeaways."""
    topic_keywords = [
        "talk about", "discuss", "mention", "explain", "bincang", "cerita",
        "share", "describe", "cover", "address",
    ]
    insight_keywords = [
        "think", "believe", "feel", "realize", "rasa", "percaya",
        "important", "key", "significant", "interesting", "surprising",
    ]

    sentences = re.split(r'[.!?]+', full_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]

    topics: List[str] = []
    insights: List[str] = []
    for sentence in sentences:
        sl = sentence.lower()
        if len(topics) < config.max_key_points and any(k in sl for k in topic_keywords):
            topics.append(sentence)
        elif len(insights) < config.max_key_points and any(k in sl for k in insight_keywords):
            insights.append(sentence)

    if not topics:
        topics = extract_key_points(full_text, config.max_key_points)

    takeaways = extract_key_points(full_text, 3)

    lines = ["## Main Topics Discussed", ""]
    for topic in topics:
        lines.append(f"- {topic}")

    lines.extend(["", "## Notable Insights", ""])
    if insights:
        for insight in insights:
            lines.append(f"- {insight}")
    else:
        lines.append("- No specific insights identified")

    lines.extend(["", "## Key Takeaways", ""])
    for takeaway in takeaways:
        lines.append(f"- {takeaway}")

    return lines


def _build_podcast_sections(full_text: str, config: SummaryConfig) -> List[str]:
    """Build podcast-specific sections: main discussions, key takeaways, notable moments."""
    main_discussions = extract_key_points(full_text, config.max_key_points)
    takeaways = extract_key_points(full_text, 3)

    moment_keywords = [
        "interesting", "surprising", "funny", "unexpected", "highlight",
        "moment", "amazing", "incredible", "fascinating",
    ]
    sentences = re.split(r'[.!?]+', full_text)
    notable: List[str] = []
    for sentence in sentences:
        if len(notable) >= 3:
            break
        if sentence.strip() and any(k in sentence.lower() for k in moment_keywords):
            notable.append(sentence.strip())

    lines = ["## Main Discussions", ""]
    for discussion in main_discussions:
        lines.append(f"- {discussion}")

    lines.extend(["", "## Key Takeaways", ""])
    for takeaway in takeaways:
        lines.append(f"- {takeaway}")

    lines.extend(["", "## Notable Moments", ""])
    if notable:
        for moment in notable:
            lines.append(f"- {moment}")
    else:
        lines.append("- No specific notable moments identified")

    return lines


def _build_general_sections(full_text: str, config: SummaryConfig) -> List[str]:
    """Build general-purpose sections: key points, main topics."""
    key_points = extract_key_points(full_text, config.max_key_points)
    main_topics = extract_key_points(full_text, 3)

    lines = ["## Key Points", ""]
    for point in key_points:
        lines.append(f"- {point}")

    lines.extend(["", "## Main Topics", ""])
    for topic in main_topics:
        lines.append(f"- {topic}")

    return lines


def create_structured_summary(
    transcript: Dict,
    config: Optional[SummaryConfig] = None,
    use_ai: bool = True,
    content_type: Optional[str] = None,
) -> str:
    """
    Create a structured summary in markdown format.

    Args:
        transcript: Transcript dictionary from Whisper
        config: Summary configuration
        use_ai: Whether to use AI summarization (vs extractive)
        content_type: Content type override — "meeting", "interview", "podcast", or "general".
                      If None, falls back to config.content_type (default "general").

    Returns:
        Structured summary in markdown format
    """
    if config is None:
        config = SummaryConfig()

    resolved_type = content_type if content_type is not None else config.content_type

    full_text = extract_full_text(transcript)

    if not full_text:
        raise SummarizationError("Transcript is empty")

    logger.info(f"Creating {resolved_type} summary from {len(full_text)} characters of text")

    # Generate executive summary
    if use_ai:
        try:
            executive_summary = generate_mistral_summary(full_text, config)
        except Exception as e:
            logger.warning(f"Mistral summarization failed, using extractive: {e}")
            executive_summary = generate_extractive_summary(full_text)
    else:
        executive_summary = generate_extractive_summary(full_text)

    # Build content-type-specific sections
    _section_builders = {
        "meeting":   _build_meeting_sections,
        "interview": _build_interview_sections,
        "podcast":   _build_podcast_sections,
        "general":   _build_general_sections,
    }
    builder = _section_builders.get(resolved_type, _build_general_sections)
    section_lines = builder(full_text, config)

    title = _CONTENT_TYPE_TITLES.get(resolved_type, "# Content Summary")

    lines = [
        title,
        "",
        "## Executive Summary",
        "",
        executive_summary,
        "",
    ] + section_lines + [""]

    summary = "\n".join(lines)
    logger.info(f"Summary created: {len(summary)} characters")

    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Create structured summary from transcript with AI"
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to transcript JSON file",
    )
    parser.add_argument(
        "--out",
        help="Output path for summary (default: transcript_path.summary.md)",
    )
    parser.add_argument(
        "--summary-model-path",
        default=None,
        help=(
            "Path to a local Mistral 7B GGUF file. "
            "If not provided, auto-downloads Q4_K_M (~4.4GB) from HuggingFace Hub."
        ),
    )
    parser.add_argument(
        "--content-type",
        default="general",
        choices=["meeting", "interview", "podcast", "general"],
        help="Type of content: affects output sections (default: general)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=130,
        help="Maximum summary length",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=30,
        help="Minimum summary length",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu, cuda, or auto",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI summarization (use extractive only)",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    transcript_path = Path(args.transcript).expanduser().resolve()
    
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    
    # Load transcript
    transcript = load_transcript(transcript_path)
    
    # Create config
    config = SummaryConfig(
        max_length=args.max_length,
        min_length=args.min_length,
        content_type=args.content_type,
        llm_model_path=args.summary_model_path,
    )

    # Create summary
    summary = create_structured_summary(
        transcript,
        config,
        use_ai=not args.no_ai,
        content_type=args.content_type,
    )
    
    # Determine output path
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = transcript_path.with_suffix(".summary.md")
    
    # Save summary
    out_path.write_text(summary, encoding="utf-8")
    
    print(f"✅ Summary created: {out_path}")
    print(f"\nPreview:")
    print("=" * 60)
    print(summary[:500])
    if len(summary) > 500:
        print("...")
    print("=" * 60)


if __name__ == "__main__":
    main()