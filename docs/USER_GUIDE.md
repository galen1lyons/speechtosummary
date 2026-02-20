# User Guide

Complete feature reference for the speechtosummary pipeline.

---

## Table of Contents

- [Basic Usage](#basic-usage)
- [Whisper Models](#whisper-models)
- [Core Features](#core-features)
  - [Transcription](#transcription)
  - [Speaker Diarization](#speaker-diarization)
  - [Summarization](#summarization)
  - [Quality Metrics](#quality-metrics)
- [Advanced Configuration](#advanced-configuration)
- [Output Files](#output-files)
- [Common Workflows](#common-workflows)

---

## Basic Usage

The pipeline is accessed through the `src.pipeline` module:

```bash
python -m src.pipeline --audio <path-to-audio> [options]
```

**Minimum required:** `--audio` parameter with path to your audio file

---

## Whisper Models

### Supported Models

| Model | Size | Speed | Accuracy | RAM | Use When |
|-------|------|-------|----------|-----|----------|
| `tiny` | 39M | ⚡⚡⚡ | ⭐⭐ | 1GB | Testing/drafts |
| `base` | 74M | ⚡⚡ | ⭐⭐⭐ | 1GB | **Default - good balance** |
| `small` | 244M | ⚡ | ⭐⭐⭐⭐ | 2GB | Better accuracy needed |
| `medium` | 769M | 🐌 | ⭐⭐⭐⭐⭐ | 5GB | Best quality |
| `mesolitica/malaysian-whisper-base` | 74M | ⚡⚡ | ⭐⭐⭐⭐ (Manglish) | 1GB | Malaysian English/Malay |

### Selecting a Model

```bash
# Default (OpenAI Whisper base)
python -m src.pipeline --audio meeting.mp3

# Fast testing
python -m src.pipeline --audio meeting.mp3 --whisper-model tiny

# Best quality
python -m src.pipeline --audio meeting.mp3 --whisper-model medium

# Malaysian Whisper (for Manglish)
python -m src.pipeline --audio meeting.mp3 --whisper-model mesolitica/malaysian-whisper-base
```

### Model Selection Guide

**Choose `tiny` or `base` if:**
- You need quick results for testing
- You have limited RAM/resources
- Audio quality is excellent with clear speech

**Choose `small` or `medium` if:**
- Accuracy is critical
- You have noisy audio
- Multiple speakers with overlapping speech
- Technical terminology present

**Choose Malaysian Whisper if:**
- Speakers use Manglish (Malaysian English)
- Code-switching between English and Malay
- Malaysian accents and colloquialisms

---

## Core Features

### Transcription

Basic transcription converts audio to text with timestamps.

#### Language Options

```bash
# Auto-detect language (default)
python -m src.pipeline --audio meeting.mp3 --language auto

# Force English
python -m src.pipeline --audio meeting.mp3 --language en

# Force Malay
python -m src.pipeline --audio meeting.mp3 --language ms
```

**When to specify language:**
- `--language en`: Pure English audio, speeds up processing
- `--language ms`: Pure Malay audio
- `--language auto`: Mixed languages, Manglish, or unsure (default)

#### Advanced Whisper Parameters

```bash
# Higher beam size (better quality, slower)
python -m src.pipeline --audio meeting.mp3 --beam-size 10

# Temperature (0.0 = deterministic, >0 = more variation)
python -m src.pipeline --audio meeting.mp3 --temperature 0.0

# Initial prompt (guides model context)
python -m src.pipeline --audio meeting.mp3 \
  --initial-prompt "Technical discussion about machine learning"

# Combined advanced settings
python -m src.pipeline --audio meeting.mp3 \
  --whisper-model medium \
  --beam-size 10 \
  --temperature 0.0 \
  --language en \
  --initial-prompt "Quarterly business review meeting"
```

**Parameter Guide:**
- `--beam-size`: 1-10 (default: 5). Higher = better quality but slower
- `--temperature`: 0.0-1.0 (default: 0.0). Use 0.0 for transcription
- `--initial-prompt`: Provide context to improve accuracy for technical terms

For detailed parameter documentation, see [demo/WHISPER_PARAMETERS.md](../demo/WHISPER_PARAMETERS.md)

---

### Speaker Diarization

Identify who spoke when in multi-speaker audio.

#### Basic Usage

```bash
# Enable diarization
python -m src.pipeline --audio meeting.mp3 --diarize
```

#### Specify Speaker Count

```bash
# If you know the number of speakers
python -m src.pipeline --audio meeting.mp3 \
  --diarize \
  --min-speakers 2 \
  --max-speakers 4
```

**Tips:**
- Specify speaker range for better accuracy
- Default: 1-5 speakers auto-detected
- Works best with clean audio and distinct voices

#### Output Files

With diarization enabled, you get additional outputs:
- `meeting.speakers.txt` - Transcript with speaker labels (SPEAKER_00, SPEAKER_01, etc.)
- `meeting.speaker_stats.json` - Speaking time statistics per speaker

#### Example Output

```
SPEAKER_00: Good morning everyone.
SPEAKER_01: Morning. Let's start with the project update.
SPEAKER_00: Sure, we completed phase one last week.
```

---

### Summarization

Automatically generate structured summaries with action items.

#### Basic Usage

```bash
# Summarization is enabled by default
python -m src.pipeline --audio meeting.mp3
```

The summary includes:
- Executive summary paragraph
- Key discussion points
- Decisions made
- Action items with owners

#### Summary Output

Check `outputs/meeting.summary.md`:

```markdown
# Meeting Summary

## Executive Summary
[High-level overview of the meeting]

## Key Discussion Points
- Point 1
- Point 2

## Decisions Made
- Decision 1
- Decision 2

## Action Items
- [ ] Action 1 (Owner: Person A)
- [ ] Action 2 (Owner: Person B)
```

---

### Quality Metrics

Evaluate transcription quality against a reference transcript.

#### Usage

```bash
python -m src.pipeline \
  --audio meeting.mp3 \
  --reference-transcript ground_truth.txt
```

#### Metrics Calculated

- **WER (Word Error Rate)**: Percentage of word-level errors (lower is better)
- **CER (Character Error Rate)**: Percentage of character-level errors (lower is better)
- **RTF (Real-Time Factor)**: Processing speed ratio
  - RTF < 1.0 = Faster than realtime ✅
  - RTF = 1.0 = Realtime speed
  - RTF > 1.0 = Slower than realtime

#### Example Output

```
Transcription Metrics:
- WER: 12.5%
- CER: 8.3%
- RTF: 0.85 (faster than realtime)
- Processing time: 127.4 seconds
- Audio duration: 150.0 seconds
```

---

## Advanced Configuration

### Hallucination Suppression

Control hallucination detection thresholds (prevents model from generating repeated or nonsense text):

```bash
python -m src.pipeline --audio meeting.mp3 \
  --compression-ratio-threshold 2.4 \
  --logprob-threshold -1.0 \
  --no-speech-threshold 0.6
```

**Default values:**
- `compression_ratio_threshold`: 2.4 (lower = more aggressive filtering)
- `logprob_threshold`: -1.0 (higher = more aggressive filtering)
- `no_speech_threshold`: 0.6 (higher = more likely to mark as silence)

**When to adjust:**
- Seeing repeated text → Lower compression_ratio_threshold to 2.0
- Model generating nonsense → Increase logprob_threshold to -0.8
- Too much silence marked → Lower no_speech_threshold to 0.5

### Custom Output Directories

```bash
python -m src.pipeline \
  --audio meeting.mp3 \
  --output-dir custom_outputs/
```

---

## Output Files

### Standard Outputs (Always Generated)

After running the pipeline, check `outputs/` folder:

```
outputs/
├── meeting.txt                # Plain text transcript
├── meeting.json               # Transcript with timestamps
└── meeting.summary.md         # Structured summary
```

### With Diarization (When --diarize Enabled)

```
outputs/
├── meeting.txt
├── meeting.json
├── meeting.summary.md
├── meeting.speakers.txt       # Transcript with speaker labels
└── meeting.speaker_stats.json # Speaking time per speaker
```

### JSON Format

`meeting.json` structure:

```json
{
  "text": "Complete transcript...",
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Good morning everyone."
    }
  ],
  "language": "en"
}
```

---

## Common Workflows

### Workflow 1: Quick Draft Transcript

Fast transcription for quick review:

```bash
python -m src.pipeline \
  --audio meeting.mp3 \
  --whisper-model tiny \
  --language en
```

**Use when:** Quick draft needed, will manually correct later

---

### Workflow 2: High-Quality Production Transcript

Best quality for final deliverables:

```bash
python -m src.pipeline \
  --audio meeting.mp3 \
  --whisper-model medium \
  --beam-size 10 \
  --temperature 0.0 \
  --language auto
```

**Use when:** Final transcript for distribution, accuracy critical

---

### Workflow 3: Meeting Minutes with Speakers

Complete meeting documentation:

```bash
python -m src.pipeline \
  --audio meeting.mp3 \
  --whisper-model base \
  --diarize \
  --min-speakers 3 \
  --max-speakers 5 \
  --language en
```

**Use when:** Need to know who said what, generating formal minutes

---

### Workflow 4: Malaysian/Manglish Content

Optimized for Malaysian English and code-switching:

```bash
python -m src.pipeline \
  --audio meeting.mp3 \
  --whisper-model mesolitica/malaysian-whisper-base \
  --language auto \
  --beam-size 7 \
  --initial-prompt "Malaysian English conversation"
```

**Use when:** Speakers use Manglish, mix English and Malay

---

### Workflow 5: Quality Evaluation

Test and measure transcription quality:

```bash
python -m src.pipeline \
  --audio meeting.mp3 \
  --whisper-model base \
  --reference-transcript reference.txt \
  --language en
```

**Use when:** Comparing models, benchmarking accuracy

---

## Tips and Best Practices

### For Best Accuracy

1. **Clean audio**: Remove background noise if possible
2. **Specify language**: If audio is single language, use `--language en` or `--language ms`
3. **Use larger models**: `medium` model for critical transcripts
4. **Higher beam size**: Use `--beam-size 10` for better quality
5. **Provide context**: Use `--initial-prompt` for technical topics

### For Best Performance

1. **Use smaller models**: `tiny` or `base` for quick results
2. **Specify language**: Avoid auto-detection overhead
3. **Batch processing**: Process multiple files in scripts
4. **GPU acceleration**: Ensure CUDA is available for faster processing

### For Malaysian Content

1. **Use Malaysian Whisper**: Better for Manglish and local accents
2. **Auto-detect language**: Use `--language auto` for code-switching
3. **Context prompts**: Add `--initial-prompt "Malaysian English"` for better results
4. **Test configurations**: See [testing documentation](testing/README.md) for optimal settings

---

## Command Reference

### Basic Commands

```bash
# Minimal
python -m src.pipeline --audio file.mp3

# With model
python -m src.pipeline --audio file.mp3 --whisper-model medium

# With diarization
python -m src.pipeline --audio file.mp3 --diarize

# Full features
python -m src.pipeline \
  --audio file.mp3 \
  --whisper-model medium \
  --diarize \
  --min-speakers 2 \
  --max-speakers 4 \
  --language en \
  --beam-size 10
```

### Help

```bash
# Get all available options
python -m src.pipeline --help
```

---

## Troubleshooting

For common issues and solutions, see the [Troubleshooting Guide](TROUBLESHOOTING.md).

For testing and model comparison, see the [Testing Documentation](testing/README.md).

---

**Need more help?** Check:
- [Getting Started Guide](GETTING_STARTED.md) for setup issues
- [Troubleshooting](TROUBLESHOOTING.md) for error solutions
- [Testing Documentation](testing/README.md) for model comparisons
