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
| `mesolitica/malaysian-whisper-base` | 74M | ❌ | ❌ (unviable) | 1GB | Not recommended — see note below |

### Selecting a Model

```bash
# Default (faster-whisper base)
python -m src.pipeline --audio data/meeting.mp3

# Fast testing
python -m src.pipeline --audio data/meeting.mp3 --whisper-model tiny

# Best quality
python -m src.pipeline --audio data/meeting.mp3 --whisper-model medium

# Optimal Manglish settings (recommended)
python scripts/transcribe.py transcribe --audio "data/meeting.mp3" --config fw5_optimal
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

**Malaysian Whisper (`mesolitica/malaysian-whisper-*`):**
Evaluated and found unviable for production use — RTF 23.7x (real-time), 1,027 hallucinations
on a 9-minute file. Use `fw5_optimal` config with faster-whisper base instead.
Requires `--backend openai-whisper` if you still wish to test it.

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

For detailed parameter documentation, see [FASTER_WHISPER_OPTIMIZATION.md](FASTER_WHISPER_OPTIMIZATION.md)

---

### Speaker Diarization

Identify who spoke when in multi-speaker audio.

#### Basic Usage

```bash
# Enable diarization
python -m src.pipeline --audio data/meeting.mp3 --enable-diarization
```

#### Specify Speaker Count

```bash
# If you know the number of speakers
python -m src.pipeline --audio data/meeting.mp3 \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 4
```

**Tips:**
- Specify speaker range for better accuracy
- Default: 1-5 speakers auto-detected
- Works best with clean audio and distinct voices

#### Output Files

With diarization enabled, you get additional outputs inside the run folder:
- `speakers.txt` - Transcript with speaker labels (SPEAKER_00, SPEAKER_01, etc.)
- `speaker_stats.json` - Speaking time statistics per speaker
- `diarization.rttm` - RTTM output for DER/JER evaluation

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

The default faster-whisper backend (`fw5_optimal` config) uses strict VAD settings that eliminate
~99% of hallucinations. If you're using the openai-whisper backend and see repeated or nonsense
text, use a lower beam size or switch to faster-whisper:

```bash
# Recommended: use faster-whisper with optimal config
python scripts/transcribe.py transcribe --audio "data/meeting.mp3" --config fw5_optimal

# Or switch backend explicitly
python -m src.pipeline --audio data/meeting.mp3 --backend faster-whisper --beam-size 7
```

### Custom Output Directories

```bash
python -m src.pipeline \
  --audio meeting.mp3 \
  --output-dir custom_outputs/
```

---

## Output Files

### Standard Outputs (Always Generated)

Each run creates a timestamped folder under `outputs/runs/`:

```
outputs/runs/
└── 20260220T153045Z__meeting__fw-base-int8-general/
    ├── transcript.txt          # Plain text transcript with timestamps
    ├── transcript.json         # Transcript with segment-level detail
    ├── summary.md              # Structured summary
    └── manifest.json           # Run metadata
```

### With `--enable-diarization`

```
    ├── speakers.txt            # Transcript with speaker labels
    ├── speaker_stats.json      # Speaking time per speaker
    └── diarization.rttm        # RTTM output for DER/JER evaluation
```

### With `--reference-transcript`

```
    └── asr_metrics.json        # WER, CER, RTF results
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
  --enable-diarization \
  --min-speakers 3 \
  --max-speakers 5 \
  --language en
```

**Use when:** Need to know who said what, generating formal minutes

---

### Workflow 4: Malaysian/Manglish Content

Optimized for Malaysian English and code-switching using the proven `fw5_optimal` config:

```bash
python scripts/transcribe.py transcribe \
  --audio "data/meeting.mp3" \
  --config fw5_optimal
```

Or via the pipeline directly:

```bash
python -m src.pipeline \
  --audio data/meeting.mp3 \
  --backend faster-whisper \
  --beam-size 7 \
  --language en
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

1. **Use `fw5_optimal` config**: Best results for Manglish — `python scripts/transcribe.py transcribe --audio file.mp3 --config fw5_optimal`
2. **Auto-detect language**: Use `--language auto` for code-switching between English and Malay
3. **Context prompts**: Add `--initial-prompt "Malaysian English"` to guide the model
4. **Avoid Malaysian Whisper**: `mesolitica/malaysian-whisper-*` is unviable (RTF 23.7x, 1,027 hallucinations) — see note in model table above

---

## Command Reference

### Basic Commands

```bash
# Minimal
python -m src.pipeline --audio file.mp3

# With model
python -m src.pipeline --audio file.mp3 --whisper-model medium

# With diarization
python -m src.pipeline --audio file.mp3 --enable-diarization

# Full features
python -m src.pipeline \
  --audio file.mp3 \
  --whisper-model medium \
  --enable-diarization \
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

For model comparison and CLI options, see the [Transcribe CLI Guide](../scripts/TRANSCRIBE_CLI_GUIDE.md).

---

**Need more help?** Check:
- [Getting Started Guide](GETTING_STARTED.md) for setup issues
- [Troubleshooting](TROUBLESHOOTING.md) for error solutions
- [Scripts Guide](../scripts/README.md) for model comparison workflows
