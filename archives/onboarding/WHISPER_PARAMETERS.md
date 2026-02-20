# Whisper Parameters Guide

Quick reference for the Whisper parameters you can test.

## Key Parameters

### beam_size (default: 5)
Controls search thoroughness during decoding.

| Value | Effect |
|-------|--------|
| 1 | Fastest, least accurate |
| 5 | Good balance (default) |
| 10 | More accurate, ~50% slower |

**When to increase:** Noisy audio, unclear speech

### temperature (default: 0.0)
Controls randomness in transcription.

| Value | Effect |
|-------|--------|
| 0.0 | Deterministic, most consistent |
| 0.2-0.4 | Slight variation, can help with unclear audio |

**Fallback mode:** Set `temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` to retry with higher temps if initial attempt fails.

### initial_prompt (default: None)
Provides context to guide transcription. **Most impactful for accented speech.**

**Examples:**
```bash
# Malaysian English
--initial-prompt "Malaysian English business meeting with local accents"

# Technical content
--initial-prompt "Technical discussion about machine learning and neural networks"

# Medical terminology
--initial-prompt "Medical consultation discussing diagnosis and treatment options"
```

## Recommended Settings

| Scenario | Settings |
|----------|----------|
| Clear podcast | `--beam-size 5 --temperature 0.0` |
| Noisy meeting | `--beam-size 10 --temperature 0.0` |
| Manglish/Singlish | `--beam-size 5 --initial-prompt "Malaysian English conversation"` |
| Multiple accents | `--beam-size 10 --initial-prompt "[describe context]"` |

## Example Commands

```bash
# Basic English transcription
python -m src.pipeline --audio test.mp3 --language en

# Malaysian English with prompt
python -m src.pipeline --audio test.mp3 --language en \
  --initial-prompt "Malaysian English meeting with Manglish"

# Higher accuracy for noisy audio
python -m src.pipeline --audio test.mp3 --beam-size 10

# Chinese transcription
python -m src.pipeline --audio test.mp3 --language zh
```
