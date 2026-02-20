# Optimal Whisper Test Suite

## Overview

The `optimal_whisper_test.py` script runs 8 targeted test configurations to find the optimal Whisper setup for transcribing Manglish (Malaysian English + Malay code-switching) audio.

### Purpose

Based on diagnosis of comprehensive tests, two critical issues were identified:
- **L3 (base + language=ms)**: Catastrophic hallucination loop ("hello" repeated)
- **L4 (Malaysian Whisper + language=auto)**: Functional but has localized hallucinations

This test suite solves the problem by:
- Using Malaysian Whisper model (proven to handle code-switching)
- Testing beam size variations (7, 10)
- Testing temperature variations (0.0, 0.2)
- Testing initial prompts (None vs context-aware)
- Testing hallucination suppression thresholds (Standard vs Aggressive)

The goal is to identify the optimal configuration that minimizes hallucinations while maintaining transcription accuracy.

## Test Configurations

The suite runs 8 tests (OPT1-OPT8) in a fractional factorial design:

| Test ID | Description | Beam | Temp | Prompt | Compression Ratio | Logprob | No Speech |
|---------|-------------|------|------|--------|-------------------|---------|-----------|
| OPT1 | Baseline enhanced | 7 | 0.0 | None | 2.4 | -1.0 | 0.6 |
| OPT2 | High beam size | 10 | 0.0 | None | 2.4 | -1.0 | 0.6 |
| OPT3 | Context prompt | 7 | 0.0 | Yes | 2.4 | -1.0 | 0.6 |
| OPT4 | High beam + prompt | 10 | 0.0 | Yes | 2.4 | -1.0 | 0.6 |
| OPT5 | Higher temperature | 7 | 0.2 | None | 2.4 | -1.0 | 0.6 |
| OPT6 | High beam + higher temp | 10 | 0.2 | None | 2.4 | -1.0 | 0.6 |
| OPT7 | Aggressive suppression | 10 | 0.0 | Yes | 2.0 | -0.8 | 0.5 |
| OPT8 | Max quality (kitchen sink) | 10 | 0.0 | Yes | 2.0 | -0.8 | 0.5 |

### Threshold Settings Explained

**Standard Thresholds** (OPT1-OPT6):
- `compression_ratio_threshold: 2.4` - Moderate hallucination detection
- `logprob_threshold: -1.0` - Standard confidence filtering
- `no_speech_threshold: 0.6` - Standard silence detection

**Aggressive Thresholds** (OPT7-OPT8):
- `compression_ratio_threshold: 2.0` - Stricter hallucination detection
- `logprob_threshold: -0.8` - More aggressive confidence filtering
- `no_speech_threshold: 0.5` - More permissive silence detection

### Test Strategy

1. **Baseline** (OPT1-OPT2): Standard vs high beam
2. **Context** (OPT3-OPT4): With vs without prompts
3. **Temperature** (OPT5-OPT6): Deterministic vs stochastic
4. **Suppression** (OPT7-OPT8): Standard vs aggressive hallucination control

## Execution Instructions

### Prerequisites

1. **Environment Setup**
   ```bash
   cd /home/dedmtiintern/speechtosummary
   source venv/bin/activate
   ```

2. **Verify Audio File**
   ```bash
   # Check the audio file exists
   ls -lh "data/mamak session scam.mp3"
   ```

3. **Check Dependencies**
   - Python dependencies: transformers, whisper, torch
   - HuggingFace token configured (if needed for Malaysian Whisper)
   - Sufficient disk space (>500 MB for outputs)

### Basic Usage

```bash
# Preview tests (dry run)
python scripts/optimal_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --dry-run

# Run all tests
python scripts/optimal_whisper_test.py \
  --audio "data/mamak session scam.mp3"

# Resume interrupted test run
python scripts/optimal_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --resume
```

### Advanced Options

```bash
# Custom output directory
python scripts/optimal_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --output-dir "outputs/my_custom_test" \
  --results-dir "results/my_custom_test"

# Test different audio file
python scripts/optimal_whisper_test.py \
  --audio "data/different_audio.mp3"
```

### Resume After Interruption

The script automatically saves progress after each test. If interrupted:

```bash
# Simply run with --resume flag
python scripts/optimal_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --resume

# Already-completed tests will be skipped automatically
```

## Output Structure

```
outputs/optimal_test/
├── OPT1.json                    # Test 1 results (full metadata)
├── OPT1.txt                     # Test 1 transcript (plain text)
├── OPT2.json
├── OPT2.txt
├── OPT3.json
├── OPT3.txt
├── OPT4.json
├── OPT4.txt
├── OPT5.json
├── OPT5.txt
├── OPT6.json
├── OPT6.txt
├── OPT7.json
├── OPT7.txt
├── OPT8.json
├── OPT8.txt
└── optimal_test_results.json    # Summary of all tests with rankings
```

### Reading Results

```bash
# View results summary
cat outputs/optimal_test/optimal_test_results.json | jq '.results[] | {test_id, hallucination_count, rtf: .metrics.rtf}'

# View best configuration (lowest hallucination count)
cat outputs/optimal_test/optimal_test_results.json | jq '.results | sort_by(.hallucination_count) | .[0]'

# View specific test transcript
cat outputs/optimal_test/OPT7.txt | head -50
```

### Key Metrics

From `optimal_test_results.json`:

1. **Hallucination Count**: Lower is better (target: <50)
2. **RTF (Real-Time Factor)**: Lower is faster (target: <2.0)
3. **Segment Count**: Should be reasonable for 10-min audio (~150-200)
4. **Average Segment Length**: Should be 2-5 seconds

## Estimated Runtime

- **Per test**: 3-7 minutes (depending on audio length and beam size)
- **Total suite**: ~40-60 minutes for 8 tests
- **Audio duration**: ~10 minutes (mamak session scam.mp3)

Runtime factors:
- Beam size 10 is slower than beam size 7
- Aggressive thresholds may add slight overhead
- CPU inference (default) is slower than GPU

## Troubleshooting

### Error: Audio file not found

```bash
# Verify path
ls -la "data/mamak session scam.mp3"

# Use absolute path if needed
python scripts/optimal_whisper_test.py \
  --audio "/home/dedmtiintern/speechtosummary/data/mamak session scam.mp3"
```

### Error: Model download fails

```bash
# Check internet connection
ping -c 3 huggingface.co

# Pre-download model
python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='mesolitica/malaysian-whisper-base')"
```

### Error: CUDA out of memory

```bash
# The script uses CPU by default for HuggingFace models
# If using GPU, reduce beam size or use smaller model
```

### Tests are interrupted

```bash
# Simply resume - already-completed tests are skipped
python scripts/optimal_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --resume
```

### Results look wrong

```bash
# Check for errors in results file
cat outputs/optimal_test/optimal_test_results.json | jq '.results[] | select(.error != null)'

# View hallucination counts
cat outputs/optimal_test/optimal_test_results.json | jq '.results[] | {test_id, hallucinations: .hallucination_count, rtf: .metrics.rtf}'
```

### Virtual environment issues

```bash
# Verify venv is activated
echo $VIRTUAL_ENV
# Should output: /home/dedmtiintern/speechtosummary/venv

# Reactivate if needed
source venv/bin/activate
```

## Success Criteria

A successful test run should show:

- ✅ All 8 tests completed without errors
- ✅ OPT1-OPT8 transcription files generated
- ✅ At least one config has hallucination_count < 50
- ✅ Best config RTF < 3.0
- ✅ Transcripts are readable and contextually accurate
- ✅ Clear winner identified in summary

## Next Steps After Testing

1. **Review Results**
   ```bash
   # View ranked results
   cat outputs/optimal_test/optimal_test_results.json | jq '.results | sort_by(.hallucination_count) | .[]'
   ```

2. **Read Best Transcript**
   ```bash
   # Identify best test (e.g., OPT7) and read it
   cat outputs/optimal_test/OPT7.txt | head -50
   ```

3. **Manual Verification**
   - Listen to audio while reading transcript
   - Check for code-switching accuracy
   - Verify hallucinations are minimal

4. **Apply to Production**
   - Use the winning configuration in your main pipeline
   - Update `WhisperConfig` with optimal parameters

## Related Documentation

- Main project documentation: [../README.md](../README.md)
- Getting started guide: [../GETTING_STARTED.md](../GETTING_STARTED.md)
- Troubleshooting guide: [../TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

---

**Last Updated**: 2026-02-09
**Script**: `scripts/optimal_whisper_test.py`
**Status**: Production Ready
