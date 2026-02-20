# Comprehensive Whisper Model Testing Guide

**Complete testing framework for optimizing Whisper model configurations for Malaysian English transcription**

---

## Table of Contents

1. [Overview & Purpose](#overview--purpose)
2. [Test Methodology](#test-methodology)
3. [Complete Test Matrix](#complete-test-matrix)
4. [Execution Instructions](#execution-instructions)
5. [Analysis & Presentation](#analysis--presentation)
6. [Troubleshooting](#troubleshooting)

---

## Overview & Purpose

### What This Is

A **systematic, data-driven approach** to identify the optimal Whisper configuration for Malaysian English transcription by testing **36 different parameter combinations** across **2 models**.

### The Problem We're Solving

- Previous test (`chaos_parliament.mp3`) failed with repetitive output patterns and low semantic content
- Unclear which Whisper parameters caused the issue
- No quantitative data on Original Whisper vs Malaysian Whisper performance
- Need evidence-based configuration for production deployment

### The Solution

- Test 36 configurations systematically using fractional factorial design
- Compare both models quantitatively with performance metrics
- Identify optimal parameters with statistical evidence
- Provide production-ready recommendations

### Test Context

**Testing Context:**
- **Previous Test Failure:** `chaos_parliament.mp3` with Original Whisper produced poor results
- **Root Cause Hypothesis:** Suboptimal Whisper configuration parameters
- **Test Objective:** Identify optimal configuration for Malaysian English (Manglish) transcription

---

## Test Methodology

### Models Under Test

#### Model 1: Original OpenAI Whisper
- **Model ID:** `base` (default OpenAI Whisper)
- **Size:** 74M parameters
- **Training Data:** Multilingual, general-purpose
- **Strengths:** Robust general performance, good English support
- **Weaknesses:** Not specialized for Malaysian context

#### Model 2: Malaysian Whisper (Mesolitica)
- **Model ID:** `mesolitica/malaysian-whisper-base`
- **Size:** 74M parameters (fine-tuned from Whisper base)
- **Training Data:** Malaysian English, Malay, code-switching
- **Strengths:** Specialized for Malaysian context, Manglish support
- **Weaknesses:** May struggle with clean English or overlapping speech

### Test Audio File

**File:** `data/mamak session scam.mp3`
- **Size:** 9.6 MB (9,589,159 bytes)
- **Duration:** ~10 minutes (estimated)
- **Content:** Mamak stall conversation about scam prevention
- **Characteristics:**
  - Malaysian English (Manglish)
  - Informal conversational speech
  - Code-switching (English + Malay)
  - Moderate background noise expected

### Whisper Configuration Parameters

#### 1. Model (`model_name`)
- **Options:**
  - `base` (OpenAI Whisper Base)
  - `mesolitica/malaysian-whisper-base` (Malaysian Whisper)
- **Impact:** Core model architecture and training data
- **Test Strategy:** Test both models across all other parameter combinations

#### 2. Language (`language`)
- **Options:**
  - `"auto"` - Automatic language detection
  - `"en"` - English (force)
  - `"ms"` - Malay (force)
- **Impact:** Language-specific processing and token decoding
- **Test Strategy:** Test auto-detection vs forced languages
- **Hypothesis:** Malaysian Whisper may benefit from `"ms"` or `"auto"`, Original Whisper from `"en"`

#### 3. Beam Size (`beam_size`)
- **Options:** `1, 3, 5, 7, 10`
- **Default:** 5
- **Impact:**
  - Higher = More thorough search, better quality, slower
  - Lower = Faster, less accurate
- **Test Strategy:** Test 5 levels (1, 3, 5, 7, 10)
- **Hypothesis:** Higher beam sizes may fix chaos_parliament issues

#### 4. Temperature (`temperature`)
- **Options:** `0.0, 0.2, 0.4, 0.6, 0.8`
- **Default:** 0.0
- **Impact:**
  - 0.0 = Deterministic (always same output)
  - >0.0 = Stochastic (sampling, more creative)
- **Test Strategy:** Test 5 levels
- **Hypothesis:** 0.0 (deterministic) is best for transcription accuracy

#### 5. Initial Prompt (`initial_prompt`)
- **Options:**
  - `None` (default)
  - `"Malaysian English conversation about scams at a mamak stall."`
  - `"Manglish. Code-switching between English and Malay."`
  - `"Informal Malaysian conversation."`
- **Impact:**
  - Guides model context and vocabulary
  - Can improve domain-specific accuracy
  - May reduce hallucinations
- **Test Strategy:** Test 4 variations (None + 3 prompts)
- **Hypothesis:** Context-aware prompts will improve Malaysian Whisper performance

### Fractional Factorial Design

Testing all combinations would result in:
- **2 models** × 3 languages × 5 beam_sizes × 5 temperatures × 4 prompts = **600 tests** (infeasible)

Instead, use a **fractional factorial design** with:
1. **Baseline Tests:** Default parameters for both models
2. **Single-Factor Tests:** Vary one parameter at a time
3. **Interaction Tests:** Test promising combinations

This approach reduces tests from 600 to 36 while maintaining statistical validity.

---

## Complete Test Matrix

### Phase 1: Baseline Tests (2 tests)
Test both models with default parameters to establish baseline performance.

| Test ID | Model | Language | Beam | Temp | Prompt |
|---------|-------|----------|------|------|--------|
| B1      | base  | auto     | 5    | 0.0  | None   |
| B2      | malaysian-whisper-base | auto | 5 | 0.0 | None |

### Phase 2: Language Detection Tests (6 tests)
Test language parameter impact on both models.

| Test ID | Model | Language | Beam | Temp | Prompt |
|---------|-------|----------|------|------|--------|
| L1      | base  | auto     | 5    | 0.0  | None   |
| L2      | base  | en       | 5    | 0.0  | None   |
| L3      | base  | ms       | 5    | 0.0  | None   |
| L4      | malaysian-whisper-base | auto | 5 | 0.0 | None |
| L5      | malaysian-whisper-base | en | 5 | 0.0 | None |
| L6      | malaysian-whisper-base | ms | 5 | 0.0 | None |

### Phase 3: Beam Size Tests (10 tests)
Test beam size impact on both models.

| Test ID | Model | Language | Beam | Temp | Prompt |
|---------|-------|----------|------|------|--------|
| BS1     | base  | auto     | 1    | 0.0  | None   |
| BS2     | base  | auto     | 3    | 0.0  | None   |
| BS3     | base  | auto     | 5    | 0.0  | None   |
| BS4     | base  | auto     | 7    | 0.0  | None   |
| BS5     | base  | auto     | 10   | 0.0  | None   |
| BS6     | malaysian-whisper-base | auto | 1 | 0.0 | None |
| BS7     | malaysian-whisper-base | auto | 3 | 0.0 | None |
| BS8     | malaysian-whisper-base | auto | 5 | 0.0 | None |
| BS9     | malaysian-whisper-base | auto | 7 | 0.0 | None |
| BS10    | malaysian-whisper-base | auto | 10 | 0.0 | None |

### Phase 4: Temperature Tests (10 tests)
Test temperature impact on both models.

| Test ID | Model | Language | Beam | Temp | Prompt |
|---------|-------|----------|------|------|--------|
| T1      | base  | auto     | 5    | 0.0  | None   |
| T2      | base  | auto     | 5    | 0.2  | None   |
| T3      | base  | auto     | 5    | 0.4  | None   |
| T4      | base  | auto     | 5    | 0.6  | None   |
| T5      | base  | auto     | 5    | 0.8  | None   |
| T6      | malaysian-whisper-base | auto | 5 | 0.0 | None |
| T7      | malaysian-whisper-base | auto | 5 | 0.2 | None |
| T8      | malaysian-whisper-base | auto | 5 | 0.4 | None |
| T9      | malaysian-whisper-base | auto | 5 | 0.6 | None |
| T10     | malaysian-whisper-base | auto | 5 | 0.8 | None |

### Phase 5: Initial Prompt Tests (8 tests)
Test prompt impact on both models (using best language setting from Phase 2).

| Test ID | Model | Language | Beam | Temp | Prompt |
|---------|-------|----------|------|------|--------|
| P1      | base  | [best]   | 5    | 0.0  | None   |
| P2      | base  | [best]   | 5    | 0.0  | "Malaysian English conversation about scams at a mamak stall." |
| P3      | base  | [best]   | 5    | 0.0  | "Manglish. Code-switching between English and Malay." |
| P4      | base  | [best]   | 5    | 0.0  | "Informal Malaysian conversation." |
| P5      | malaysian-whisper-base | [best] | 5 | 0.0 | None |
| P6      | malaysian-whisper-base | [best] | 5 | 0.0 | "Malaysian English conversation about scams at a mamak stall." |
| P7      | malaysian-whisper-base | [best] | 5 | 0.0 | "Manglish. Code-switching between English and Malay." |
| P8      | malaysian-whisper-base | [best] | 5 | 0.0 | "Informal Malaysian conversation." |

**Total Tests: 36 tests**

### Evaluation Metrics

#### 1. Transcription Quality Metrics

**A. Semantic Quality (Manual Review)**
- **Coherence:** Does the transcript make logical sense?
- **Repetitions:** Are there repetitive phrases (like chaos_parliament issue)?
- **Hallucinations:** Any content not present in audio?
- **Code-switching Accuracy:** Are Malay words correctly transcribed?
- **Contextual Accuracy:** Does it match mamak/scam conversation context?

**Scoring:** 1-5 scale (1=poor, 5=excellent)

**B. ASR Metrics (Automated)**
- **Word Error Rate (WER):** Requires reference transcript
- **Character Error Rate (CER):** Requires reference transcript
- **Compression Ratio:** Detect potential hallucinations (<2.4 is concerning)

**Note:** If reference transcript available, calculate WER/CER. Otherwise, use manual review + compression ratio.

#### 2. Performance Metrics (Automated)

- **RTF (Real-Time Factor):** Processing time / Audio duration
  - RTF < 1.0 = Faster than realtime (good)
  - RTF = 1.0 = Processes at realtime speed
  - RTF > 1.0 = Slower than realtime
- **Processing Time:** Absolute time in seconds
- **Number of Segments:** Transcript segment count

#### 3. Metadata

- **Language Detected:** Auto-detected language code
- **Model Used:** Confirm correct model loaded
- **Device Used:** CPU vs CUDA
- **Timestamp:** Test execution time

---

## Execution Instructions

### Prerequisites

#### 1. Environment Setup

**IMPORTANT:** You MUST activate the virtual environment before running any tests!

```bash
# Navigate to project directory
cd /home/dedmtiintern/speechtosummary

# Activate virtual environment
source venv/bin/activate

# Your prompt should now show (venv) prefix
# (venv) user@host:~/speechtosummary$

# Verify dependencies
pip list | grep -E "(whisper|transformers|torch)"
```

**Validate your setup:**
```bash
python scripts/validate_setup.py
```

This will check:
- All required Python modules are installed
- Audio file exists
- Directories are set up
- Scripts are present

#### 2. Audio File Verification

```bash
# Check file exists
ls -lh "data/mamak session scam.mp3"

# Get audio info (optional)
ffprobe "data/mamak session scam.mp3" 2>&1 | grep Duration
```

#### 3. Output Directory Preparation

```bash
mkdir -p outputs/comprehensive_test
mkdir -p results/comprehensive_test
```

### Running Tests

#### Quick Start

Run the complete test suite with default settings:

```bash
python scripts/comprehensive_whisper_test.py --audio "data/mamak session scam.mp3"
```

#### Execution Options

**Option 1: Full Test Suite (36 tests, ~3-6 hours)**

```bash
python scripts/comprehensive_whisper_test.py \
  --audio "data/mamak session scam.mp3"
```

**Option 2: Quick Test (First 5 tests only)**

```bash
python scripts/comprehensive_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --limit 5
```

**Option 3: Resume from Specific Test**

```bash
python scripts/comprehensive_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --start-from BS1
```

**Option 4: Custom Output Directories**

```bash
python scripts/comprehensive_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --output-dir outputs/my_test \
  --results-dir results/my_test
```

### Test Phases

The comprehensive test runs in 5 phases:

1. **Baseline Tests (2 tests)** - Default configs for both models
2. **Language Tests (6 tests)** - Test auto, en, ms language settings
3. **Beam Size Tests (10 tests)** - Test beam sizes 1, 3, 5, 7, 10
4. **Temperature Tests (10 tests)** - Test temperatures 0.0-0.8
5. **Prompt Tests (8 tests)** - Test different initial prompts

### Monitoring Progress

#### During Test Execution

Tests print real-time progress:

```
======================================================================
Test 1/36: B1
======================================================================
Running test B1: Original Whisper - Default config
✅ Test B1 complete - RTF: 1.23x, Time: 145.2s, Segments: 87
Progress: 1/36 tests complete
```

#### Check Intermediate Results

Results are saved after each test completes. You can check progress anytime:

```bash
# View test summary
cat results/comprehensive_test/test_summary.md

# Count completed tests
ls outputs/comprehensive_test/*.json | wc -l
```

### Best Practices

#### 1. Run Quick Test First

Before committing to the full 36-test suite:

```bash
# Run 2 baseline tests to verify everything works
python scripts/comprehensive_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --limit 2
```

#### 2. Monitor Disk Space

Each test generates ~500KB-2MB of output. Full suite needs ~50-100MB.

```bash
# Check available space
df -h .
```

#### 3. Run in Background

For long-running tests:

```bash
# Use nohup to run in background
nohup python scripts/comprehensive_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  > test_output.log 2>&1 &

# Check progress
tail -f test_output.log
```

#### 4. Save Logs

```bash
# Capture all output
python scripts/comprehensive_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  2>&1 | tee comprehensive_test.log
```

### Timeline Estimates

| Task | Time Estimate |
|------|---------------|
| Environment setup | 5-10 minutes |
| Full test suite (36 tests) | 3-6 hours |
| Quick test (5 tests) | 30-60 minutes |
| Analysis generation | 1-2 minutes |
| Presentation generation | 1 minute |
| Manual quality review | 1-2 hours |

**Total for Complete Workflow:** 4-9 hours (mostly automated)

---

## Analysis & Presentation

### Step 1: Generate Analysis Report

```bash
python scripts/analyze_test_results.py \
  --results results/comprehensive_test/test_results.json \
  --output results/comprehensive_test/analysis_report.md
```

This creates:
- Comparative analysis tables
- Model performance comparison
- Parameter impact analysis
- Top 5 fastest configurations
- Production recommendations

### Step 2: Generate Presentation Materials

```bash
python scripts/generate_presentation.py \
  --results results/comprehensive_test/test_results.json \
  --output-dir results/comprehensive_test/presentation
```

This creates:
- Executive summary slides (Markdown/Marp format)
- Quick reference guide
- CSV export for custom visualizations

### Output Files

#### Transcription Outputs

```
outputs/comprehensive_test/
├── B1_base_auto_beam5_temp0.0_none.json
├── B1_base_auto_beam5_temp0.0_none.txt
├── B2_malaysian_auto_beam5_temp0.0_none.json
├── B2_malaysian_auto_beam5_temp0.0_none.txt
└── ... (72 files total: 36 JSON + 36 TXT)
```

#### Results and Analysis

```
results/comprehensive_test/
├── test_results.json          # Raw test results
├── test_summary.md            # Auto-generated summary
├── analysis_report.md         # Detailed analysis
└── presentation/
    ├── executive_summary_slides.md
    ├── quick_reference.md
    └── test_results.csv
```

### Presenting Results

#### For Your Colleague

1. **Show Test Plan:** Explain methodology, show test matrix, justify approach

2. **Show Analysis Report:** `results/comprehensive_test/analysis_report.md`
   - Key findings
   - Model comparison
   - Parameter impact
   - Recommendations

3. **Show Executive Slides:** `results/comprehensive_test/presentation/executive_summary_slides.md`
   - High-level overview
   - Best configuration
   - Production recommendations

4. **Provide Quick Reference:** `results/comprehensive_test/presentation/quick_reference.md`
   - Top 3 configs for each model
   - Copy-paste ready configurations

#### Optional: Create PDF Slides

If you want PDF slides from the Markdown:

```bash
# Install Marp CLI (one-time)
npm install -g @marp-team/marp-cli

# Generate PDF
marp results/comprehensive_test/presentation/executive_summary_slides.md --pdf
```

### Results Analysis Framework

#### 1. Comparative Analysis

**Per-Parameter Analysis**
- **Language:** Which language setting works best per model?
- **Beam Size:** Quality vs speed tradeoff
- **Temperature:** Impact on determinism
- **Initial Prompt:** Does context improve accuracy?

**Cross-Model Comparison**
- **Original Whisper vs Malaysian Whisper:** Which is better overall?
- **Parameter Sensitivity:** Which model is more affected by parameter changes?
- **Optimal Configurations:** Best config for each model

#### 2. Understanding Results

**RTF (Real-Time Factor)**

**Primary performance metric:**
- **RTF < 1.0** → Faster than realtime ✅
- **RTF = 1.0** → Processes at realtime speed
- **RTF > 1.0** → Slower than realtime ⚠️

**Example:** RTF of 0.5x means processing takes half the audio duration.

**Quality Indicators**

From `analysis_report.md`:
- **Best Configuration** - Lowest RTF (fastest)
- **Model Comparison** - Average RTF per model
- **Parameter Impact** - How each parameter affects performance

**Top Configurations**

From `presentation/quick_reference.md`:
- Top 3 configs for Original Whisper
- Top 3 configs for Malaysian Whisper
- Copy-paste ready code snippets

#### 3. Summary Statistics

| Metric | Original Whisper | Malaysian Whisper |
|--------|------------------|-------------------|
| Best WER | [value] | [value] |
| Best Config | [config] | [config] |
| Avg RTF | [value] | [value] |
| Fastest Config | [config] | [config] |
| Quality Score | [value] | [value] |

### Next Steps After Testing

#### 1. Manual Quality Review

Review transcriptions for top 3 configurations:

```bash
# Read best transcription
cat outputs/comprehensive_test/B1_base_auto_beam5_temp0.0_none.txt
```

Score quality (1-5) based on:
- Coherence
- Repetitions
- Hallucinations
- Code-switching accuracy

#### 2. Create Reference Transcript (Optional)

For quantitative WER/CER calculation:

1. Manually transcribe a portion of the audio (2-3 minutes)
2. Save as `data/mamak_session_scam_reference.txt`
3. Use ASR metrics to calculate WER/CER

#### 3. Validate on Additional Audio

Test best configuration on other files:

```bash
# Test on different audio
python scripts/test_single_config.py \
  --audio "data/base_mamak.mp3" \
  --model mesolitica/malaysian-whisper-base \
  --language auto \
  --beam-size 5
```

### Validation Checklist

Before presenting results:

- [ ] All 36 tests completed without errors
- [ ] `test_results.json` exists with 36 entries
- [ ] `analysis_report.md` generated successfully
- [ ] `presentation/` folder contains all 3 files
- [ ] Reviewed top 3 configurations manually
- [ ] Verified transcription quality for best config
- [ ] No hallucinations or repetitive patterns

---

## Troubleshooting

### Issue: "Audio file not found"

**Solution:**
```bash
# Check file path (note the space in filename)
ls -la "data/mamak session scam.mp3"

# If missing, verify you're in the correct directory
pwd  # Should be: /home/dedmtiintern/speechtosummary
```

### Issue: "Model download error"

**Solution:**
```bash
# Pre-download models
python -c "import whisper; whisper.load_model('base')"

# For Malaysian Whisper
python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', 'mesolitica/malaysian-whisper-base')"
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Force CPU mode (slower but more stable)
export CUDA_VISIBLE_DEVICES=""
python scripts/comprehensive_whisper_test.py --audio "data/mamak session scam.mp3"
```

### Issue: Tests taking too long

**Solution 1:** Run smaller subset
```bash
python scripts/comprehensive_whisper_test.py \
  --audio "data/mamak session scam.mp3" \
  --limit 10
```

**Solution 2:** Use faster baseline model
```bash
# Switch from "base" to "tiny" model
# Edit script to use "tiny" model
```

### Issue: Out of Memory

**Solution 1:** Force CPU mode
```bash
export CUDA_VISIBLE_DEVICES=""
```

**Solution 2:** Test one model at a time
```bash
# Edit script to remove one model from test configs
```

### Issue: Model Download Fails

**Solution:** Pre-download models
```bash
python -c "import whisper; whisper.load_model('base')"
python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', 'mesolitica/malaysian-whisper-base')"
```

---

## Quick Command Reference

```bash
# Full test suite
python scripts/comprehensive_whisper_test.py --audio "data/mamak session scam.mp3"

# Analyze results
python scripts/analyze_test_results.py

# Generate presentation
python scripts/generate_presentation.py

# View summary
cat results/comprehensive_test/test_summary.md

# View analysis
cat results/comprehensive_test/analysis_report.md

# View best config
cat results/comprehensive_test/presentation/quick_reference.md
```

---

## Appendix: Configuration Reference

### WhisperConfig Parameters

```python
@dataclass
class WhisperConfig:
    model_name: str = "base"           # Model ID
    language: str = "auto"              # Language code or "auto"
    device: str = "auto"                # "auto", "cuda", "cpu"
    beam_size: int = 5                  # Beam search size (1-10)
    temperature: float = 0.0            # Sampling temperature (0.0-1.0)
    initial_prompt: str = None          # Context prompt
    best_of: int = 5                    # Candidates to evaluate
    patience: float = 1.0               # Early stopping
    length_penalty: float = 1.0         # Length penalty
    suppress_tokens: str = "-1"         # Tokens to suppress
    temperature_increment_on_fallback: float = 0.2  # Temp increase
    compression_ratio_threshold: float = 2.4  # Hallucination detection
    logprob_threshold: float = -1.0     # Log probability threshold
    no_speech_threshold: float = 0.6    # Silence detection
```

---

## Cross-References

### Related Documentation
- Project Setup: [../GETTING_STARTED.md](../GETTING_STARTED.md)
- General Troubleshooting: [../TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
- Project README: [../../README.md](../../README.md)
- Testing Overview: [README.md](README.md)

### Scripts
- Main test script: `scripts/comprehensive_whisper_test.py`
- Analysis script: `scripts/analyze_test_results.py`
- Presentation generator: `scripts/generate_presentation.py`

---

## Version Information

**Version:** 1.0
**Last Updated:** 2026-02-09
**Status:** Ready for execution

**Author:** dedmtiintern
**Project:** speechtosummary - Malaysian English Speech-to-Text Pipeline

---

## Support & Contact

**Issues:**
- Check test logs in `results/comprehensive_test/`
- Review individual transcriptions in `outputs/comprehensive_test/`
- Refer to [../TROUBLESHOOTING.md](../TROUBLESHOOTING.md) for common issues

**Documentation:**
- Test methodology details in this file
- Quick start guide in [README.md](README.md)
- Project instructions in [../../CLAUDE.md](../../CLAUDE.md)
