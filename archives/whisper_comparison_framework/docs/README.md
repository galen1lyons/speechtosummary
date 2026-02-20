# Testing Documentation

Systematic testing frameworks for evaluating and comparing Whisper model configurations.

---

## Overview

This directory contains documentation for two test suites designed to identify optimal Whisper configurations for Malaysian English transcription:

- **[Comprehensive Test Suite](COMPREHENSIVE_TESTS.md)** - 36 tests, full parameter sweep
- **[Optimal Test Suite](OPTIMAL_TESTS.md)** - 8 tests, focused optimization

---

## Which Test Suite Should I Use?

### Use Comprehensive Tests When:

- You want to systematically evaluate all parameter combinations
- You're comparing Original Whisper vs Malaysian Whisper models
- You need data-driven evidence for configuration decisions
- You have 3-6 hours for test execution
- You're presenting findings to stakeholders

**[Go to Comprehensive Test Suite →](COMPREHENSIVE_TESTS.md)**

### Use Optimal Tests When:

- You need quick, targeted optimization
- You've identified a specific issue (hallucinations, accuracy)
- You want to test focused parameter variations
- You have 40-60 minutes for test execution
- You're fine-tuning an existing configuration

**[Go to Optimal Test Suite →](OPTIMAL_TESTS.md)**

---

## Quick Start

### Comprehensive Tests (36 tests, ~3-6 hours)

```bash
# Activate environment
source venv/bin/activate

# Run all tests
python scripts/comprehensive_whisper_test.py --audio "data/mamak session scam.mp3"

# Analyze results
python scripts/analyze_test_results.py

# Generate presentation
python scripts/generate_presentation.py
```

### Optimal Tests (8 tests, ~40-60 minutes)

```bash
# Activate environment
source venv/bin/activate

# Preview tests
python scripts/optimal_whisper_test.py --audio "data/mamak session scam.mp3" --dry-run

# Run tests
python scripts/optimal_whisper_test.py --audio "data/mamak session scam.mp3"
```

---

## Test Coverage

### Comprehensive Test Suite

**Parameters tested:**
- 2 Models (Original Whisper, Malaysian Whisper)
- 3 Languages (auto, en, ms)
- 5 Beam sizes (1, 3, 5, 7, 10)
- 5 Temperatures (0.0, 0.2, 0.4, 0.6, 0.8)
- 4 Initial prompts (None, Context, Manglish, Informal)

**Total: 36 configurations**

**Outputs:**
- Raw test results JSON
- Detailed analysis report
- Executive summary slides
- Quick reference guide
- CSV export for custom charts

### Optimal Test Suite

**Parameters tested:**
- Malaysian Whisper model (specialized)
- Beam sizes (7, 10)
- Temperatures (0.0, 0.2)
- Initial prompts (None, Context)
- Hallucination suppression thresholds

**Total: 8 configurations**

**Outputs:**
- Test results JSON
- Individual transcripts (JSON + TXT)

---

## Results Analysis

### Comprehensive Tests

Results are automatically analyzed and formatted:

```
results/comprehensive_test/
├── test_results.json                    # Raw data
├── test_summary.md                      # Auto-generated summary
├── analysis_report.md                   # Detailed analysis
└── presentation/
    ├── executive_summary_slides.md      # Presentation deck
    ├── quick_reference.md               # Top configs
    └── test_results.csv                 # For Excel/charts
```

### Optimal Tests

Results are saved for manual review:

```
outputs/optimal_test/
├── OPT1.json                            # Transcription results
├── OPT1.txt                             # Plain text transcripts
├── OPT2.json
├── OPT2.txt
└── ...
└── optimal_test_results.json            # Summary
```

---

## Evaluation Metrics

Both test suites measure:

### Primary Metric: RTF (Real-Time Factor)
- **RTF < 1.0** = Faster than realtime ✅
- **RTF = 1.0** = Realtime speed
- **RTF > 1.0** = Slower than realtime

### Secondary Metrics
- Processing time (seconds)
- Number of segments generated
- Language detected
- Audio duration

### Quality (Manual Review)
- Transcription coherence
- Repetition/hallucination detection
- Code-switching accuracy
- Technical term handling

---

## Timeline Estimates

### Comprehensive Test Suite

| Phase | Duration | Can Run Unattended? |
|-------|----------|---------------------|
| Setup & validation | 10 min | No |
| Test execution (36 tests) | 3-6 hours | ✅ Yes |
| Analysis generation | 2 min | No |
| Presentation generation | 1 min | No |
| Manual quality review | 1-2 hours | No |
| **Total** | **4-9 hours** | **Most automated** |

### Optimal Test Suite

| Phase | Duration | Can Run Unattended? |
|-------|----------|---------------------|
| Setup & validation | 5 min | No |
| Test execution (8 tests) | 40-60 min | ✅ Yes |
| Manual quality review | 30-60 min | No |
| **Total** | **75-125 min** | **Most automated** |

---

## Prerequisites

### For Both Test Suites

1. **Virtual Environment Activated**
   ```bash
   source venv/bin/activate
   ```

2. **Audio File Available**
   ```bash
   ls -lh "data/mamak session scam.mp3"
   ```

3. **Dependencies Installed**
   ```bash
   pip install -r requirements.txt
   ```

4. **Sufficient Disk Space**
   - Comprehensive: ~100 MB
   - Optimal: ~50 MB

---

## Troubleshooting

### Common Issues

**"Audio file not found"**
```bash
# Check file exists (note: filename has spaces)
ls -la "data/mamak session scam.mp3"
```

**"Out of memory"**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

**Tests too slow**
```bash
# Comprehensive: Run subset
python scripts/comprehensive_whisper_test.py --audio "..." --limit 10

# Optimal: Already fast (8 tests only)
```

**Need to resume**
```bash
# Comprehensive: Resume from specific test
python scripts/comprehensive_whisper_test.py --audio "..." --start-from BS1

# Optimal: Use resume flag
python scripts/optimal_whisper_test.py --audio "..." --resume
```

For detailed troubleshooting, see [../TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

---

## Next Steps

1. **Choose your test suite** based on your needs
2. **Read the detailed guide**:
   - [Comprehensive Tests](COMPREHENSIVE_TESTS.md)
   - [Optimal Tests](OPTIMAL_TESTS.md)
3. **Activate environment** and run tests
4. **Review results** and apply findings

---

**Need setup help?** See [Getting Started Guide](../GETTING_STARTED.md)

**Need feature details?** See [User Guide](../USER_GUIDE.md)
