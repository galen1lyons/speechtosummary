# Archives

This directory contains historical work that led to the current FasterWhisper-based pipeline. Organized by campaign/exploration phase.

---

## 📁 Directory Structure

### `faster_whisper_exploration/`
**Purpose**: Iterative testing that identified optimal FasterWhisper settings

**Contents**:
- `test_faster_whisper.py` - Initial baseline test (FW1)
- `test_faster_whisper_tuning.py` - Parameter sweep exploration (FW2-FW4)
- `test_faster_whisper_final.py` - Combined best settings test (FW5)

**Findings**:
- FW1 (baseline): 119 hallucinations
- FW2 (strict VAD, vad_threshold=0.7): 0 hallucinations
- FW4 (beam_size=7): 5 hallucinations
- **FW5 (combined: beam=7 + vad=0.7)**: Optimal configuration

**Current Status**: Superseded by `scripts/validate_optimal_settings.py` which uses FW5 config

---

### `whisper_comparison_framework/`
**Purpose**: Systematic comparison of Original Whisper vs Malaysian Whisper models

**Contents**:
- Scripts:
  - `comprehensive_whisper_test.py` - 36-test systematic evaluation
  - `optimal_whisper_test.py` - 8-test focused evaluation
  - `analyze_test_results.py` - Result analysis
  - `generate_presentation.py` - Presentation generator
- Docs:
  - `docs/COMPREHENSIVE_TESTS.md` - 36-test documentation
  - `docs/OPTIMAL_TESTS.md` - 8-test documentation

**Current Status**: Framework was designed but results were not preserved. Superseded by FasterWhisper optimization work.

---

### `onboarding/`
**Purpose**: Early onboarding materials for Google Colab experimentation

**Contents**:
- `GOOGLE_COLAB.md` - Copy-paste cells for Colab testing
- `WHISPER_PARAMETERS.md` - Parameter reference guide
- `test_results.md` - Results template

**Current Status**: Outdated. Replaced by `docs/FASTER_WHISPER_OPTIMIZATION.md`

---

## 🔄 Migration Path

If you need to reference this historical work:

1. **FasterWhisper evolution**: See `faster_whisper_exploration/`
2. **Current optimal settings**: See `docs/FASTER_WHISPER_OPTIMIZATION.md`
3. **Active validation**: Use `scripts/validate_optimal_settings.py`

---

**Last Updated**: 2026-02-19
