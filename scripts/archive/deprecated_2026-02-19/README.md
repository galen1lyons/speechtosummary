# Deprecated Scripts (2026-02-19)

These scripts were replaced by the unified modular CLI: `scripts/transcribe.py`

---

## Why Deprecated

The old workflow had **9 separate scripts**, each doing one thing. This led to:
- Code duplication
- Inconsistent interfaces
- Hard to maintain
- Scripts created "willy nilly" for each new task

**Solution**: One modular CLI with subcommands (`scripts/transcribe.py`)

---

## Migration Guide

### `validate_optimal_settings.py` → `transcribe.py transcribe`

**Old:**
```bash
python scripts/validate_optimal_settings.py --audio file.mp3
```

**New:**
```bash
python scripts/transcribe.py transcribe --audio file.mp3 --config fw5_optimal
```

---

### `evaluate_with_ground_truth.py` → `transcribe.py evaluate`

**Old:**
```bash
python scripts/evaluate_with_ground_truth.py \
  --reference human.txt \
  --hypothesis result.txt
```

**New:**
```bash
python scripts/transcribe.py evaluate \
  --hypothesis result.txt \
  --reference human.txt
```

---

### `compare_whisper.py` → `transcribe.py compare`

**Old:**
```bash
python scripts/compare_whisper.py --audio file.mp3
```

**New:**
```bash
python scripts/transcribe.py compare \
  --audio file.mp3 \
  --model1 base \
  --model2 large-v3
```

---

### `run_tests.sh` → Not directly replaced

This was a launcher for the old comprehensive testing framework (`comprehensive_whisper_test.py`, `optimal_whisper_test.py`), which are already archived in `scripts/archive/whisper_comparison_framework/`.

For batch testing, use:
```bash
python scripts/transcribe.py batch \
  --input-dir data/ \
  --output-dir outputs/
```

---

### `transcription_helper.sh` → Not replaced

This was a helper for creating manual ground truth transcriptions. If you need to create manual transcriptions, you can still use this script from the archive.

---

## If You Need These Scripts

These scripts still work - they're just deprecated in favor of the modular CLI.

To use them:
```bash
cd scripts/archive/deprecated_2026-02-19/
python validate_optimal_settings.py --audio ../../data/file.mp3
```

But we recommend using the new `scripts/transcribe.py` instead.

---

**Archived**: 2026-02-19
**Reason**: Replaced by modular CLI
**New CLI**: `scripts/transcribe.py` (see `scripts/TRANSCRIBE_CLI_GUIDE.md`)
