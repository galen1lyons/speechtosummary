# Scripts Directory

**Modular CLI for transcription tasks.**

---

## 🎯 Main CLI (Use This)

### **`transcribe.py`** - Unified Modular CLI

**ONE script for all transcription tasks.**

```bash
# Show help
python scripts/transcribe.py --help

# Transcribe audio
python scripts/transcribe.py transcribe --audio file.mp3 --config fw5_optimal

# Evaluate against reference
python scripts/transcribe.py evaluate --hypothesis result.txt --reference human.txt

# Compare two models
python scripts/transcribe.py compare --audio file.mp3 --model1 base --model2 large-v3

# Batch transcribe
python scripts/transcribe.py batch --input-dir data/ --output-dir outputs/

# Show available configs
python scripts/transcribe.py validate
```

**📖 Full documentation**: See [TRANSCRIBE_CLI_GUIDE.md](TRANSCRIBE_CLI_GUIDE.md)

---

## 🔧 Diagnostic Tools

### **`check_project.py`** - Project Health Check
```bash
python scripts/check_project.py
```
- Checks directory structure
- Validates Python environment
- Verifies dependencies
- **Run this first when troubleshooting**

### **`probe.py`** - Hardware & Model Diagnostics
```bash
python scripts/probe.py
```
- Hardware inventory (RAM, GPU, disk)
- Model download and loading tests
- Memory requirement checks
- **Run before working with large models**

### **`validate_setup.py`** - Environment Validation
```bash
python scripts/validate_setup.py
```
- Checks all Python modules installed
- Verifies audio files exist
- Validates directory structure
- **Run this before starting work**

---

## ⚙️ Setup Scripts

### **`setup_hf_token.sh`** - HuggingFace Token Setup
```bash
bash scripts/setup_hf_token.sh
```
- Interactive HuggingFace token setup
- Required for speaker diarization
- Creates `.env` file with `HF_TOKEN`
- Safe - never commits token to git

---

## 📁 Archive Directories

### `archive/deprecated_2026-02-19/`
Scripts replaced by the modular CLI:
- `validate_optimal_settings.py` → `transcribe.py transcribe`
- `evaluate_with_ground_truth.py` → `transcribe.py evaluate`
- `compare_whisper.py` → `transcribe.py compare`
- `run_tests.sh` → Old testing framework launcher
- `transcription_helper.sh` → Ground truth helper

See [archive/deprecated_2026-02-19/README.md](archive/deprecated_2026-02-19/README.md) for migration guide.

### `archive/`
Other archived scripts:
- `verify_parameter_fix.py` - One-time verification
- `verify_optimal_test_setup.sh` - One-time setup
- `create_stress_test.sh` - Stress test utility

### `experimental/`
Experimental features:
- `summarize_mallam.py` - Malaysian LLM summarization
- `sync_project.py` - Import audit tool

---

## 🚀 Quick Start

### First-Time Setup
```bash
# 1. Validate environment
python scripts/validate_setup.py

# 2. Setup HuggingFace token (for diarization)
bash scripts/setup_hf_token.sh

# 3. Check project health
python scripts/check_project.py
```

### Transcription Workflow
```bash
# Transcribe with optimal settings
python scripts/transcribe.py transcribe \
  --audio "data/your_file.mp3" \
  --config fw5_optimal

# Evaluate accuracy
python scripts/transcribe.py evaluate \
  --hypothesis outputs/your_file.txt \
  --reference human_transcribe.txt

# Compare models
python scripts/transcribe.py compare \
  --audio "data/your_file.mp3" \
  --model1 base \
  --model2 large-v3 \
  --reference human_transcribe.txt
```

---

## 📊 Available Configurations

Run `python scripts/transcribe.py validate` to see all configs:

- **`fw5_optimal`** - Optimal FasterWhisper (beam=7, strict VAD) ← **Recommended**
- **`baseline`** - Default FasterWhisper (beam=5, no VAD)
- **`large`** - FasterWhisper large-v3 (multilingual)

---

## 🆘 Common Issues

### "ModuleNotFoundError"
**Solution:** Activate virtual environment
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Audio file not found"
**Solution:** Use quotes for filenames with spaces
```bash
python scripts/transcribe.py transcribe --audio "data/file with spaces.mp3"
```

### Out of memory
**Solution:** Force CPU mode
```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/transcribe.py transcribe --audio file.mp3
```

---

## 📚 Documentation

- [Transcribe CLI Guide](TRANSCRIBE_CLI_GUIDE.md) - Full CLI documentation
- [Project Docs](../docs/) - General documentation
- [Troubleshooting](../docs/TROUBLESHOOTING.md) - Common issues

---

**Last Updated:** 2026-02-19
