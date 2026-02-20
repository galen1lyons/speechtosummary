# Transcribe CLI Guide

**ONE script to rule them all.**

This unified CLI replaces 9 different scripts with one modular tool.

---

## 🎯 Quick Start

```bash
# Show help
python scripts/transcribe.py --help

# Show subcommands
python scripts/transcribe.py {transcribe,evaluate,compare,batch,validate}

# Show available configs
python scripts/transcribe.py validate
```

---

## 📋 Subcommands

### **1. transcribe** - Transcribe audio

```bash
# Basic usage (uses fw5_optimal config by default)
python scripts/transcribe.py transcribe --audio file.mp3

# With preset config
python scripts/transcribe.py transcribe \
  --audio file.mp3 \
  --config fw5_optimal

# Custom model
python scripts/transcribe.py transcribe \
  --audio file.mp3 \
  --backend faster-whisper \
  --model large-v3

# Specify output location
python scripts/transcribe.py transcribe \
  --audio file.mp3 \
  --output outputs/my_transcript
```

**Available configs:**
- `fw5_optimal` - Optimal settings (beam=7, strict VAD) ← **Recommended**
- `baseline` - Default settings (beam=5, no VAD)
- `large` - Large-v3 model for multilingual

---

### **2. evaluate** - Calculate WER/CER

```bash
# Evaluate transcription against human reference
python scripts/transcribe.py evaluate \
  --hypothesis outputs/result.txt \
  --reference human_transcribe.txt

# Save metrics to file
python scripts/transcribe.py evaluate \
  --hypothesis outputs/result.txt \
  --reference human_transcribe.txt \
  --output results/metrics.json
```

**Output:** WER, CER, word/char counts, and detailed metrics

---

### **3. compare** - Compare two models

```bash
# Compare base vs large-v3
python scripts/transcribe.py compare \
  --audio file.mp3 \
  --model1 base \
  --model2 large-v3

# Compare with reference for WER/CER
python scripts/transcribe.py compare \
  --audio file.mp3 \
  --model1 base \
  --model2 large-v3 \
  --reference human_transcribe.txt

# Compare different backends
python scripts/transcribe.py compare \
  --audio file.mp3 \
  --model1 base \
  --backend1 faster-whisper \
  --model2 base \
  --backend2 openai-whisper
```

**Output:** Side-by-side comparison table with WER, CER, RTF, and winner

---

### **4. batch** - Batch transcribe multiple files

```bash
# Transcribe all audio files in a directory
python scripts/transcribe.py batch \
  --input-dir data/ \
  --output-dir outputs/batch/

# With preset config
python scripts/transcribe.py batch \
  --input-dir data/ \
  --output-dir outputs/batch/ \
  --config fw5_optimal

# Custom model
python scripts/transcribe.py batch \
  --input-dir data/ \
  --output-dir outputs/batch/ \
  --backend faster-whisper \
  --model large-v3
```

**Supported formats:** `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`

---

### **5. validate** - Show available configurations

```bash
python scripts/transcribe.py validate
```

**Output:** List of preset configurations with descriptions

---

## 🔧 Common Workflows

### **Workflow 1: Transcribe with optimal settings**
```bash
python scripts/transcribe.py transcribe \
  --audio "data/Studio Sembang Slice.mp3" \
  --config fw5_optimal
```

### **Workflow 2: Transcribe and evaluate**
```bash
# Step 1: Transcribe
python scripts/transcribe.py transcribe \
  --audio file.mp3 \
  --output outputs/test

# Step 2: Evaluate
python scripts/transcribe.py evaluate \
  --hypothesis outputs/test.txt \
  --reference human_transcribe.txt
```

### **Workflow 3: Compare models to find best**
```bash
python scripts/transcribe.py compare \
  --audio file.mp3 \
  --model1 base \
  --model2 medium \
  --reference human_transcribe.txt \
  --output results/comparison.json
```

### **Workflow 4: Batch process dataset**
```bash
python scripts/transcribe.py batch \
  --input-dir data/recordings/ \
  --output-dir outputs/batch_results/ \
  --config fw5_optimal
```

---

## 📊 What Replaced What

| Old Script | New Command |
|------------|-------------|
| `validate_optimal_settings.py` | `python scripts/transcribe.py transcribe --config fw5_optimal` |
| `evaluate_with_ground_truth.py` | `python scripts/transcribe.py evaluate --hypothesis X --reference Y` |
| `compare_whisper.py` | `python scripts/transcribe.py compare --model1 X --model2 Y` |
| *(manual batch loop)* | `python scripts/transcribe.py batch --input-dir X --output-dir Y` |

**Old way:** 3+ scripts, each doing one thing
**New way:** 1 script, 5 subcommands, modular and reusable

---

## 🏗️ Architecture

**MBT Analogy:** One master training manual with different drill types

```
scripts/transcribe.py (thin CLI wrapper)
  └─ Calls reusable functions from src/

src/
├── transcribe_faster.py    ← FasterWhisper transcription
├── transcribe.py           ← OpenAI Whisper transcription
├── asr_metrics.py          ← WER/CER evaluation
├── comparison.py           ← Model comparison logic (NEW)
└── config.py               ← Configuration classes
```

**Key principle:** Scripts are thin, libraries are thick

---

## 🎛️ Preset Configurations

```python
# fw5_optimal (RECOMMENDED)
{
  'backend': 'faster-whisper',
  'model_name': 'base',
  'beam_size': 7,              # Optimal
  'vad_threshold': 0.7,        # Strict (reduces hallucinations)
  'min_speech_duration_ms': 500,
  'min_silence_duration_ms': 3000,
}

# baseline
{
  'backend': 'faster-whisper',
  'model_name': 'base',
  'beam_size': 5,              # Default
  'use_optimal_vad': False,
}

# large (multilingual)
{
  'backend': 'faster-whisper',
  'model_name': 'large-v3',    # Best for Chinese/Japanese
  'beam_size': 5,
  'vad_threshold': 0.5,
}
```

Add your own configs by editing the `CONFIGS` dict in `scripts/transcribe.py`

---

## 🚀 Next Steps

1. **Test it:** Try `python scripts/transcribe.py transcribe --audio "data/Studio Sembang Slice.mp3" --config fw5_optimal`

2. **Compare models:** Test base vs large-v3 on your audio

3. **Deprecate old scripts:** Once comfortable, move old scripts to `scripts/archive/`

4. **Add new features:** Need a new command? Add a subcommand instead of creating a new script!

---

**Last Updated:** 2026-02-19
