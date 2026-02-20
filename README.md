# Speech to Summary 🎤→📝

**Automatically transcribe Malaysian office meetings and generate structured summaries.**

Specialized for Manglish (Malaysian English + Malay code-switching), with speaker identification and action item extraction.

---

## 🎯 What This Does

**Input:** Audio recording of a meeting (MP3, WAV, etc.)

**Output:**
- 📄 **Transcript** - Who said what, with timestamps
- 📋 **Summary** - Local Manglish-aware summary (no API key required)
- 👥 **Speaker Diarization** - Speaker labels + RTTM output
- 📊 **Metrics** - ASR metrics (WER, CER, RTF) and diarization metrics (DER, JER)

**Special Features:**
- ✅ Handles Manglish (Malaysian English + Malay mixing)
- ✅ Speaker diarization with pyannote.audio 4.0 (identifies who spoke)
- ✅ Dual transcription backends (faster-whisper default + OpenAI Whisper)
- ✅ Local summarization with Mesolitica T5-base (no API key needed)
- ✅ Diarization evaluation metrics (DER/JER) with RTTM I/O
- ✅ Production-ready with comprehensive testing

---

## 🌍 Language Support & Known Limitations

### Optimized For
- **Malaysian English** (Manglish)
- **Malay** (Bahasa Malaysia)
- **Code-switching** (English ↔ Malay in same conversation)

### Other Languages
Whisper supports 90+ languages, but accuracy varies:
- ✅ English, Spanish, Chinese, etc. work well with Original Whisper
- ⚠️ Manglish accuracy best with `mesolitica/malaysian-whisper-*`
- ℹ️ For non-Malaysian languages, use Original Whisper models

### Known Limitations
- **Overlapping speech:** Diarization accuracy drops when multiple people talk simultaneously
- **Background noise:** Heavy noise impacts transcription quality
- **Long meetings (>2 hours):** Higher memory usage, consider splitting audio
- **Summarization:** Mesolitica T5-base produces extractive-style summaries; quality depends on transcript clarity
- **Accents:** Strong non-standard accents may reduce accuracy
- **HuggingFace models:** Malaysian Whisper models require `--backend openai-whisper` (not compatible with faster-whisper)

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.12 recommended)
- **FFmpeg** for audio processing
  ```bash
  # Ubuntu/WSL
  sudo apt-get install ffmpeg

  # macOS
  brew install ffmpeg
  ```

### Hardware & Performance

- **CPU-only:** Works but slower (10-20x slower for medium+ models)
- **GPU (CUDA):** Strongly recommended for medium/large models + diarization
  - ~5-10x faster transcription
  - Required for real-time processing
- **Apple Silicon (M1/M2/M3):** Supported via CPU (decent performance)
- **Memory requirements:**
  - Tiny/Base models: ~4GB RAM
  - Medium model: ~8-12GB RAM
  - Large models: ~16GB+ RAM
  - Diarization: +2-4GB overhead

**Note:** First run downloads models (~1-5GB depending on size)

### Installation (First Time)

```bash
# 1. Navigate to project
cd speechtosummary

# 2. Create virtual environment (this project uses venv/)
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # Linux/macOS/WSL
# or: venv\Scripts\activate  # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

### Run Your First Transcription

```bash
# Activate environment (if not already active)
source venv/bin/activate

# Transcribe a meeting
python -m src.pipeline --audio data/your_meeting.mp3

# Check results
ls outputs/
```

**That's it!** Your transcript and summary are in `outputs/` ✅

---

## 🔑 API Keys & Environment Variables

### Required for Speaker Diarization

- `HF_TOKEN` - Hugging Face token with access to gated models (pyannote)
  - Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - Accept model terms at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

### Setup

```bash
# Add to .env file (recommended)
echo 'HF_TOKEN=hf_your_token_here' > .env
```

**Without HF_TOKEN:**
- ✅ Transcription works normally
- ✅ Summarization works normally (runs locally, no API key needed)
- ❌ Speaker diarization skipped

**Security:** `.env` is already in `.gitignore` - never commit tokens!

---

## 📂 What You Get

```
outputs/
├── your_meeting.txt               # Plain text transcript with timestamps
├── your_meeting.json              # Transcript with segment-level timestamps
├── your_meeting.summary.md        # Manglish-aware summary (Mesolitica T5-base)
├── your_meeting.speakers.txt      # Speaker-labeled transcript (with diarization)
├── your_meeting.speaker_stats.json # Speaker participation statistics
├── your_meeting.rttm              # RTTM diarization file (for DER/JER evaluation)
└── your_meeting.asr_metrics.json  # WER/CER metrics (when reference provided)
```

---

## 🚀 Common Use Cases

### Basic Transcription (faster-whisper, default)
```bash
python -m src.pipeline --audio meeting.mp3
```

### With Speaker Identification
```bash
python -m src.pipeline --audio meeting.mp3 --enable-diarization
# Outputs: meeting.speakers.txt, meeting.rttm, meeting.speaker_stats.json
```

### Using OpenAI Whisper Backend
```bash
python -m src.pipeline --audio meeting.mp3 --backend openai-whisper
```

### For Manglish with Malaysian Whisper (HuggingFace model)
```bash
python -m src.pipeline --audio meeting.mp3 \
  --backend openai-whisper \
  --whisper-model mesolitica/malaysian-whisper-base \
  --language auto
```

### Evaluate Diarization (DER/JER)
```bash
# Evaluate against a reference RTTM
python scripts/transcribe.py diarize-evaluate \
  --hypothesis outputs/meeting.rttm \
  --reference data/meeting_reference.rttm

# Or inline during pipeline run
python -m src.pipeline --audio meeting.mp3 \
  --enable-diarization --reference-rttm data/meeting_reference.rttm
```

### Evaluate Transcription (WER/CER)
```bash
python scripts/transcribe.py evaluate \
  --hypothesis outputs/meeting.txt \
  --reference data/meeting_human_transcribe.txt \
  --output outputs/meeting.asr_metrics.json
```

**See [User Guide](docs/USER_GUIDE.md) for all options**

---

## 🧪 Model Testing & Comparison

Want to compare Whisper models or optimize parameters? Use the unified `scripts/transcribe.py` CLI.

### Compare Two Models
```bash
# Compare Original Whisper base vs large-v3
python scripts/transcribe.py compare \
  --audio "data/meeting.mp3" \
  --model1 base \
  --model2 large-v3

# Compare with reference transcript for WER/CER
python scripts/transcribe.py compare \
  --audio "data/meeting.mp3" \
  --model1 base \
  --model2 large-v3 \
  --reference data/meeting_human_transcribe.txt
```

### Transcribe with Specific Config
```bash
# Optimal faster-whisper settings (beam=7, strict VAD)
python scripts/transcribe.py transcribe \
  --audio "data/meeting.mp3" \
  --config fw5_optimal

# See all available configs
python scripts/transcribe.py validate
```

**See [Scripts Guide](scripts/README.md) for all CLI options**

---

## 📚 Documentation

### For New Users
1. **[Getting Started Guide](docs/GETTING_STARTED.md)** - Complete installation walkthrough
2. **[User Guide](docs/USER_GUIDE.md)** - Features, options, workflows
3. **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues & fixes

### For Researchers/Testing
- **[faster-whisper Optimization](docs/FASTER_WHISPER_OPTIMIZATION.md)** - VAD and parameter tuning
- **[Ground Truth Guide](docs/GROUND_TRUTH_GUIDE.md)** - Creating reference transcripts for WER/CER

### Quick References
- **[Scripts Guide](scripts/README.md)** - All utility scripts explained
- **[Transcribe CLI Guide](scripts/TRANSCRIBE_CLI_GUIDE.md)** - Full CLI reference

---

## 🎯 What's Inside?

**Core Technologies:**
- **faster-whisper** (default) - CTranslate2-optimized Whisper with strict VAD, 99% hallucination reduction
- **OpenAI Whisper** (fallback) - Original Whisper, required for HuggingFace models (e.g., Malaysian Whisper)
- **pyannote.audio 4.0** - Speaker diarization with speaker-diarization-community-1
- **Mesolitica T5-base** - Local Manglish-aware summarization (no API key needed)
- **ASR Metrics** - WER, CER, RTF for transcription quality
- **Diarization Metrics** - DER, JER with RTTM I/O for speaker accuracy

**Key Features:**
- ✅ Dual transcription backends (faster-whisper + OpenAI Whisper)
- ✅ Automatic language detection
- ✅ Speaker diarization with RTTM export
- ✅ Manglish handling (code-switching)
- ✅ Local summarization (no cloud API required)
- ✅ Configurable parameters (beam size, VAD threshold, compute type, etc.)
- ✅ Comprehensive evaluation metrics (WER, CER, RTF, DER, JER)

---

## 📊 Project Status

**Production Ready:**
- ✅ Transcription working (faster-whisper + OpenAI Whisper)
- ✅ Summaries working (local Mesolitica T5-base, no API key)
- ✅ Speaker diarization working (pyannote.audio 4.0)
- ✅ Diarization evaluation (DER/JER with RTTM I/O)
- ✅ Manglish support
- ✅ Multi-language support
- ✅ ASR metrics (WER, CER, RTF)
- ✅ Comprehensive test suites
- ✅ Complete documentation

**Recent Updates:**
- Feb 2026: Switched default backend to faster-whisper (better VAD, hallucination control)
- Feb 2026: Replaced BART-CNN summarization with Mesolitica T5-base (Manglish-native)
- Feb 2026: Upgraded diarization to pyannote.audio 4.0 (speaker-diarization-community-1)
- Feb 2026: Added DER/JER diarization metrics with RTTM I/O
- Feb 2026: Documentation refactored and consolidated
- Feb 2026: Comprehensive 36-test framework complete

---

## 📁 Project Structure

```
speechtosummary/
├── README.md                    # ← You are here
├── CLAUDE.md                    # Development standards
│
├── src/                         # Core pipeline code
│   ├── pipeline.py              # Main entry point (orchestrator)
│   ├── config.py                # Pipeline configuration (WhisperConfig, SummaryConfig)
│   ├── transcribe.py            # OpenAI Whisper transcription
│   ├── transcribe_faster.py     # faster-whisper transcription (default)
│   ├── summarize.py             # Mesolitica T5-base summarization
│   ├── diarize.py               # Speaker diarization + DER/JER metrics
│   ├── asr_metrics.py           # ASR metrics (WER, CER, RTF)
│   ├── comparison.py            # Model comparison utilities
│   ├── utils.py                 # Shared utilities
│   ├── exceptions.py            # Custom exceptions
│   └── logger.py                # Logging configuration
│
├── data/                        # Audio test files
│   └── README.md                # Naming conventions
│
├── outputs/                     # Generated transcripts
│   ├── README.md                # Output structure guide
│   ├── comprehensive_test/      # Test suite outputs
│   └── optimal_test/            # Optimal test outputs
│
├── results/                     # Test analysis results
│   ├── README.md                # Results guide
│   ├── comprehensive_test/      # Analysis & presentations
│   └── optimal_test/            # Optimal test results
│
├── scripts/                     # Utility scripts
│   ├── README.md                # Complete scripts guide
│   ├── TRANSCRIBE_CLI_GUIDE.md  # Full CLI documentation
│   ├── transcribe.py            # Unified CLI (transcribe, compare, evaluate, batch)
│   ├── evaluate_human_ground_truth.py
│   ├── validate_setup.py        # Environment validation
│   ├── check_project.py         # Project health check
│   ├── probe.py                 # Hardware & model diagnostics
│   ├── setup_hf_token.sh        # HuggingFace token setup
│   └── archive/                 # Deprecated scripts (kept for reference)
│
├── docs/                        # Documentation
│   ├── README.md                # Documentation hub
│   ├── GETTING_STARTED.md       # Installation guide
│   ├── USER_GUIDE.md            # Feature reference
│   ├── TROUBLESHOOTING.md       # Fix common issues
│   ├── FASTER_WHISPER_OPTIMIZATION.md
│   └── GROUND_TRUTH_GUIDE.md
│
└── venv/                        # Python virtual environment
```

---

## 🆘 Need Help?

### First Steps
1. Check [Getting Started Guide](docs/GETTING_STARTED.md)
2. Try [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
3. Read [User Guide](docs/USER_GUIDE.md) for features

### Common Issues

**"No module named 'whisper'"**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**"ffmpeg not found"**
```bash
sudo apt-get install ffmpeg  # Ubuntu/WSL
```

**Too slow?**
```bash
# Use faster model
python -m src.pipeline --audio meeting.mp3 --whisper-model tiny
```

**For more:** See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 🔍 Two Main Workflows

### 1. **Use the Tool** (Transcribe Meetings)
```bash
source venv/bin/activate
python -m src.pipeline --audio data/meeting.mp3 --enable-diarization
# → Get transcript + summary + speaker labels in outputs/
```

### 2. **Test & Compare Models** (Research/Optimization)
```bash
./scripts/run_tests.sh
# → Compare models, find optimal settings
# → Results in results/comprehensive_test/
```

Choose based on your goal!

---

## 📖 Documentation Hub

**Start Here:** [docs/README.md](docs/README.md)

All documentation is organized by user type:
- **New Users** → Installation and first run
- **Researchers** → Model comparison and testing
- **Maintainers** → Development standards and architecture

Every directory has a README.md explaining its purpose and contents.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Free to use commercially, modify, and distribute with attribution.

---

## 🤝 Contributing

Contributions welcome!

- **Issues:** Report bugs or request features via GitHub Issues
- **Pull Requests:** Follow standards in [CLAUDE.md](CLAUDE.md)
- **Development:** See [docs/README.md](docs/README.md) for architecture

**Development standards:**
- Run tests before committing
- Follow conventional commits (`feat:`, `fix:`, etc.)
- No TODOs in committed code

---

**Ready to start?** → [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)

**Have questions?** → [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

**Want to test models?** → [docs/testing/README.md](docs/testing/README.md)
