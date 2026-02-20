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
- ⚠️ Manglish accuracy best with faster-whisper base (VAD tuned)
- ℹ️ For non-Malaysian languages, use Original Whisper models

> **Note on Malaysian Whisper:** `mesolitica/malaysian-whisper-*` HuggingFace models
> were evaluated and found unviable for this use case (RTF 23.7x real-time,
> 1,027 hallucinations on 9-minute audio). Use faster-whisper base with the
> `fw5_optimal` config instead.

### Known Limitations
- **Overlapping speech:** Diarization accuracy drops when multiple people talk simultaneously
- **Background noise:** Heavy noise impacts transcription quality
- **Long meetings (>2 hours):** Higher memory usage, consider splitting audio
- **Summarization:** Mesolitica T5-base produces extractive-style summaries; quality depends on transcript clarity
- **Accents:** Strong non-standard accents may reduce accuracy

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

# Check results (each run gets its own timestamped folder)
ls outputs/runs/
```

**That's it!** Your transcript and summary are in `outputs/runs/<run-folder>/` ✅

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

Each pipeline run writes to a dedicated folder under `outputs/runs/`:

```
outputs/runs/
└── 20260220T153045Z__base_mamak__fw-base-int8-general/
    ├── transcript.txt           # Plain text transcript with timestamps
    ├── transcript.json          # Transcript with segment-level detail
    ├── summary.md               # Manglish-aware summary (Mesolitica T5-base)
    └── manifest.json            # Run metadata (model, flags, paths, metrics)
```

With `--enable-diarization`:
```
    ├── speakers.txt             # Speaker-labeled transcript
    ├── speaker_stats.json       # Speaking time per speaker
    └── diarization.rttm         # RTTM output (for DER/JER evaluation)
```

With `--reference-transcript`:
```
    └── asr_metrics.json         # WER/CER/RTF evaluation results
```

Evaluation runs (with `--output-dir outputs/campaigns/eval`) follow the same structure.

---

## 🚀 Common Use Cases

### Basic Transcription (faster-whisper, default)
```bash
python -m src.pipeline --audio data/meeting.mp3
```

### With Speaker Identification
```bash
python -m src.pipeline --audio data/meeting.mp3 --enable-diarization
```

### Using OpenAI Whisper Backend
```bash
python -m src.pipeline --audio data/meeting.mp3 --backend openai-whisper
```

### Evaluate Against Human Transcript (WER/CER)
```bash
python -m src.pipeline --audio data/meeting.mp3 \
  --reference-transcript outputs/reference/human/mamak_session_scam/FW5_Human_Transcribe.txt \
  --output-dir outputs/campaigns/eval
```

### Evaluate Diarization (DER/JER)
```bash
# Evaluate against a reference RTTM
python scripts/transcribe.py diarize-evaluate \
  --hypothesis outputs/runs/.../diarization.rttm \
  --reference outputs/reference/human/studio_sembang/studio_sembang_human_diarization.txt

# Or inline during pipeline run
python -m src.pipeline --audio data/meeting.mp3 \
  --enable-diarization --reference-rttm data/meeting_reference.rttm
```

### Evaluate Transcription via CLI
```bash
python scripts/transcribe.py evaluate \
  --hypothesis outputs/runs/.../transcript.txt \
  --reference outputs/reference/human/mamak_session_scam/FW5_Human_Transcribe.txt
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
  --reference outputs/reference/human/mamak_session_scam/FW5_Human_Transcribe.txt
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
- **OpenAI Whisper** (fallback) - Original Whisper, required for HuggingFace models
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
- ✅ Comprehensive test suites (140 tests)
- ✅ Complete documentation

**Recent Updates:**
- Feb 2026: Fixed RuntimeWarning from pipeline entry point imported in `src/__init__`
- Feb 2026: Dead-code purge — removed compatibility shims, archived stale scripts
- Feb 2026: `results/` retired; eval metrics moved to `outputs/reference/eval_metrics/`
- Feb 2026: Timestamp stripping applied to all reference-transcript reads (phantom WER fix)
- Feb 2026: Switched default backend to faster-whisper (better VAD, hallucination control)
- Feb 2026: Replaced BART-CNN summarization with Mesolitica T5-base (Manglish-native)
- Feb 2026: Upgraded diarization to pyannote.audio 4.0 (speaker-diarization-community-1)
- Feb 2026: Added DER/JER diarization metrics with RTTM I/O

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
│   ├── transcribe.py            # OpenAI Whisper backend
│   ├── transcribe_faster.py     # faster-whisper backend (default)
│   ├── summarize.py             # Mesolitica T5-base summarization
│   ├── diarize.py               # Speaker diarization (pyannote.audio 4.0)
│   ├── comparison.py            # Model comparison utilities
│   ├── evaluation/
│   │   ├── asr_metrics.py       # WER, CER, RTF
│   │   └── diarization_metrics.py  # DER, JER
│   ├── utils.py                 # Shared utilities
│   ├── exceptions.py            # Custom exceptions
│   └── logger.py                # Logging configuration
│
├── data/                        # Audio test files
│   └── README.md                # Naming conventions
│
├── outputs/                     # Generated transcripts & reports
│   ├── README.md                # Output structure guide
│   ├── runs/                    # Production pipeline runs
│   ├── campaigns/               # Evaluation & legacy experiment runs
│   └── reference/               # Human transcripts & legacy eval metrics
│
├── scripts/                     # Utility scripts
│   ├── README.md                # Complete scripts guide
│   ├── TRANSCRIBE_CLI_GUIDE.md  # Full CLI documentation
│   ├── transcribe.py            # Unified CLI (transcribe, compare, evaluate, batch)
│   ├── validate_setup.py        # Environment validation
│   ├── check_project.py         # Project health check
│   ├── setup_hf_token.sh        # HuggingFace token setup
│   ├── experimental/            # Import audit tool (sync_project.py)
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
python -m src.pipeline --audio data/meeting.mp3 --whisper-model tiny
```

**For more:** See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

**Ready to start?** → [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)

**Have questions?** → [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

**Want to evaluate models?** → [scripts/TRANSCRIBE_CLI_GUIDE.md](scripts/TRANSCRIBE_CLI_GUIDE.md)
