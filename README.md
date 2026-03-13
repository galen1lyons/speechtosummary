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
- ✅ Audio preprocessing: spectral denoising + volume normalization
- ✅ Speaker diarization with pyannote.audio 4.0 (identifies who spoke)
- ✅ Dual transcription backends (faster-whisper default + OpenAI Whisper)
- ✅ Local summarization with Mistral 7B Instruct via llama-cpp-python (no API key needed)
- ✅ Diarization evaluation metrics (DER/JER) with RTTM I/O
- ✅ Multi-model comparison CLI (`python -m src.comparison`)
- ✅ Comprehensive test suite (182 tests)

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
- **Background noise:** Denoising (noisereduce) helps with steady hum/HVAC; heavy music or transient noise may still impact quality
- **Long meetings (>2 hours):** Higher memory usage; consider splitting audio
- **Summarization:** Mistral 7B uses a grounding prompt ("summarize ONLY what is in the transcript") to reduce hallucination, but still treat summaries as drafts to be reviewed
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
  - Mistral 7B summarizer (Q4_K_M): +4.4GB RAM (CPU-only)

**Note:** First run downloads models (~1-5GB for Whisper; ~4.4GB for Mistral 7B summarizer)

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
    ├── preprocessed.wav         # Denoised + normalized 16kHz mono WAV
    ├── transcript.txt           # Plain text transcript with timestamps
    ├── transcript.json          # Transcript with segment-level detail
    ├── summary.md               # Grounded summary (Mistral 7B Instruct, local)
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

### Disable Preprocessing (skip denoising)
```bash
# Skip denoising but keep volume normalization
python -m src.pipeline --audio data/meeting.mp3 --no-denoise

# Skip preprocessing entirely (format conversion to 16kHz WAV still runs)
python -m src.pipeline --audio data/meeting.mp3 --disable-preprocessing
```

### Using OpenAI Whisper Backend
```bash
python -m src.pipeline --audio data/meeting.mp3 --backend openai-whisper
```

### Use a Local Mistral GGUF (skip auto-download)
```bash
python -m src.pipeline --audio data/meeting.mp3 \
  --summary-model-path /path/to/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
```

### Evaluate Against Human Transcript (WER/CER)
```bash
python -m src.pipeline --audio data/meeting.mp3 \
  --reference-transcript outputs/reference/human/mamak_session_scam/FW5_Human_Transcribe.txt \
  --output-dir outputs/campaigns/eval
```

### Evaluate Diarization (DER/JER)
```bash
python -m src.pipeline --audio data/meeting.mp3 \
  --enable-diarization --reference-rttm data/meeting_reference.rttm
```

**See [User Guide](docs/USER_GUIDE.md) for all options**

---

## 🧪 Model Testing & Comparison

Compare multiple Whisper model sizes on the same audio with WER/CER/RTF metrics:

### Compare Multiple Models
```bash
# Compare base, small, medium on CPU
python -m src.comparison \
  --audio "data/dedm meeting audio test.mp3" \
  --models base small medium \
  --reference outputs/reference/human/dedm_meeting/dedm_meeting_human_plain.txt

# Compare all models on GPU with per-model settings
python -m src.comparison \
  --audio "data/dedm meeting audio test.mp3" \
  --models base small medium large-v3 large-v3-turbo \
  --devices cuda cuda cuda cuda cuda \
  --compute-types float16 float16 float16 float16 float16 \
  --reference outputs/reference/human/dedm_meeting/dedm_meeting_human_plain.txt
```

Results are saved as JSON to `outputs/comparisons/` with extracted transcripts in `outputs/comparisons/transcripts/`.

### Benchmark Results (DEDM Meeting Audio)

| Model | Device | Compute | WER | CER | RTF |
|-------|--------|---------|-----|-----|-----|
| base | cpu | int8 | 49.46% | 58.43% | 0.207x |
| **small** | **cpu** | **int8** | **30.31%** | **32.69%** | **0.453x** |
| medium | cpu | int8 | 43.46% | 55.86% | 1.203x |
| large-v3 | cuda | float16 | 43.53% | 47.61% | 0.180x |
| large-v3-turbo | cuda | float16 | 32.29% | 32.17% | 0.048x |

`small` with int8 on CPU remains the best overall. Larger models don't improve due to domain mismatch (Malaysian accents, code-switching).

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
- **[Scripts Guide](scripts/README.md)** - Utility scripts explained
- **[Model Size Experiment](docs/MODEL_SIZE_EXPERIMENT.md)** - base vs small vs medium results
- **[Prompt & Hotwords Experiment](docs/PROMPT_HOTWORDS_EXPERIMENT.md)** - initial_prompt and hotwords ablation

---

## 🎯 What's Inside?

**Architecture (transcribe-first):**
```
Audio → Preprocess → Transcribe (full audio) → Diarize (full audio) → Align/Merge → Summarize
```

Transcription and diarization run independently on the same preprocessed audio, then timestamps are used to map speaker labels onto transcript segments.

When diarization is disabled:
```
Audio → Preprocess → Transcribe full audio → Summarize
```

> **Lesson learned:** Transcribe first, then diarize. Diarizing first fragments audio into short clips, starving Whisper of decoder context. Measured: **57% WER** with diarize-first vs **30% WER** with transcribe-first.

**Core Technologies:**
- **noisereduce + soundfile** - Spectral noise reduction and volume normalization
- **ffmpeg-python** - Format conversion and audio processing
- **faster-whisper** (default) - CTranslate2-optimized Whisper with VAD
- **OpenAI Whisper** (fallback) - Required for HuggingFace models (e.g. Malaysian Whisper)
- **pyannote.audio 4.0** - Speaker diarization with speaker-diarization-community-1
- **Mistral 7B Instruct v0.3** (llama-cpp-python, Q4_K_M GGUF) - Grounded local summarization, no API key needed; auto-downloads ~4.4GB on first run
- **ASR Metrics** - WER, CER, RTF for transcription quality
- **Diarization Metrics** - DER, JER with RTTM I/O for speaker accuracy

**Key Features:**
- ✅ Audio denoising before transcription (noisereduce spectral reduction)
- ✅ Dual transcription backends (faster-whisper + OpenAI Whisper)
- ✅ Automatic language detection
- ✅ Speaker diarization with RTTM export
- ✅ Manglish handling (code-switching)
- ✅ Local summarization (no cloud API required)
- ✅ Configurable parameters (beam size, VAD threshold, compute type, etc.)
- ✅ Comprehensive evaluation metrics (WER, CER, RTF, DER, JER)

---

## 📊 Project Status

**Working:**
- ✅ Audio preprocessing (denoising + normalization)
- ✅ Transcription (faster-whisper + OpenAI Whisper)
- ✅ Summaries (local Mistral 7B Instruct via llama-cpp-python, no API key)
- ✅ Speaker diarization (pyannote.audio 4.0)
- ✅ Diarization evaluation (DER/JER with RTTM I/O)
- ✅ Manglish support
- ✅ Multi-language support
- ✅ ASR metrics (WER, CER, RTF)
- ✅ Multi-model comparison CLI with benchmark results
- ✅ Comprehensive test suite (182 tests)
- ✅ Complete documentation

**Recent Updates:**
- Mar 2026: Multi-model comparison across base/small/medium/large-v3/large-v3-turbo (CPU int8, CPU float32, GPU float16)
- Mar 2026: Transcribe-first architecture implemented — WER reduced from 57% to 30%
- Mar 2026: Added initial_prompt and hotwords support for faster-whisper
- Feb 2026: Replaced mT5 with Mistral 7B Instruct v0.3 (Q4_K_M GGUF, llama-cpp-python)
- Feb 2026: Added audio preprocessing (noisereduce spectral denoising, peak normalization)
- Feb 2026: Switched default backend to faster-whisper (better VAD, hallucination control)
- Feb 2026: Upgraded diarization to pyannote.audio 4.0

---

## 📁 Project Structure

```
speechtosummary/
├── README.md                    # ← You are here
├── CLAUDE.md                    # Development standards
│
├── src/                         # Core pipeline code
│   ├── pipeline.py              # Main entry point (orchestrator)
│   ├── config.py                # Pipeline configuration (WhisperConfig, SummaryConfig, PreprocessConfig)
│   ├── preprocess.py            # Audio denoising, normalization, segment slicing
│   ├── transcribe.py            # OpenAI Whisper backend
│   ├── transcribe_faster.py     # faster-whisper backend (default, recommended)
│   ├── summarize.py             # Mistral 7B Instruct summarization (llama-cpp-python)
│   ├── diarize.py               # Speaker diarization (pyannote.audio 4.0)
│   ├── comparison.py            # Multi-model comparison CLI
│   ├── evaluation/
│   │   ├── asr_metrics.py       # WER, CER, RTF
│   │   └── diarization_metrics.py  # DER, JER
│   ├── utils.py                 # Shared utilities
│   ├── exceptions.py            # Custom exceptions
│   └── logger.py                # Logging configuration
│
├── data/                        # Audio test files
│
├── outputs/                     # Generated transcripts & reports
│   ├── runs/                    # Production pipeline runs
│   ├── campaigns/               # Evaluation runs
│   ├── comparisons/             # Model comparison results (JSON)
│   │   └── transcripts/         # Extracted plain-text transcripts per model/config
│   └── reference/human/         # Human reference transcripts for WER/CER
│
├── scripts/                     # Utility scripts
│   ├── README.md                # Scripts guide
│   ├── validate_setup.py        # Environment validation
│   ├── check_project.py         # Project health check
│   └── setup_hf_token.sh        # HuggingFace token setup
│
├── docs/                        # Documentation
│   ├── README.md                # Documentation hub
│   ├── GETTING_STARTED.md       # Installation guide
│   ├── USER_GUIDE.md            # Feature reference
│   ├── TROUBLESHOOTING.md       # Fix common issues
│   ├── FASTER_WHISPER_OPTIMIZATION.md
│   ├── GROUND_TRUTH_GUIDE.md
│   ├── MODEL_SIZE_EXPERIMENT.md # base vs small vs medium results
│   └── PROMPT_HOTWORDS_EXPERIMENT.md
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

**Want to compare models?** → `python -m src.comparison --help`
