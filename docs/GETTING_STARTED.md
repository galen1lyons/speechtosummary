# Getting Started Guide

This guide will help you install and run your first transcription.

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** (Python 3.12 recommended)
- **FFmpeg** - Required for audio processing
- **HuggingFace Token** - Optional, only needed for speaker diarization

### Install FFmpeg

```bash
# Ubuntu/WSL
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

---

## Installation

### Step 1: Navigate to Project

```bash
cd /home/dedmtiintern/speechtosummary
```

**Note:** This project uses `venv/` (not `.venv`)

### Step 2: Create Virtual Environment

```bash
# Create virtual environment (first time only)
python3 -m venv venv
```

### Step 3: Activate Virtual Environment

```bash
# Linux/macOS/WSL
source venv/bin/activate

# Windows
venv\Scripts\activate
```

You should see `(venv)` prefix in your prompt:
```
(venv) user@host:~/speechtosummary$
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- OpenAI Whisper
- transformers (for Malaysian Whisper)
- pyannote.audio (for speaker diarization)
- ASR evaluation metrics
- Other required packages

---

## Setup for Speaker Diarization (Optional)

If you want to identify who said what in your meetings:

### Step 1: Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Copy the token (starts with `hf_...`)

### Step 2: Configure Token

**Option 1: Using the setup script (recommended)**

```bash
bash scripts/setup_hf_token.sh
```

**Option 2: Manual setup**

Create a `.env` file in the project root:

```bash
# Create .env file
echo "HF_TOKEN=hf_your_token_here" > .env
```

**IMPORTANT:** Never commit your `.env` file to git. It should already be in `.gitignore`.

---

## Your First Transcription

### Step 1: Prepare Audio File

Place your audio file in the `data/` folder:

```bash
# Copy your file
cp /path/to/your/meeting.mp3 data/

# Or verify test file exists
ls -lh "data/mamak session scam.mp3"
```

### Step 2: Run Basic Transcription

```bash
python -m src.pipeline --audio data/your_meeting.mp3
```

The pipeline will:
1. Load the Whisper model
2. Transcribe your audio
3. Generate a summary
4. Save results to `outputs/` folder

### Step 3: Check Your Results

Each run creates a timestamped folder inside `outputs/runs/`:

```bash
# List run folders
ls outputs/runs/

# Enter the latest run folder (tab-complete the folder name)
ls outputs/runs/20260220T153045Z__your_meeting__fw-base-int8-general/
```

**Expected outputs inside the run folder:**
- `transcript.txt` - Plain text transcript with timestamps
- `transcript.json` - Transcript with segment-level detail
- `summary.md` - Executive summary with action items
- `manifest.json` - Run metadata (model, flags, paths, timings)

---

## Common Variations

### Better Accuracy (Slower)

```bash
python -m src.pipeline --audio data/meeting.mp3 --whisper-model medium
```

### With Speaker Identification

```bash
python -m src.pipeline --audio data/meeting.mp3 --enable-diarization
```

### Specify Language (Faster)

```bash
# English
python -m src.pipeline --audio data/meeting.mp3 --language en

# Malay
python -m src.pipeline --audio data/meeting.mp3 --language ms

# Auto-detect (default)
python -m src.pipeline --audio data/meeting.mp3 --language auto
```

---

## Troubleshooting

### Error: "No module named 'whisper'"

**Cause:** Virtual environment not activated or dependencies not installed

**Fix:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "ffmpeg not found"

**Cause:** FFmpeg not installed

**Fix:** See [Prerequisites](#install-ffmpeg) section above

### Error: "Audio file not found"

**Cause:** Wrong working directory or incorrect file path

**Fix:**
```bash
# Check current directory
pwd  # Should be: /home/dedmtiintern/speechtosummary

# Verify file exists (note: filename has spaces)
ls -la "data/mamak session scam.mp3"
```

### Tests Taking Too Long

**Solution:** Use a smaller/faster model

```bash
# Fast but lower accuracy
python -m src.pipeline --audio data/meeting.mp3 --whisper-model tiny

# Good balance (default)
python -m src.pipeline --audio data/meeting.mp3 --whisper-model base
```

### Out of Memory

**Solution:** Use a smaller model or force CPU mode

```bash
# Use smaller model
python -m src.pipeline --audio data/meeting.mp3 --whisper-model base

# Force CPU mode (disable GPU)
export CUDA_VISIBLE_DEVICES=""
python -m src.pipeline --audio data/meeting.mp3
```

---

## Next Steps

Now that you've completed your first transcription:

1. **Explore Features** - Read the [User Guide](USER_GUIDE.md) to learn about:
   - Different Whisper models
   - Speaker diarization options
   - Advanced Whisper parameters
   - Quality metrics

2. **Compare Models** - Use `scripts/transcribe.py compare` (see [Scripts Guide](../scripts/README.md))

3. **Troubleshoot Issues** - See [Troubleshooting Guide](TROUBLESHOOTING.md) for more help

---

## Quick Reference

```bash
# Activate environment
source venv/bin/activate

# Basic transcription
python -m src.pipeline --audio data/meeting.mp3

# With all features
python -m src.pipeline \
  --audio data/meeting.mp3 \
  --whisper-model medium \
  --enable-diarization \
  --language auto

# Check outputs
ls outputs/runs/
```

---

**Need more help?** See [Troubleshooting](TROUBLESHOOTING.md) or [User Guide](USER_GUIDE.md)
