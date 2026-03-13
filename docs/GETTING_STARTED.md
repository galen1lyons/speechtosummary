# Getting Started

## Prerequisites

- Python 3.8+ (3.12 recommended)
- FFmpeg: `sudo apt-get install ffmpeg` (Ubuntu/WSL) or `brew install ffmpeg` (macOS)
- HuggingFace token (optional, only for speaker diarization)

## Installation

```bash
cd /home/dedmtiintern/speechtosummary
python3 -m venv venv          # uses venv/, not .venv/
source venv/bin/activate
pip install -r requirements.txt
```

## Diarization Setup (Optional)

1. Get a token at https://huggingface.co/settings/tokens (read access)
2. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create `.env` in project root: `echo "HF_TOKEN=hf_your_token_here" > .env`

## First Run

```bash
source venv/bin/activate
python -m src.pipeline --audio data/your_meeting.mp3 --whisper-model small --language en
```

Results are saved to a timestamped folder under `outputs/runs/`:
- `transcript.txt` — plain text with timestamps
- `transcript.json` — segment-level detail
- `summary.md` — structured summary with action items
- `manifest.json` — run metadata

## Common Variations

```bash
# With speaker identification
python -m src.pipeline --audio data/meeting.mp3 --enable-diarization --min-speakers 2

# Evaluate against reference transcript
python -m src.pipeline --audio data/meeting.mp3 \
  --reference-transcript outputs/reference/human/your_ref.txt

# Quick draft (lower accuracy, faster)
python -m src.pipeline --audio data/meeting.mp3 --whisper-model base
```

See [User Guide](USER_GUIDE.md) for full options. See [Troubleshooting](TROUBLESHOOTING.md) for common errors.
