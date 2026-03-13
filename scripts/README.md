# Scripts

Utility scripts for setup and diagnostics. **Not for transcription** — use `python -m src.pipeline` instead.

---

## Setup

### `setup_hf_token.sh`
Interactive HuggingFace token setup for speaker diarization.
```bash
bash scripts/setup_hf_token.sh
```

Creates `.env` file with `HF_TOKEN`. Required for `--enable-diarization`.

---

## Diagnostics

### `validate_setup.py`
Check Python environment and dependencies.
```bash
python scripts/validate_setup.py
```

Run this first when setting up or troubleshooting.

### `check_project.py`
Project health check — directory structure, imports, configs.
```bash
python scripts/check_project.py
```

---

## Transcription

**Don't look here.** Use the main pipeline:
```bash
# Basic transcription
python -m src.pipeline --audio meeting.mp3

# With diarization
python -m src.pipeline --audio meeting.mp3 --enable-diarization

# With evaluation
python -m src.pipeline --audio meeting.mp3 --reference-transcript ref.txt

# Full options
python -m src.pipeline --help
```

See `CLAUDE.md` for recommended settings.
