# Documentation Hub

Welcome to the speechtosummary documentation. This page helps you navigate to the right guide based on your needs.

---

## 👋 For New Users

**Just getting started?** Follow these guides in order:

1. **[Getting Started Guide](GETTING_STARTED.md)** - Install and run your first transcription
2. **[User Guide](USER_GUIDE.md)** - Learn about features and options
3. **[Troubleshooting](TROUBLESHOOTING.md)** - Fix common issues

---

## 🔬 For Researchers

**Comparing Whisper models or running tests?**

- **[faster-whisper Optimization](FASTER_WHISPER_OPTIMIZATION.md)** - VAD and parameter tuning guide
- **[Ground Truth Guide](GROUND_TRUTH_GUIDE.md)** - Creating reference transcripts for WER/CER evaluation
- **[Transcribe CLI Guide](../scripts/TRANSCRIBE_CLI_GUIDE.md)** - Full CLI for compare/evaluate subcommands

---

## 🛠️ For Maintainers

**Working on the codebase?**

- **Project Context**: See [CLAUDE.md](../CLAUDE.md) for development standards
- **Data Files**: See [data/README.md](../data/README.md) for naming conventions and test files
- **Outputs**: See [outputs/README.md](../outputs/README.md) for run folder structure

---

## 🚀 Quick Reference

### Common Commands

```bash
# Basic transcription (faster-whisper, default)
python -m src.pipeline --audio data/meeting.mp3

# With speaker identification
python -m src.pipeline --audio data/meeting.mp3 --enable-diarization

# Better accuracy (slower)
python -m src.pipeline --audio data/meeting.mp3 --whisper-model medium

# Evaluate against human transcript
python -m src.pipeline --audio data/meeting.mp3 \
  --reference-transcript outputs/reference/human/mamak_session_scam/FW5_Human_Transcribe.txt
```

### File Locations

- **Audio files**: `data/`
- **Pipeline outputs**: `outputs/runs/` (production) or `outputs/campaigns/eval/` (evaluation)
- **Human references**: `outputs/reference/human/`
- **Scripts**: `scripts/`

---

## 🆘 Need Help?

1. Check [Troubleshooting](TROUBLESHOOTING.md) for common issues
2. Review [User Guide](USER_GUIDE.md) for feature documentation
3. See [Getting Started](GETTING_STARTED.md) for setup issues

---

**Documentation Version**: 2.1
**Last Updated**: 2026-02-20
