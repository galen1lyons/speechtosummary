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

- **[Testing Documentation](testing/README.md)** - Model comparison and testing suites
- **[Comprehensive Test Suite](testing/COMPREHENSIVE_TESTS.md)** - 36-test systematic evaluation
- **[Optimal Test Suite](testing/OPTIMAL_TESTS.md)** - 8-test focused evaluation

---

## 🛠️ For Maintainers

**Working on the codebase?**

- **Project Context**: See [CLAUDE.md](../CLAUDE.md) for development standards
- **Data Files**: See [data/README.md](../data/README.md) for naming conventions
- **Archive**: Historical docs in [ARCHIVE/](ARCHIVE/)

---

## 🎓 Demos and Examples

- **[Google Colab Demo](../demo/GOOGLE_COLAB.md)** - Run in the browser
- **[Whisper Parameters Guide](../demo/WHISPER_PARAMETERS.md)** - Parameter reference

---

## 🚀 Quick Reference

### Common Commands

```bash
# Basic transcription
python -m src.pipeline --audio meeting.mp3

# With speaker identification
python -m src.pipeline --audio meeting.mp3 --diarize

# Better accuracy (slower)
python -m src.pipeline --audio meeting.mp3 --whisper-model medium
```

### File Locations

- **Audio files**: `data/`
- **Transcripts**: `outputs/`
- **Test results**: `results/`
- **Scripts**: `scripts/`

---

## 🆘 Need Help?

1. Check [Troubleshooting](TROUBLESHOOTING.md) for common issues
2. Review [User Guide](USER_GUIDE.md) for feature documentation
3. See [Getting Started](GETTING_STARTED.md) for setup issues

---

**Documentation Version**: 2.0
**Last Updated**: 2026-02-09
