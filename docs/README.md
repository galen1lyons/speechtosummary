# Documentation

## Guides

- [Getting Started](GETTING_STARTED.md) — Install and run your first transcription
- [User Guide](USER_GUIDE.md) — Full CLI options, models, output format
- [Troubleshooting](TROUBLESHOOTING.md) — Common errors and fixes

## Research & Experiments

- [Model Size Experiment](MODEL_SIZE_EXPERIMENT.md) — base vs small vs medium WER comparison (March 2026)
- [Prompt & Hotwords Experiment](PROMPT_HOTWORDS_EXPERIMENT.md) — initial_prompt and hotwords effects (March 2026)
- [Faster-Whisper Optimization](FASTER_WHISPER_OPTIMIZATION.md) — VAD/beam tuning, 99% hallucination reduction (Feb 2026)
- [Ground Truth Guide](GROUND_TRUTH_GUIDE.md) — Creating reference transcripts for evaluation

## Other

- [Daikin Pipeline](DAIKIN_PIPELINE.md) — Separate future project spec (multilingual, on-prem)

## Quick Reference

```bash
# Recommended production command
python -m src.pipeline --audio data/meeting.mp3 --whisper-model small --language en

# With evaluation
python -m src.pipeline --audio data/meeting.mp3 --whisper-model small \
  --reference-transcript outputs/reference/human/your_ref.txt

# All options
python -m src.pipeline --help
```

## File Locations

- Audio files: `data/`
- Pipeline outputs: `outputs/runs/` (production) or `outputs/campaigns/eval/` (evaluation)
- Human references: `outputs/reference/human/`
