# Faster-Whisper Optimization

**Date:** February 2026

## Summary

Achieved **99.2% hallucination reduction** (119 → 1) by tuning VAD and beam search parameters. No performance penalty.

## Optimal Configuration (FW5)

```python
model = WhisperModel("base", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    audio_path,
    language="en",
    beam_size=7,                         # increased from default 5
    vad_filter=True,
    vad_parameters={
        "threshold": 0.7,                # stricter than default 0.5
        "min_speech_duration_ms": 500,    # longer than default 250
        "min_silence_duration_ms": 3000   # longer than default 2000
    }
    # DO NOT set temperature=0.0 — causes catastrophic looping
)
```

These settings are the defaults in `src/transcribe_faster.py`.

## Parameter Impact

| Parameter | Default | Optimal | Impact |
|-----------|---------|---------|--------|
| beam_size | 5 | 7 | High — reduces hallucinations |
| vad_threshold | 0.5 | 0.7 | Very high — filters noise segments |
| min_speech_duration | 250ms | 500ms | High — removes spurious short segments |
| min_silence_duration | 2000ms | 3000ms | Medium — better segmentation |
| temperature | default array | **leave default** | Critical — setting 0.0 causes infinite loops |

## Test Results

| Config | Hallucinations | Segments | Time | RTF |
|--------|----------------|----------|------|-----|
| FW1 (defaults) | 119 | 191 | 441s | 0.48x |
| FW4 (beam=7) | 5 | 210 | 446s | — |
| **FW5 (beam=7 + VAD)** | **1** | **320** | **434s** | **0.47x** |

## Critical: Temperature Bug

Setting `temperature=0.0` explicitly causes catastrophic hallucinations — the model loops infinitely ("Hello. Hello. Hello..." for 13 minutes). The default fallback array `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` allows the model to recover from uncertainty. Never override it.

## March 2026 Update

Model size experiments showed `small` (23.84% WER) outperforms `base` (34.67%) and even `medium` (26.23%) on CPU int8. Full-audio mode (no diarization) is critical — diarization fragments audio into short clips, increasing WER by ~23 points. See `MODEL_SIZE_EXPERIMENT.md`.
