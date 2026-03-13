# Speech-to-Summary Pipeline — Project Guide

## Project Overview

On-premise speech-to-summary pipeline for a Malaysian-based Japanese company. Handles English/Malay code-switching with local accents.

**Stack:**
- **ASR:** faster-whisper (CPU int8 or GPU float16)
- **Diarization:** pyannote.audio
- **Summarization:** Mistral 7B via llama-cpp-python
- **Preprocessing:** noisereduce + ffmpeg
- **Model comparison:** Multi-model benchmarking CLI (`python -m src.comparison`)

**Environment:** `venv` (not `.venv`)

---

## Architecture

```
Audio → Preprocess → Transcribe (full audio) → Diarize → Align/Merge → Summarize
```

**Critical lesson learned:** Transcribe first, then diarize. Never slice audio before transcription—it starves Whisper of decoder context and nearly doubles WER.

| Approach | WER |
|----------|-----|
| Diarize → Slice → Transcribe per segment | 57% |
| Transcribe full → Diarize → Align | 34% |

---

## Optimal Settings

These are baked into `transcribe_faster.py` defaults:

```python
model_name = "small"       # Best accuracy/speed tradeoff
beam_size = 7              # vs default 5
vad_threshold = 0.7        # Strict, reduces hallucinations
min_speech_duration_ms = 500
min_silence_duration_ms = 3000
compute_type = "int8"      # CPU-friendly
```

**Model benchmarks (DEDM meeting audio, no diarization):**

| Model | Device | Compute | WER | CER | RTF |
|-------|--------|---------|-----|-----|-----|
| base | cpu | int8 | 49.46% | 58.43% | 0.207x |
| base | cpu | float32 | 43.46% | 45.93% | 0.157x |
| base | cuda | float16 | 45.50% | 53.16% | 0.037x |
| **small** | **cpu** | **int8** | **30.31%** | **32.69%** | **0.453x** |
| small | cpu | float32 | 31.74% | 37.95% | 0.403x |
| small | cuda | float16 | 30.59% | 34.65% | 0.036x |
| medium | cpu | int8 | 43.46% | 55.86% | 1.203x |
| medium | cpu | float32 | 39.78% | 49.88% | 1.230x |
| medium | cuda | float16 | 40.19% | 50.57% | 0.065x |
| large-v3 | cuda | float16 | 43.53% | 47.61% | 0.180x |
| large-v3-turbo | cuda | float16 | 32.29% | 32.17% | 0.048x |

**Key findings:**
- `small` int8 on CPU is the best overall (30.31% WER)
- Larger models don't help — domain mismatch (Malaysian accents, code-switching) is the bottleneck, not model capacity
- `medium` is worse than `small` and slower than real-time on CPU
- `large-v3` hallucinates on this audio; `large-v3-turbo` is competitive but doesn't beat `small`
- WER varies by recording (24-31% range observed across different audio files)

---

## Project Structure

```
src/
├── pipeline.py          # Main orchestrator
├── transcribe_faster.py # faster-whisper backend (primary)
├── transcribe.py        # openai-whisper backend (fallback)
├── diarize.py           # Speaker diarization + RTTM I/O
├── preprocess.py        # Denoise, normalize, slice
├── summarize.py         # Mistral 7B summarization
├── comparison.py        # Multi-model comparison CLI
├── config.py            # Dataclass configs
├── evaluation/
│   ├── asr_metrics.py   # WER, CER, RTF
│   └── diarization_metrics.py  # DER, JER
tests/
├── test_pipeline.py     # Integration tests
├── test_transcribe_faster.py
├── test_diarize.py
└── ...
outputs/
├── comparisons/         # Model comparison results (JSON + transcripts)
│   └── transcripts/     # Extracted plain-text transcripts per model/config
├── reference/human/     # Human reference transcripts for WER/CER
```

---

## BE RIGID: Always Follow

### Code Quality
- Run `pytest` before committing
- Run linters before committing
- Never commit TODOs or FIXMEs
- Never hardcode API keys — use `.env` only
- Ensure `.env` is in `.gitignore`

### Commits
Use conventional commits:
```
feat: add speaker statistics export
fix: handle empty diarization gracefully
refactor: transcribe full audio first, then diarize
test: add pipeline integration tests
docs: update CLAUDE.md for current architecture
```

### Workflow
- Explore codebase before writing code
- Use `think hard` during implementation
- For new features: plan → PLAN.md → implement → delete PLAN.md

### Environment
```bash
source venv/bin/activate
python -m pytest tests/ -v
python -m src.pipeline --audio input.mp3 --enable-diarization
```

---

## BE FLEXIBLE: Apply Judgment

### Testing Strategy
After 5-10 tests in a long suite, pause and ask:
- What patterns are emerging?
- Is there an obvious problem to diagnose now?
- Should we pivot?

**Proactively suggest:** "I've run 7/36 tests. Should we pause to analyze?"

### Efficiency Over Completeness
If early results show a clear pattern → diagnose now, don't wait.
If a problem is obvious → stop and fix.

### Architecture Decisions
When something isn't working, question the approach:
- "Is this a parameter problem or an architecture problem?"
- "Are we measuring the right thing?"

The diarize-first failure was an architecture problem. No amount of prompt tuning or hotwords could fix it.

---

## Common Tasks

### Run Full Pipeline
```bash
python -m src.pipeline \
  --audio data/meeting.mp3 \
  --backend faster-whisper \
  --whisper-model small \
  --enable-diarization \
  --content-type meeting
```

### Evaluate Against Reference
```bash
python -m src.pipeline \
  --audio data/meeting.mp3 \
  --reference-transcript reference.txt \
  --reference-rttm reference.rttm \
  --run-name eval_run
```

### Run Tests
```bash
python -m pytest tests/ -v
python -m pytest tests/test_pipeline.py -v  # Integration only
```

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/pipeline.py` | Entry point, orchestrates everything |
| `src/transcribe_faster.py` | Where optimal ASR settings live |
| `src/comparison.py` | Multi-model benchmarking (`python -m src.comparison`) |
| `src/diarize.py` | `merge_diarization_with_transcript()` is critical |
| `src/config.py` | All config dataclasses with validation |
| `tests/test_pipeline.py` | Integration tests for the 3 main scenarios |

---

## Lessons Learned

1. **Architecture > Parameters** — No prompt or hotword could fix the 23pp WER gap from bad architecture.

2. **Measure early** — The diarize-first bug was caught by comparing WER between two runs, not by staring at code.

3. **Keep utilities testable** — Functions like `transcribe_segments_faster()` stayed in the codebase even after the refactor because they have tests and might be useful later.

4. **Graceful degradation** — Diarization can fail (missing HF token, model download issues). The pipeline continues without speaker labels rather than crashing.

5. **Model size doesn't always help** — `large-v3` hallucinated worse than `small` on Malaysian-accented audio. Domain mismatch > model capacity.

---

## Multi-Model Comparison

```bash
# Compare models on same audio with WER/CER/RTF
python -m src.comparison \
  --audio "data/dedm meeting audio test.mp3" \
  --models base small medium \
  --reference outputs/reference/human/dedm_meeting/dedm_meeting_human_plain.txt

# GPU with per-model device/compute settings
python -m src.comparison \
  --audio "data/dedm meeting audio test.mp3" \
  --models base small medium large-v3 large-v3-turbo \
  --devices cuda cuda cuda cuda cuda \
  --compute-types float16 float16 float16 float16 float16 \
  --reference outputs/reference/human/dedm_meeting/dedm_meeting_human_plain.txt
```

---

## What's Next

- [ ] Fine-tune `small` or `large-v3-turbo` on Malaysian English/Malay data to reduce WER
- [ ] Build GUI for comparing human vs machine transcripts with diff highlighting
- [ ] Word-level alignment for finer speaker boundaries
- [ ] Streaming transcription for real-time use

---

*Last updated: 2026-03-13*
