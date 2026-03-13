# Experiment: Model Size vs. WER (No Diarization, Full Audio)

**Date:** March 2026
**Audio:** `data/dedm meeting audio test.mp3` (11.4 min, Malaysian-accented English, AMR/robotics domain)
**Backend:** faster-whisper, int8, CPU
**Reference:** human transcript (dedm_meeting_human_plain.txt, 1468 words / 6375 chars)

---

## Motivation

Previous experiments (see `PROMPT_HOTWORDS_EXPERIMENT.md`) showed WER of 57.43% using `base` with diarization enabled. Two confounds were identified:

1. **Diarization fragments audio** into 225 short clips (~2-3s each), giving Whisper less decoder context per clip — likely hurting accuracy.
2. **Model size** hadn't been varied; `base` may simply be below the accuracy ceiling needed for this audio.

This experiment isolates model size by running full-audio transcription (no diarization) across `base`, `small`, and `medium`.

---

## Experiment Design

| Variable | Value |
|----------|-------|
| Backend | faster-whisper |
| Compute type | int8 |
| Device | CPU |
| Diarization | Disabled (full-audio mode) |
| Language | en |
| Beam size | 7 |
| VAD threshold | 0.7 |
| Models tested | base, small, medium |

All other settings at defaults. Runs executed sequentially, same machine.

---

## Results

| Model | WER | CER | RTF | Trans. time | Segments |
|-------|-----|-----|-----|-------------|----------|
| **base** | 34.67% | 25.68% | 0.16x | 106s (~1.8 min) | — |
| **small** | **23.84%** | **18.02%** | 0.38x | 261s (~4.4 min) | — |
| **medium** | 26.23% | 19.81% | 0.99x | 673s (~11.2 min) | 84 |

### Error breakdown

| Model | Substitutions | Deletions | Insertions | Total errors |
|-------|--------------|-----------|------------|--------------|
| base | 341 | 127 | 41 | 509 / 1468 |
| small | 190 | 124 | 36 | 350 / 1468 |
| medium | 166 | 184 | 35 | 385 / 1468 |

---

## Key Findings

### 1. Removing diarization was the single biggest WER improvement
- `base` with diarization: **57.43%** WER
- `base` without diarization: **34.67%** WER
- **Delta: −22.76 percentage points** from this change alone.
- Cause: full-audio mode gives Whisper the full 11-min context instead of 225 isolated 2-3s clips.

### 2. `small` is the accuracy sweet spot
- Best WER (23.84%) and best CER (18.02%) of all three models.
- **10.83 WER points better than base** (full-audio).
- 2.6x faster than medium.

### 3. `medium` underperforms `small` on this audio (+2.39% WER)
- Likely cause: at ~1x realtime on CPU, int8 quantization artifacts become a bottleneck for medium — the larger model's theoretical gains are partially offset by quantization noise at this compute budget.
- Medium's substitution count is lower (166 vs 190) but deletions spike (184 vs 124), suggesting it is more conservative but also drops more words.

### 4. Cumulative improvement from baseline to best config
| Config | WER | Improvement vs. original baseline |
|--------|-----|-----------------------------------|
| base + diarization (original) | 57.43% | — |
| base, no diarization | 34.67% | −22.76 pts |
| small, no diarization | **23.84%** | **−33.59 pts** |

---

## Recommendation

**Production config: `small`, no diarization.**

- Cuts WER by more than half vs. original baseline (23.84% vs. 57.43%)
- Runs in ~4-5 min total pipeline time (preprocessing + transcription + Mistral summary)
- No benefit from `medium` at this compute level; would need float16 or GPU to unlock medium's true capacity

**Next steps if further WER reduction is needed:**
- Test `large-v2` or `large-v3` (expect ~15-20% WER, 3-5x slower than medium on CPU)
- Test Malaysian Whisper fine-tuned model via `--backend openai-whisper` (designed for this accent)
- Try float16 compute type if GPU becomes available

---

## Run Directories

All under `outputs/campaigns/eval/`:

- `20260310T065716Z__nodiar_base__fw-base-int8-meeting`
- `20260310T070452Z__nodiar_small__fw-small-int8-meeting`
- `20260310T071438Z__nodiar_medium__fw-medium-int8-meeting`
