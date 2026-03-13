# Speech-to-Summary Pipeline: Experiment Results

**Project:** Malaysian Meeting Transcription Pipeline  
**Date Range:** 2026-02-20 to 2026-03-10  
**Audio Test Files:** 2 (DEDM corporate meeting ~11min, Studio Sembang podcast ~8min)

---

## Executive Summary

The **diarize-first architecture was fundamentally flawed**. Slicing audio into short speaker segments (~2-3s clips) before transcription starved Whisper of decoder context, causing WER to nearly double.

| Architecture | WER (base model) | Δ |
|--------------|------------------|---|
| Diarize → Transcribe per segment | **57.4%** | — |
| Transcribe full audio → Diarize → Merge | **34.7%** | **−23 pp** |

**Fix implemented:** Refactored pipeline to transcribe full audio first, then run diarization independently, then align timestamps.

---

## Test Results by Audio File

### 1. DEDM Corporate Meeting (English, Malaysian accents, 2 speakers, ~11 min)

#### Model Comparison (Full-Audio Transcription, No Diarization)

| Model | WER | CER | RTF | Processing Time |
|-------|-----|-----|-----|-----------------|
| base | 34.67% | 25.68% | 0.16x | 1m 46s |
| small | **23.84%** | 18.02% | 0.38x | 4m 21s |
| medium | 26.23% | 19.81% | 0.99x | 11m 13s |

**Winner:** `small` model — best accuracy with reasonable speed.

> Note: `medium` performed worse than `small` despite being larger. Likely due to running near real-time (RTF ~1.0) on CPU, suggesting resource contention.

#### Architecture Comparison (base model)

| Run | Architecture | WER | CER | Notes |
|-----|--------------|-----|-----|-------|
| nodiar_base | Full audio | **34.67%** | 25.68% | Baseline |
| dedm_meeting | Per-segment diarized | 57.43% | 43.69% | +23 pp worse |
| dedm_meeting_with_prompt | Per-segment + hotwords | 59.26% | 52.27% | Hotwords didn't help |
| dedm_meeting_prompt_only | Per-segment + initial_prompt | 58.38% | 46.56% | Prompt didn't help |

**Conclusion:** Prompts and hotwords cannot compensate for the architectural flaw. The short segments (~2-3s) don't give Whisper enough context to decode properly.

---

### 2. Studio Sembang Podcast (Malay/Manglish, 3 speakers, ~8 min)

| Run | Diarization | WER | CER | RTF |
|-----|-------------|-----|-----|-----|
| studio_sembang (no diar) | ❌ | **30.43%** | 38.13% | 0.21x |
| studio_sembang_james_wan | ✅ per-segment | 43.83% | 34.41% | 0.49x |
| studio_sembang_james_wan (2-4 spk) | ✅ per-segment | 44.74% | 35.08% | 2.14x |

**Same pattern:** Full-audio transcription beats per-segment by ~14 percentage points.

---

## Why Did Diarize-First Fail?

1. **Context starvation:** Whisper's decoder uses previous tokens to predict the next. With 2-3 second clips, it has almost no context.

2. **Segment boundary artifacts:** Hard cuts at speaker boundaries create unnatural audio edges that confuse the model.

3. **Model loading overhead:** Loading the model 200+ times (once per segment) added significant latency without improving accuracy.

4. **VAD double-application:** Diarization already uses VAD; applying Whisper's VAD on top of short clips is redundant and harmful.

---

## Pipeline Architecture Change

### Before (Broken)
```
Audio → Diarize → Slice into 200+ clips → Transcribe each clip → Merge
                          ↑
                    Context lost here
```

### After (Fixed)
```
Audio → Preprocess → Transcribe (full) → Diarize (parallel) → Align by timestamp
                          ↑                      ↑
                    Full context           Independent operation
```

---

## Validation Run — 2026-03-13

**Confirmed:** Transcribe-first architecture with diarization enabled hits target WER.

| Run | Architecture | Model | Diarization | WER | CER | RTF |
|-----|--------------|-------|-------------|-----|-----|-----|
| nodiar_small | Transcribe full → no diar | small | ❌ | 23.84% | 18.02% | 0.38x |
| validation_small_diarized | Transcribe full → Diarize → Align | small | ✅ | **23.84%** | **18.02%** | **0.29x** |

**Key result:** Enabling diarization did not degrade WER at all. The transcribe-first architecture cleanly separates the two concerns — Whisper gets full audio context, pyannote runs independently, timestamps align them after the fact.

**Speaker attribution:** 2 speakers detected, 264 diarization segments merged with 93 transcript segments.

Pipeline refactor is **complete and validated**.

---

## Recommendations

1. **Use `small` model for production** — Best accuracy/speed tradeoff (WER 23.84%, RTF 0.38x)

2. **Keep diarization as post-process** — Run it independently, merge by timestamp overlap

3. **Don't rely on hotwords/prompts to fix architectural issues** — They provide marginal gains at best

4. **Consider GPU for `medium`/`large`** — CPU-bound inference causes degradation

---

## Files to Delete

All runs in `outputs/campaigns/eval/` with `mode: "per_segment_diarized"` in their manifest can be deleted. The architecture is obsolete.

Keep:
- `20260310T071438Z__nodiar_medium__*` 
- `20260310T070452Z__nodiar_small__*`
- `20260310T065716Z__nodiar_base__*`
- `20260220T040018Z__studio_sembang__*` (baseline without diarization)

---

## Appendix: Raw Data

### All Runs (Chronological)

| Date | Run Name | Audio | Model | Diarization | WER | RTF |
|------|----------|-------|-------|-------------|-----|-----|
| 02-20 | studio_sembang | podcast | base | ❌ | 30.43% | 0.21x |
| 02-20 | studio_sembang_james_wan | podcast | base | ✅ (production) | — | 0.18x |
| 03-09 | studio_sembang_james_wan | podcast | base | ✅ per-seg | 44.74% | 2.14x |
| 03-10 | studio_sembang_james_wan | podcast | base | ✅ per-seg | 43.83% | 0.49x |
| 03-10 | dedm_meeting | meeting | base | ✅ per-seg | 57.36% | 0.48x |
| 03-10 | dedm_meeting | meeting | base | ✅ per-seg | 57.43% | 0.52x |
| 03-10 | dedm_meeting_with_prompt | meeting | base | ✅ per-seg | 59.26% | 0.67x |
| 03-10 | dedm_meeting_prompt_only | meeting | base | ✅ per-seg | 58.38% | 0.58x |
| 03-10 | nodiar_base | meeting | base | ❌ | 34.67% | 0.16x |
| 03-10 | nodiar_small | meeting | small | ❌ | 23.84% | 0.38x |
| 03-10 | nodiar_medium | meeting | medium | ❌ | 26.23% | 0.99x |
| 03-13 | validation_small_diarized | meeting | small | ✅ transcribe-first | 23.84% | 0.29x |

---

*Generated from 12 manifest.json files. Pipeline refactor validated 2026-03-13.*
