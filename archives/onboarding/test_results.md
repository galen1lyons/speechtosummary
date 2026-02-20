# Test Results Log

## Test Environment
- **Date:** 2026-01-29
- **Hardware:** [Your laptop model]
- **Device:** CPU / GPU (CUDA)
- **Whisper Model:** base

---

## Test 1: English Clear Audio

| Field | Value |
|-------|-------|
| **File** | `test_audio/podcast_english.mp3` |
| **Duration** | X min |
| **Language** | en |
| **Model** | base |
| **Parameters** | beam_size=5, temp=0.0 |
| **Processing Time** | X min |
| **Speed Ratio** | X.Xx realtime |
| **Quality** | Excellent / Good / Fair |

**Notes:**
- 

---

## Test 2: Manglish/Singlish Meeting

| Field | Value |
|-------|-------|
| **File** | `test_audio/meeting_manglish.mp3` |
| **Duration** | X min |
| **Language** | en |
| **Model** | base |
| **Parameters** | beam_size=5, temp=0.0, initial_prompt="Malaysian English business meeting" |
| **Processing Time** | X min |
| **Speed Ratio** | X.Xx realtime |
| **Quality** | Excellent / Good / Fair |

**Notes:**
- 

---

## Test 3: Chinese (Malaysian/Mainland)

| Field | Value |
|-------|-------|
| **File** | `test_audio/chinese.mp3` |
| **Duration** | X min |
| **Language** | zh |
| **Model** | base |
| **Parameters** | beam_size=5, temp=0.0 |
| **Processing Time** | X min |
| **Speed Ratio** | X.Xx realtime |
| **Quality** | Excellent / Good / Fair |

**Notes:**
- 

---

## Test 4: Japanese

| Field | Value |
|-------|-------|
| **File** | `test_audio/japanese.mp3` |
| **Duration** | X min |
| **Language** | ja |
| **Model** | base |
| **Parameters** | beam_size=5, temp=0.0 |
| **Processing Time** | X min |
| **Speed Ratio** | X.Xx realtime |
| **Quality** | Excellent / Good / Fair |

**Notes:**
- 

---

## Test 5: Malay

| Field | Value |
|-------|-------|
| **File** | `test_audio/malay.mp3` |
| **Duration** | X min |
| **Language** | ms |
| **Model** | base |
| **Parameters** | beam_size=5, temp=0.0 |
| **Processing Time** | X min |
| **Speed Ratio** | X.Xx realtime |
| **Quality** | Excellent / Good / Fair |

**Notes:**
- 

---

## Parameter Comparison Test

### Beam Size Comparison (same audio file)

| Beam Size | Processing Time | Accuracy Notes |
|-----------|-----------------|----------------|
| 5 (default) | X min | |
| 10 | X min | |

### Initial Prompt Test (accented audio)

| With Prompt | Without Prompt |
|-------------|----------------|
| Accuracy: X% | Accuracy: X% |
| Notes: | Notes: |

---

## GPU vs CPU Benchmark

| Device | Audio Duration | Processing Time | Speed Ratio |
|--------|----------------|-----------------|-------------|
| CPU (laptop) | 13 min | 24 min | 1.8x realtime |
| GPU (Colab T4) | 13 min | ~4 min | 0.3x realtime |

**Speedup:** ~6x faster on GPU

---

## Summary of Findings

### Audio Types Tested
- [ ] Clear podcasts
- [ ] Meetings with Manglish/Singlish accents
- [ ] Chinese (Malaysian and Mainland)
- [ ] Japanese
- [ ] Malay
- [ ] Noisy/reverb environments

### Key Observations
1. 
2. 
3. 

### Recommendations
- **Clear audio:** beam_size=5, temperature=0.0
- **Noisy audio:** beam_size=10, temperature fallback
- **Accented speech:** beam_size=5 + initial_prompt
