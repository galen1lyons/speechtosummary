# Experiment: initial_prompt + hotwords on faster-whisper (base)

**Date:** March 2026
**Audio:** `data/dedm meeting audio test.mp3` (11.4 min, Malaysian-accented English, AMR/robotics domain)
**Model:** faster-whisper `base`, int8, CPU
**Reference:** human transcript (dedm_meeting_human_plain.txt)

---

## Background

Two issues discovered after the first full pipeline run:

1. **Issue 1 (Hallucinations/WER):** The `base` model produced garbled transcription on technical domain audio (WER 57.43%). `initial_prompt` and `hotwords` parameters existed in `WhisperConfig` but were never wired to `model.transcribe()`. Fix: connected them (commit `8b9e750`).
2. **Issue 2 (Garbage summary):** `llama-cpp-python` was not installed, causing fallback to extractive summarization. Fix: install confirmed, Mistral 7B Q4_K_M now auto-downloaded and cached.

---

## Experiment Design

Three pipeline runs, identical settings except prompting parameters:

| Run | `--initial-prompt` | `--hotwords` | Run dir slug |
|-----|--------------------|--------------|--------------|
| Baseline | — | — | `dedm_meeting` |
| Prompt only | "Corporate meeting discussing AMR strategy, autonomous mobile robots, safety compliance, fleet management software." | — | `dedm_meeting_prompt_only` |
| Prompt + hotwords | same as above | `AMR,AGV,ROS,ISO,fleet,docking` | `dedm_meeting_with_prompt` |

All runs: diarization enabled, min/max speakers = 2, beam_size = 7, VAD threshold = 0.7.

---

## Results

| Metric | Baseline | Prompt only | Prompt + hotwords |
|--------|----------|-------------|-------------------|
| **WER** | 57.43% | 58.38% (+0.95) | 59.26% (+1.83) |
| **CER** | 43.69% | 46.56% (+2.87) | 52.27% (+8.58) |
| RTF | 0.52x | 0.58x | 0.67x |
| Segments transcribed | 195 | ~195 | ~195 |

---

## Key Findings

### 1. `initial_prompt` alone: mildly hurts WER on `base`
- WER +0.95%, CER +2.87% vs baseline.
- The `base` model is small enough that a long initial_prompt shifts decoder attention toward prompt vocabulary, causing substitution errors where prompt-adjacent words are hallucinated over actual speech.
- Substitutions: 630 (prompt only) vs 581 (baseline).

### 2. `hotwords` causes severe CER degradation (+8.58%)
- The hotword string `"AMR,AGV,ROS,ISO,fleet,docking"` was transcribed verbatim into the output as a repeated token sequence: `"AMR,AGV,ROS,O,LECISO,LECISO,LECISO,LECISO,LECISO"`.
- This is a known faster-whisper behavior: the `hotwords` string is injected as a pseudo-initial-prompt; on small models with short clips, the model can loop on these tokens.
- Deletions dropped (132 → 92) but insertions spiked (95 → 92 for prompt-only; hotwords caused further distortion).

### 3. Summary quality improved despite worse WER
- With `initial_prompt`, Mistral correctly identified: "AMR (Autonomous Mobile Robots) strategy", "fleet management software", "manufacturing execution systems", "electromagnetic compatibility".
- Without prompt, the summary described vague "D-A-R or P-S-O-C-C-T" and "Hulk and the Wheel Eimer".
- Conclusion: `initial_prompt` helps summarization even when it hurts transcription WER, because domain vocabulary in correctly-recognized segments gets reinforced.

### 4. Mistral 7B generates proper summaries (Issue 2 resolved)
- First run with Mistral produced 3000-char structured prose with **Executive Summary**, **Key Discussion Points**, **Action Items**, **Decisions Made** sections.
- Previously: extractive fallback copy-pasted raw transcript sentences.
- Model cached at: `~/.cache/huggingface/hub/models--bartowski--Mistral-7B-Instruct-v0.3-GGUF/`

---

## Recommendations

| Goal | Action |
|------|--------|
| Reduce WER on this audio | Upgrade model size: `base` → `small` or `medium` |
| Use `initial_prompt` | Safe for summarization benefit; avoid on `base` if WER is the primary metric |
| Use `hotwords` | Avoid on `base` model. May be safer on `large-v3` where the model is less susceptible to token looping. Test before using. |
| Production runs | Drop `--hotwords`; keep `--initial-prompt` optional if domain is known |

---

## Run Directories

All runs under `outputs/campaigns/eval/`:

- `20260310T040004Z__dedm_meeting__fw-base-int8-meeting` — baseline
- `20260310T052119Z__dedm_meeting_with_prompt__fw-base-int8-meeting` — prompt + hotwords
- `20260310T060821Z__dedm_meeting_prompt_only__fw-base-int8-meeting` — prompt only
