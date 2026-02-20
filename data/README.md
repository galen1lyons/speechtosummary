# Data Folder - Naming Convention

## Purpose
This folder contains audio files for testing the speech-to-summary pipeline.

## Naming Convention

**IMPORTANT:** Follow this naming convention to ensure the pipeline works correctly!

### Format
```
category_description.mp3
```

### Rules
1. **Use underscores** (`_`) instead of spaces
2. **Lowercase preferred** for consistency
3. **Be descriptive** but concise
4. **Category prefix** helps organize files

### Categories
- `base_` - Clean, standard Malaysian English recordings
- `chaos_` - Noisy environments, overlapping speech
- `interference_` - Background noise, other languages interfering
- `manglish_` - Heavy Manglish (Malaysian English) content
- `test_` - General test files
- `stress_` - Stress test files (long duration, complex scenarios)

### Examples
✅ **Good:**
- `base_mamak.mp3`
- `chaos_parliament.mp3`
- `interference_chinese.mp3`
- `manglish_office_meeting.mp3`
- `stress_test_multilingual.mp3`

❌ **Bad:**
- `HO LEE FAK! 😡😡😡😡.mp3` (emojis, special chars)
- `My Recording 2024.mp3` (spaces)
- `untitled.mp3` (not descriptive)

## Pipeline Behavior
- The pipeline will create outputs with the same base filename
- Output files go to `outputs/` folder:
  - `{filename}.txt` - Plain text transcript
  - `{filename}.json` - Transcript with timestamps
  - `{filename}.summary.md` - Summary with action items

## Current Test Files
- `base_mamak.mp3` - Clean mamak stall conversation
- `chaos_parliament.mp3` - Parliamentary debate with overlapping speech
- `interference_chinese.mp3` - Chinese language interference
- `malaysian_whisper_stress_test.mp3` - Comprehensive stress test
- `shouting.mp3` - High-volume speech test
