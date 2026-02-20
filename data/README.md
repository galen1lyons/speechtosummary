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

| File | Size | Notes |
|------|------|-------|
| `base_mamak.mp3` | 45 MB | Clean mamak stall conversation |
| `chaos_parliament.mp3` | 3.4 MB | Parliamentary debate with overlapping speech |
| `interference_chinese.mp3` | 12 MB | Chinese language interference |
| `malaysian_whisper_stress_test.mp3` | 9.2 MB | Comprehensive stress test |
| `shouting.mp3` | 3.6 MB | High-volume speech test |
| `mamak session scam.mp3` | 9.2 MB | ⚠️ Spaces in filename — human transcript available |
| `studio sembang james wan.mp3` | 5.4 MB | ⚠️ Spaces in filename — human transcript available |

> **⚠️ Naming violations:** `mamak session scam.mp3` and `studio sembang james wan.mp3` do not follow the
> `category_description.mp3` convention. Rename them if re-adding to the test suite.
