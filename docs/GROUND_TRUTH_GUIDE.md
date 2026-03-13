# Ground Truth Creation Guide

Create manual reference transcripts for WER/CER evaluation.

## Quick Start

1. Listen to the audio file
2. Transcribe exactly what you hear into a plain text file
3. Run evaluation:
```bash
python -m src.pipeline --audio data/meeting.mp3 \
  --reference-transcript data/your_ground_truth.txt
```

## Transcription Rules

**Do:**
- Transcribe word-for-word what you hear
- Include filler words (um, uh, like)
- Include repetitions if speaker actually repeats
- Use minimal punctuation (periods, commas, question marks)
- Mark unclear sections with `[unintelligible]`

**Don't:**
- Clean up or paraphrase speech
- Copy the model's output
- Add words that weren't said
- Over-punctuate

**Example:**
```
Oh today's episode is interesting I got scammed more about that in just a bit
```

## Format

Plain text, UTF-8, one continuous transcript. Timestamps are optional — the pipeline strips them automatically if present.

```
[0.00 - 10.00] Oh today's episode is interesting
[10.00 - 15.00] More about that in just a bit
```

## Time Estimates

| Approach | Time for 15 min audio |
|----------|----------------------|
| Full manual transcription | 3-5 hours |
| Whisper draft + manual correction | 1-2 hours |
| Sample sections only (2-3 min) | 30-60 min |

**Recommendation:** Start with 2-3 minute samples. Scale up if needed.

## Interpreting Results

- **WER < 15%**: Good — production-ready config
- **WER 15-25%**: Acceptable — consider larger model
- **WER > 25%**: Needs improvement — check audio quality, model size, or language setting

**Error types:** High substitutions = wrong words. High deletions = dropped words (check VAD). High insertions = hallucinations (check beam size, VAD).

## Reference Files

Existing ground truth transcripts in `outputs/reference/human/`:
- `dedm_meeting/dedm_meeting_human_plain.txt`
- `studio_sembang/studio_sembang_human_transcribe.txt`
- `mamak_session_scam/mamak_session_scam_human_transcribe_4mins.txt`
