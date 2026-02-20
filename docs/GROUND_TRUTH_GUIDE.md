# Ground Truth Creation Guide

## Purpose

Create manual transcriptions to evaluate ASR model accuracy using proper metrics (WER/CER) instead of flawed hallucination counting.

## Why We Need This

Our previous evaluation method (hallucination counter) was flawed:
- ❌ Counted legitimate repetitive speech as errors
- ❌ Missed actual transcription mistakes
- ❌ No measurement of what model got wrong vs right

**Proper evaluation requires:**
- ✅ Manual ground truth (what was actually said)
- ✅ Model hypothesis (what model thinks was said)
- ✅ WER/CER metrics (scientific accuracy measurement)

## Quick Start

### 1. Listen to Audio
```bash
# Play the audio file
mpv data/"mamak session scam.mp3"
# or
vlc data/"mamak session scam.mp3"
```

### 2. Create Ground Truth File

**Option A: Full Transcript (Most Accurate)**
- Listen to entire audio
- Transcribe word-for-word what you hear
- Save as `data/ground_truth_mamak.txt`

**Option B: Sample Sections (Faster)**
- Pick 3-5 representative sections (1-2 minutes each)
- Transcribe those sections precisely
- Use for targeted evaluation

### 3. Transcription Guidelines

**DO:**
- ✅ Transcribe exactly what you hear
- ✅ Include filler words (um, uh, like)
- ✅ Include repetitions if speaker actually repeats
- ✅ Keep natural speech patterns
- ✅ Use simple punctuation (. , ?)

**DON'T:**
- ❌ Don't "clean up" the speech
- ❌ Don't add words that weren't said
- ❌ Don't skip unclear sections (mark with [unintelligible])
- ❌ Don't try to match the model output

**Example:**
```
Audio: "Oh today's episode is interesting I got scammed more about that in just a bit"

✅ Good ground truth:
Oh today's episode is interesting I got scammed more about that in just a bit

❌ Bad ground truth (cleaning up):
Oh, today's episode is interesting. I got scammed. More about that in just a bit.
```

### 4. Format Options

**Plain Text (Recommended for WER)**
```
Oh today's episode is interesting I got scammed
More about that in just a bit and you'll hear it here on today's show
What's up everybody how you guys doing
```

**With Timestamps (Optional)**
```
[0.00 - 10.00] Oh today's episode is interesting I got scammed
[10.00 - 15.00] More about that in just a bit
```

**With Markers (For Challenging Sections)**
```
Oh today's episode is interesting I got [unclear: scammed/scared]
More about that in just a bit [background noise]
What's up [unintelligible] how you guys doing
```

## Efficient Transcription Workflow

### Tools You Can Use

**1. Manual Transcription (Most Accurate)**
- Use text editor + audio player
- Keyboard shortcuts: Space = play/pause, Ctrl+Left/Right = rewind/forward

**2. Speech-to-Text Assistant (Faster, Needs Verification)**
```bash
# Use Whisper itself to create draft (then fix mistakes)
whisper data/"mamak session scam.mp3" --model medium --language en --output_format txt

# Then manually correct the output
```

**3. Online Tools**
- YouTube auto-captions (upload audio, download captions, fix errors)
- Otter.ai / Descript (free tiers available)
- **IMPORTANT:** Always verify and correct automated transcripts!

### Time-Saving Tips

**For 15-minute audio:**
- Full manual transcription: 3-5 hours
- Draft + correction: 1-2 hours
- Sample sections only: 30-60 minutes

**Recommendation:** Start with 2-3 minute samples to validate the approach, then decide if full transcription is needed.

## Sample Sections to Transcribe

For `mamak session scam.mp3` (15 minutes), good representative samples:

**Section 1: Clear Introduction (0:00 - 2:00)**
- Clean audio, single speaker
- Tests baseline accuracy

**Section 2: Phone Call (2:00 - 8:00)**
- Mixed speakers, lower quality
- Tests handling of challenging audio

**Section 3: Conclusion (13:00 - 15:00)**
- Clear audio, single speaker
- Tests consistency

## Evaluation Workflow

### Step 1: Create Ground Truth
```bash
# Transcribe to plain text file
nano data/ground_truth_mamak.txt
# Or use your preferred editor
```

### Step 2: Run Evaluation
```bash
python scripts/evaluate_with_ground_truth.py \
    --reference data/ground_truth_mamak.txt \
    --hypothesis outputs/faster_whisper_final/FW5_combined_best.txt \
    --output results/fw5_evaluation.json
```

### Step 3: Analyze Results
```
Output shows:
- WER (Word Error Rate): % of words incorrect
- CER (Character Error Rate): % of characters incorrect
- Error breakdown: substitutions, deletions, insertions
```

### Step 4: Diagnose Issues

**High WER (>20%):** Check for:
- Systematic errors (model always gets certain words wrong)
- Audio quality issues
- Language detection problems
- Model size too small

**High Substitutions:** Model recognizes speech but gets words wrong
- Solution: Larger model, better language model, fine-tuning

**High Deletions:** Model missing parts of speech
- Solution: Adjust VAD settings, check audio quality

**High Insertions:** Model adding extra words (hallucinations)
- Solution: Stricter beam search, VAD filtering

## Example Ground Truth Template

```text
# Ground Truth Transcript
# Audio: mamak session scam.mp3
# Duration: 15:22
# Transcriber: [Your Name]
# Date: [Date]
# Notes: Some background noise in phone call section

Oh today's episode is interesting I got scammed more about that in just a bit and you'll hear it here on today's show

What's up everybody how you guys doing I hope you guys are home safe but the looks of it numbers aren't going down

[Continue transcribing...]

# End of transcript
# Sections with heavy background noise: 5:30-6:45
# Unclear phrases marked with [?]
```

## Quality Checklist

Before running evaluation, verify your ground truth:

- [ ] Listened to entire audio (or selected sections)
- [ ] Transcribed exactly what was said
- [ ] Included natural speech patterns (ums, ahs, repetitions)
- [ ] Marked unclear sections with [unintelligible]
- [ ] Saved as plain text UTF-8 format
- [ ] Double-checked first and last sentences
- [ ] Spot-checked 3-4 random sections

## Common Mistakes to Avoid

### ❌ Wrong: Matching Model Output
```
Model says: "I got scared"
You hear: "I got scammed"
Ground truth: "I got scared" ← WRONG! Don't copy the model!
```

### ✅ Right: Transcribe What You Actually Hear
```
Model says: "I got scared"
You hear: "I got scammed"
Ground truth: "I got scammed" ← Correct!
```

### ❌ Wrong: Over-Punctuating
```
Oh, today's episode is interesting. I got scammed. More about that, in just a bit, and you'll hear it here, on today's show.
```

### ✅ Right: Minimal Punctuation
```
Oh today's episode is interesting I got scammed more about that in just a bit and you'll hear it here on today's show
```

## Expected Results

Based on our testing, **FW5 (optimal faster-whisper)** should achieve:

- **WER: 5-15%** (Good to Excellent)
- **CER: 2-8%** (Very Good to Excellent)

If results are significantly worse:
- Check ground truth quality (most common issue)
- Verify audio file is correct
- Check if model output file is correct version

## Next Steps After Evaluation

### If WER < 10% (Excellent)
- Document as baseline quality
- Test on more audio files
- Consider this configuration production-ready

### If WER 10-20% (Good)
- Acceptable for most use cases
- Consider larger model if better accuracy needed
- Document known error patterns

### If WER > 20% (Needs Improvement)
- Analyze error patterns
- Test with larger model (medium/large)
- Check audio quality
- Consider fine-tuning

## Resources

- **ASR Metrics Script:** `scripts/evaluate_with_ground_truth.py`
- **ASR Metrics Module:** `src/asr_metrics.py`
- **Example Ground Truth:** `data/ground_truth_mamak.txt` (you create this)
- **Model Output:** `outputs/faster_whisper_final/FW5_combined_best.txt`

## Questions?

Common questions:

**Q: How much should I transcribe?**
A: For validation: 2-3 minutes. For full evaluation: entire audio.

**Q: What if I can't understand a word?**
A: Mark it as [unintelligible] and note the timestamp.

**Q: Should I include "um", "uh", "like"?**
A: Yes! Transcribe exactly what you hear.

**Q: Does punctuation matter?**
A: Use `--keep-punctuation` flag if you want to include it in WER calculation.

**Q: How accurate should ground truth be?**
A: 100% accurate. If unsure, listen again or mark as [unclear].

---

**Remember:** Ground truth quality directly affects evaluation accuracy. Take your time and be precise!
