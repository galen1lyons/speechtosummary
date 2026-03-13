# User Guide

## Basic Usage

```bash
python -m src.pipeline --audio <path-to-audio> [options]
```

## Model Selection

| Model | WER* | Speed (RTF) | RAM | Recommendation |
|-------|------|-------------|-----|----------------|
| `tiny` | — | fastest | 1GB | Quick drafts only |
| `base` | 34.67% | 0.16x | 1GB | Testing |
| **`small`** | **23.84%** | **0.38x** | **2GB** | **Best accuracy/speed tradeoff** |
| `medium` | 26.23% | 0.99x | 5GB | Worse than `small` on CPU int8 |

*WER measured on dedm meeting audio, full-audio mode. See `docs/MODEL_SIZE_EXPERIMENT.md`.

**Malaysian Whisper** (`mesolitica/malaysian-whisper-*`): Evaluated and found unviable (RTF 23.7x, 1027 hallucinations). Requires `--backend openai-whisper`.

## Key Options

### Transcription
```bash
--whisper-model small          # Model size (default: base)
--language en                  # Force language (default: auto)
--backend faster-whisper       # Backend (default, recommended)
--beam-size 7                  # Beam search size (default: 7)
--initial-prompt "context"     # Domain context for decoder
--hotwords "AMR,AGV,ROS"       # Boost words (faster-whisper only, use cautiously)
```

### Speaker Diarization
```bash
--enable-diarization           # Enable speaker identification
--min-speakers 2               # Constrain speaker count
--max-speakers 4
```

Note: Diarization fragments audio into short clips, which increases WER significantly (57% vs 35% on base). A pipeline refactor to transcribe-first-then-diarize is planned. See `PLAN.md`.

### Summarization
Summaries are generated automatically using Mistral 7B Instruct. Sections: Executive Summary, Key Discussion Points, Action Items, Decisions Made.

```bash
--content-type meeting         # meeting | interview | podcast | general
--summary-model-path /path.gguf  # Custom GGUF model (default: auto-download)
```

### Evaluation
```bash
--reference-transcript ref.txt   # Compute WER/CER against reference
--reference-rttm ref.rttm       # Compute DER/JER (requires --enable-diarization)
```

## Output Structure

Each run creates a timestamped folder:

```
outputs/runs/20260310T.../ or outputs/campaigns/eval/20260310T.../
├── transcript.txt            # Always
├── transcript.json           # Always
├── summary.md                # Always
├── manifest.json             # Always
├── preprocessed.wav          # Always
├── speakers.txt              # With --enable-diarization
├── speaker_stats.json        # With --enable-diarization
├── diarization.rttm          # With --enable-diarization
├── asr_metrics.json          # With --reference-transcript
└── diarization_metrics.json  # With --reference-rttm
```

Runs with `--reference-transcript` or `--reference-rttm` go to `outputs/campaigns/eval/`. Production runs go to `outputs/runs/`.

## Full CLI Reference

```bash
python -m src.pipeline --help
```
