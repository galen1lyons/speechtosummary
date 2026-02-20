"""
compare_whispers.py — Side-by-side: OpenAI Whisper vs Malaysian Whisper.

Runs the SAME audio file through both models, measures WER, CER, and RTF
using your existing asr_metrics module, and prints a clean comparison table.

This is a standalone script.  It does NOT modify any of your src/ files.
It imports from src/ only to reuse the metrics you already wrote.

Usage:
    python compare_whispers.py --audio "data/manglish ho lee fak.mp3"

    # or with a manual reference transcript for WER/CER:
    python compare_whispers.py --audio "data/manglish ho lee fak.mp3" --reference "provide the actual text here"

    # pick which Malaysian Whisper size to use:
    python compare_whispers.py --audio "data/manglish ho lee fak.mp3" --my-model mesolitica/malaysian-whisper-medium

Notes:
  - First run downloads the Malaysian Whisper model (~0.5 GB for base).
    Subsequent runs use the cached copy — no re-download.
  - Loads one model at a time and deletes it before loading the next.
    This keeps peak RAM under 4 GB even on a 7.6 GB machine.
  - If you do NOT supply --reference, WER/CER are skipped and only
    the raw transcripts + RTF are shown.  You can eyeball the output
    and manually type a reference transcript for the next run.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

# ── pull in your own metrics module ──
# Works whether you run from the project root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.asr_metrics import calculate_wer, calculate_cer, calculate_rtf


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────
def green(t):  return f"\033[92m{t}\033[0m"
def red(t):    return f"\033[91m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def bold(t):   return f"\033[1m{t}\033[0m"

def get_audio_duration(audio_path: str) -> float:
    """Get duration in seconds using librosa or soundfile, fallback to ffprobe."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_text_from_segments(segments: list) -> str:
    """Extract plain text from Whisper-style segments list."""
    return " ".join(seg.get("text", "").strip() for seg in segments).strip()


# ─────────────────────────────────────────────
# Model A — OpenAI Whisper (your current model)
# ─────────────────────────────────────────────
def run_openai_whisper(audio_path: str, model_size: str = "base") -> dict:
    """
    Run OpenAI Whisper.  Returns dict with keys:
        text, segments, processing_time
    """
    import whisper

    print(f"  Loading OpenAI Whisper ({model_size})…")
    model = whisper.load_model(model_size)

    print(f"  Transcribing…")
    t0 = time.time()
    result = model.transcribe(audio_path, task="transcribe")
    elapsed = time.time() - t0

    # free memory immediately
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", []),
        "processing_time": elapsed,
        "detected_language": result.get("language", "unknown"),
    }


# ─────────────────────────────────────────────
# Model B — Malaysian Whisper (mesolitica)
# ─────────────────────────────────────────────
def run_malaysian_whisper(audio_path: str, model_name: str = "mesolitica/malaysian-whisper-base") -> dict:
    """
    Run Malaysian Whisper via transformers.  Returns same dict shape as above.

    Malaysian Whisper uses the transformers pipeline, not the openai-whisper
    package.  The output format is slightly different — we normalise it to
    match the shape that the rest of the script expects.
    """
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    import numpy as np

    print(f"  Loading {model_name}…")
    processor = AutoProcessor.from_pretrained(model_name)
    model     = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.eval()

    # load audio as 16 kHz mono numpy array
    print(f"  Loading audio…")
    try:
        import librosa
        audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
    except ImportError:
        import soundfile as sf
        import resampy
        raw, sr = sf.read(audio_path)
        if len(raw.shape) > 1:
            raw = raw.mean(axis=1)          # mono
        audio_array = resampy.resample(raw, sr, 16000).astype("float32")

    print(f"  Transcribing…")
    t0 = time.time()

    inputs = processor([audio_array], return_tensors="pt", sampling_rate=16000)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            language="ms",              # force Malay
            return_timestamps=True,
        )

    # decode with timestamps → gives us <|startofprevious|> style tokens
    decoded = processor.tokenizer.decode(generated_ids[0], return_timestamps=True)
    elapsed = time.time() - t0

    # Also decode plain text (no timestamp tokens)
    plain_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    # free memory
    del model, processor, inputs, generated_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Parse timestamps into segments so our output format matches OpenAI Whisper.
    # Malaysian Whisper returns timestamps as <|0.00|> tokens inline.
    segments = _parse_my_whisper_timestamps(decoded, plain_text)

    return {
        "text": plain_text,
        "segments": segments,
        "processing_time": elapsed,
        "detected_language": "ms",      # we forced it
    }


def _parse_my_whisper_timestamps(decoded: str, plain_text: str) -> list:
    """
    Parse the timestamp-annotated string from Malaysian Whisper into
    a segments list matching OpenAI Whisper's format.

    Malaysian Whisper outputs something like:
        <|0.00|> Selamat pagi <|2.50|> semua orang <|5.00|>
    We turn that into:
        [{"start": 0.0, "end": 2.5, "text": "Selamat pagi"}, ...]
    """
    import re
    # find all timestamp markers and the text between them
    parts = re.split(r"<\|([0-9.]+)\|>", decoded)
    # parts alternates: [text, timestamp, text, timestamp, ...]
    # first element is usually empty or whitespace

    timestamps = []
    texts      = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # this is a timestamp
            timestamps.append(float(part))
        else:
            texts.append(part.strip())

    segments = []
    # texts[0] is before the first timestamp (usually empty)
    # texts[1] is between timestamp[0] and timestamp[1], etc.
    for i in range(len(timestamps) - 1):
        seg_text = texts[i + 1] if (i + 1) < len(texts) else ""
        if seg_text:
            segments.append({
                "start": timestamps[i],
                "end":   timestamps[i + 1],
                "text":  seg_text,
            })

    # if parsing produced nothing, fall back to one big segment
    if not segments and plain_text:
        segments = [{"start": 0.0, "end": 0.0, "text": plain_text}]

    return segments


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compare OpenAI Whisper vs Malaysian Whisper on the same audio"
    )
    parser.add_argument("--audio", required=True,
                        help="Path to audio file (e.g. data/manglish ho lee fak.mp3)")
    parser.add_argument("--reference", default=None,
                        help="Ground-truth transcript text (for WER/CER). "
                             "If omitted, only raw output + RTF are shown.")
    parser.add_argument("--openai-model", default="base",
                        help="OpenAI Whisper size: tiny, base, small, medium, large")
    parser.add_argument("--my-model", default="mesolitica/malaysian-whisper-base",
                        help="Malaysian Whisper model ID on HuggingFace")
    parser.add_argument("--save", default=None,
                        help="If provided, save the full comparison JSON to this path")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(red(f"\n❌  File not found: {audio_path}"))
        sys.exit(1)

    audio_duration = get_audio_duration(str(audio_path))

    # ── header ──
    print(bold(f"\n{'═' * 64}"))
    print(bold(f"  Whisper Comparison"))
    print(bold(f"  Audio : {audio_path.name}"))
    print(bold(f"  Length: {audio_duration:.1f}s"))
    print(bold(f"{'═' * 64}\n"))

    # ── run Model A ──
    print(bold("🔷  OpenAI Whisper  (base)\n"))
    openai_result = run_openai_whisper(str(audio_path), args.openai_model)
    print(f"  Done in {openai_result['processing_time']:.2f}s\n")

    # ── run Model B ──
    print(bold(f"🟢  Malaysian Whisper  ({args.my_model})\n"))
    my_result = run_malaysian_whisper(str(audio_path), args.my_model)
    print(f"  Done in {my_result['processing_time']:.2f}s\n")

    # ── compute RTF for both ──
    openai_rtf = calculate_rtf(audio_duration, openai_result["processing_time"]) if audio_duration > 0 else None
    my_rtf     = calculate_rtf(audio_duration, my_result["processing_time"])     if audio_duration > 0 else None

    # ── compute WER / CER if reference provided ──
    openai_wer = openai_cer = my_wer = my_cer = None
    if args.reference:
        ref = args.reference
        try:
            openai_wer, *_ = calculate_wer(ref, openai_result["text"])
            openai_cer, *_ = calculate_cer(ref, openai_result["text"])
        except Exception as e:
            print(yellow(f"  ⚠️  WER/CER failed for OpenAI Whisper: {e}"))
        try:
            my_wer, *_ = calculate_wer(ref, my_result["text"])
            my_cer, *_ = calculate_cer(ref, my_result["text"])
        except Exception as e:
            print(yellow(f"  ⚠️  WER/CER failed for Malaysian Whisper: {e}"))

    # ─────────────────────────────────────────
    # PRINT REPORT
    # ─────────────────────────────────────────
    W = 60  # column width

    print(bold(f"\n{'─' * W}"))
    print(bold(f"{'METRIC':<25} {'OpenAI Whisper':>16} {'MY Whisper':>16}"))
    print(bold(f"{'─' * W}"))

    # RTF
    if openai_rtf and my_rtf:
        # lower RTF is better
        best_rtf = "openai" if openai_rtf < my_rtf else "my"
        o_tag = green(f"{openai_rtf:.2f}x") if best_rtf == "openai" else f"{openai_rtf:.2f}x"
        m_tag = green(f"{my_rtf:.2f}x")     if best_rtf == "my"     else f"{my_rtf:.2f}x"
        print(f"  {'RTF (lower=faster)':<23} {o_tag:>16} {m_tag:>16}")

    # Processing time
    print(f"  {'Time (seconds)':<23} {openai_result['processing_time']:>14.2f}s {my_result['processing_time']:>14.2f}s")

    # WER
    if openai_wer is not None and my_wer is not None:
        best_wer = "openai" if openai_wer < my_wer else "my"
        o_tag = green(f"{openai_wer:.2%}") if best_wer == "openai" else f"{openai_wer:.2%}"
        m_tag = green(f"{my_wer:.2%}")     if best_wer == "my"     else f"{my_wer:.2%}"
        print(f"  {'WER (lower=better)':<23} {o_tag:>16} {m_tag:>16}")
    elif args.reference is None:
        print(f"  {'WER':<23} {'— no ref':>16} {'— no ref':>16}")

    # CER
    if openai_cer is not None and my_cer is not None:
        best_cer = "openai" if openai_cer < my_cer else "my"
        o_tag = green(f"{openai_cer:.2%}") if best_cer == "openai" else f"{openai_cer:.2%}"
        m_tag = green(f"{my_cer:.2%}")     if best_cer == "my"     else f"{my_cer:.2%}"
        print(f"  {'CER (lower=better)':<23} {o_tag:>16} {m_tag:>16}")
    elif args.reference is None:
        print(f"  {'CER':<23} {'— no ref':>16} {'— no ref':>16}")

    # Detected language
    print(f"  {'Detected language':<23} {openai_result['detected_language']:>16} {my_result['detected_language']:>16}")

    # Segment count
    print(f"  {'Segments':<23} {len(openai_result['segments']):>16} {len(my_result['segments']):>16}")

    print(bold(f"{'─' * W}\n"))

    # ── full transcripts ──
    print(bold("📄  OpenAI Whisper — full transcript:\n"))
    print(f"  {openai_result['text']}\n")

    print(bold(f"📄  Malaysian Whisper — full transcript:\n"))
    print(f"  {my_result['text']}\n")

    # ── timestamped segments ──
    print(bold("⏱️   OpenAI Whisper — segments:\n"))
    for seg in openai_result["segments"]:
        print(f"  [{seg['start']:>6.2f} – {seg['end']:>6.2f}]  {seg['text']}")

    print(bold(f"\n⏱️   Malaysian Whisper — segments:\n"))
    for seg in my_result["segments"]:
        print(f"  [{seg['start']:>6.2f} – {seg['end']:>6.2f}]  {seg['text']}")

    # ── hint if no reference ──
    if args.reference is None:
        print(bold(yellow(
            "\n💡  No reference transcript provided.  To get WER/CER, re-run with:\n"
            "      --reference \"paste the correct transcript here\"\n"
            "    You can read the audio and type it out, or use a known transcript.\n"
        )))

    # ── save JSON if requested ──
    if args.save:
        out = {
            "audio": str(audio_path),
            "audio_duration_s": audio_duration,
            "reference": args.reference,
            "openai_whisper": {
                "model": args.openai_model,
                "text": openai_result["text"],
                "segments": openai_result["segments"],
                "processing_time_s": openai_result["processing_time"],
                "rtf": openai_rtf,
                "wer": openai_wer,
                "cer": openai_cer,
                "detected_language": openai_result["detected_language"],
            },
            "malaysian_whisper": {
                "model": args.my_model,
                "text": my_result["text"],
                "segments": my_result["segments"],
                "processing_time_s": my_result["processing_time"],
                "rtf": my_rtf,
                "wer": my_wer,
                "cer": my_cer,
                "detected_language": my_result["detected_language"],
            },
        }
        Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved full comparison to {args.save}")

    print(bold(f"{'═' * 64}\n"))


if __name__ == "__main__":
    main()