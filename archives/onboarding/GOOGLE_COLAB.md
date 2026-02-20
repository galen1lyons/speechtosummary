# Google Colab GPU Transcription Notebook

Copy-paste these cells into your Google Colab notebook.

---

## Cell 1: Setup

```python
# Install Whisper
!pip install -q openai-whisper

import torch
import whisper
import time

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Cell 2: Upload Audio File

```python
from google.colab import files

print("Select an audio file to upload...")
uploaded = files.upload()
audio_file = list(uploaded.keys())[0]
print(f"✓ Uploaded: {audio_file}")
```

---

## Cell 3: Transcribe with Timing

```python
# Load model (uses GPU automatically)
print("Loading Whisper model...")
model = whisper.load_model("base")
print("✓ Model loaded")

# Transcribe with timing
print(f"\nTranscribing: {audio_file}")
start = time.time()
result = model.transcribe(audio_file, language="en", beam_size=5, temperature=0.0)
elapsed = time.time() - start

# Results
duration = result.get('duration', 0)
print(f"\n{'='*50}")
print(f"Audio duration: {duration/60:.1f} minutes")
print(f"Processing time: {elapsed/60:.1f} minutes")
print(f"Speed ratio: {elapsed/duration:.2f}x realtime")
print(f"{'='*50}")

# Preview
print(f"\nTranscript preview (first 500 chars):")
print(result['text'][:500])
```

---

## Cell 4: Test Different Parameters

```python
# Test with higher beam_size
print("Testing with beam_size=10...")
start = time.time()
result_10 = model.transcribe(audio_file, language="en", beam_size=10, temperature=0.0)
elapsed_10 = time.time() - start

print(f"beam_size=10: {elapsed_10/60:.1f} min ({elapsed_10/duration:.2f}x realtime)")
print(f"beam_size=5:  {elapsed/60:.1f} min ({elapsed/duration:.2f}x realtime)")
print(f"Difference: {(elapsed_10-elapsed):.1f} seconds slower")
```

---

## Cell 5: Test with Initial Prompt (for accents)

```python
# Test with initial prompt for Malaysian English
result_prompt = model.transcribe(
    audio_file, 
    language="en",
    beam_size=5,
    initial_prompt="This is a Malaysian English conversation with local accents and slang."
)

print("With initial_prompt:")
print(result_prompt['text'][:500])
```

---

## Expected Results

| Device | 13 min audio | Speed |
|--------|--------------|-------|
| CPU (laptop) | ~24 min | 1.8x realtime |
| GPU (Colab T4) | ~4 min | 0.3x realtime |

**Speedup: ~6x faster on GPU**
