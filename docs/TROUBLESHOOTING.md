# Troubleshooting

### "No module named 'whisper'" / "No module named 'src'"
```bash
source venv/bin/activate && pip install -r requirements.txt
```

### "ffmpeg not found"
```bash
sudo apt-get install ffmpeg    # Ubuntu/WSL
brew install ffmpeg            # macOS
```

### "CUDA out of memory"
```bash
python -m src.pipeline --audio meeting.mp3 --device cpu --whisper-model base
```

### "HF_TOKEN not found" (diarization)
```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```
Get token: https://huggingface.co/settings/tokens. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1

### Poor transcription quality
1. Use a larger model: `--whisper-model small` (best) or `medium`
2. Specify language: `--language en` (avoids auto-detect errors)
3. Add context: `--initial-prompt "Technical meeting about robotics"`

### Hallucinations / repetitive text
The default faster-whisper config (beam_size=7, VAD threshold=0.7) eliminates ~99% of hallucinations. If you still see issues:
- Do NOT set `--temperature 0.0` — this causes catastrophic looping in faster-whisper
- Avoid `--hotwords` on small models (causes token looping)
- Try a larger model

### Pipeline too slow
- Use `--whisper-model base` or `tiny` for drafts
- Specify `--language en` to skip auto-detection
- Diarization adds ~17 min on CPU — disable if not needed

### WSL: "python: command not found"
Use `python3` instead of `python`.

### Nuclear option: clean reinstall
```bash
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
