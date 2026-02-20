# Troubleshooting

## Quick Fixes

### "No module named 'whisper'" or "No module named 'src'"

**Fix:**
```bash
pip install -r requirements.txt
```

---

### "ffmpeg not found" or "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"

**Fix (Ubuntu/WSL):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Fix (macOS):**
```bash
brew install ffmpeg
```

**Verify:**
```bash
ffmpeg -version
```

---

### "CUDA out of memory" or "RuntimeError: CUDA out of memory"

**Fix - Use CPU:**
```bash
python -m src.pipeline --audio meeting.mp3 --device cpu
```

**Or use smaller model:**
```bash
python -m src.pipeline --audio meeting.mp3 --whisper-model tiny
```

---

### Processing is too slow

**Fix 1 - Use smaller model:**
```bash
python -m src.pipeline --audio meeting.mp3 --whisper-model tiny
```

**Fix 2 - Use GPU (if available):**
```bash
python -m src.pipeline --audio meeting.mp3 --device cuda
```

---

### "HF_TOKEN not found" (when using diarization)

**Fix - Create .env file:**
```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

Get token from: https://huggingface.co/settings/tokens

---

### "FileNotFoundError: [Errno 2] No such file or directory: 'data/meeting.mp3'"

**Fix - Use quotes for filenames with spaces:**
```bash
python -m src.pipeline --audio "data/my meeting.mp3"
```

**Or check the file exists:**
```bash
ls data/
```

---

### WSL: "python: command not found"

**Fix - Use python3:**
```bash
python3 -m src.pipeline --audio meeting.mp3
```

---

### "ModuleNotFoundError: No module named 'transformers'"

**Fix - Reinstall all dependencies:**
```bash
pip install -r requirements.txt --force-reinstall
```

---

### Transcription quality is poor

**Fix 1 - Use better model:**
```bash
python -m src.pipeline --audio meeting.mp3 --whisper-model medium
```

**Fix 2 - Add context hint:**
```bash
python -m src.pipeline --audio meeting.mp3 --initial-prompt "Technical meeting about AI"
```

**Fix 3 - Specify language:**
```bash
python -m src.pipeline --audio meeting.mp3 --language en
```

---

### Speaker diarization not working

**Fix 1 - Check HF token is set:**
```bash
cat .env
# Should show: HF_TOKEN=hf_...
```

**Fix 2 - Accept model license:**
1. Go to: https://huggingface.co/pyannote/speaker-diarization-3.1
2. Click "Agree and access repository"
3. Try again

---

### "Permission denied" errors

**Fix - Check file permissions:**
```bash
chmod +x your_script.py
```

---

### Virtual environment issues (Python packages not found)

**Fix - Activate venv:**
```bash
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

---

## Still Not Working?

1. **Check Python version:**
   ```bash
   python --version  # Need 3.8+
   ```

2. **Check you're in the right directory:**
   ```bash
   pwd  # Should end with /speechtosummary
   ls   # Should show: src/, data/, outputs/, requirements.txt
   ```

3. **Clean install:**
   ```bash
   rm -rf venv/
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Check disk space:**
   ```bash
   df -h
   ```

5. **Check logs:**
   - Look for error messages in terminal output
   - Check `outputs/` for partial results