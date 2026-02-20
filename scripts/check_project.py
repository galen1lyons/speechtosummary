"""
check_project.py — Diagnose your speechtosummary project.

Run this FIRST. It checks everything and tells you exactly what is
healthy, what is missing, and what needs fixing before you touch any code.

Usage:
    python check_project.py
"""

import sys
import os
import subprocess
from pathlib import Path

# ── colour helpers (works on modern Windows terminals too) ──
def green(t):  return f"\033[92m{t}\033[0m"
def red(t):    return f"\033[91m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def bold(t):   return f"\033[1m{t}\033[0m"

PASS = green("✅ PASS")
FAIL = red("❌ FAIL")
WARN = yellow("⚠️  WARN")

issues = []   # collected so we can print a fix-list at the end

def report(label, ok, detail=""):
    tag = PASS if ok else FAIL
    line = f"  {tag}  {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    if not ok:
        issues.append((label, detail))

# ─────────────────────────────────────────────
# 1.  ROOT DIRECTORY DETECTION
# ─────────────────────────────────────────────
print(bold("\n📁  1. Root directory"))
# We expect to be run FROM inside speechtosummary/
ROOT = Path.cwd()
print(f"  Current directory: {ROOT}")

has_src   = (ROOT / "src").is_dir()
has_data  = (ROOT / "data").is_dir()
has_req   = (ROOT / "requirements.txt").is_file()
report("src/ folder exists",           has_src)
report("data/ folder exists",          has_data)
report("requirements.txt exists",      has_req)

if not has_src:
    print(red("\n  !! You are probably not in the speechtosummary/ folder."))
    print(f"     Try:  cd speechtosummary")
    sys.exit(1)

# ─────────────────────────────────────────────
# 2.  PYTHON & VIRTUAL ENV
# ─────────────────────────────────────────────
print(bold("\n🐍  2. Python & virtual environment"))
report("Python version",
       sys.version_info >= (3, 8),
       f"running {sys.version.split()[0]}")

in_venv = (sys.prefix != sys.base_prefix) or ("VIRTUAL_ENV" in os.environ)
report("Virtual environment is active", in_venv,
       os.environ.get("VIRTUAL_ENV", sys.prefix))

# ─────────────────────────────────────────────
# 3.  REQUIRED PACKAGES (import-check)
# ─────────────────────────────────────────────
print(bold("\n📦  3. Required packages"))

REQUIRED = {
    "whisper":       "openai-whisper  (Whisper transcription)",
    "torch":         "torch / PyTorch",
    "transformers":  "transformers    (HF summarization)",
    "dotenv":        "python-dotenv   (env file loading)",
}

# diarization is optional — only warn if .env has a token
for mod, label in REQUIRED.items():
    try:
        __import__(mod)
        report(f"import {mod}", True, label)
    except ImportError:
        report(f"import {mod}", False, f"pip install {mod.replace('_','-')}")

# Check pyannote separately (optional but important)
print(bold("\n  Speaker diarization (optional)"))
try:
    from pyannote.audio import Pipeline          # noqa: F401
    report("import pyannote.audio", True)
except (ImportError, AttributeError) as e:
    if "set_audio_backend" in str(e):
        print(f"  {WARN}  pyannote.audio has compatibility issue with torchaudio")
        print(f"          (torchaudio.set_audio_backend deprecated)")
    else:
        print(f"  {WARN}  pyannote.audio not installed — diarization won't work")
        print(f"          install later with:  pip install pyannote.audio")

# ─────────────────────────────────────────────
# 4.  .env / HF TOKEN
# ─────────────────────────────────────────────
print(bold("\n🔐  4. HuggingFace token (.env)"))
env_path = ROOT / ".env"
report(".env file exists", env_path.is_file())

hf_token = None
if env_path.is_file():
    # read it ourselves so we don't need dotenv to be installed yet
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN="):
            hf_token = line.split("=", 1)[1].strip().strip("'\"")
            break
    report("HF_TOKEN found in .env",
           bool(hf_token),
           f"{hf_token[:10]}…" if hf_token else "missing")
else:
    issues.append(("HF_TOKEN", "create .env with HF_TOKEN=hf_…"))

# ─────────────────────────────────────────────
# 5.  src/ FILES — what actually exists?
# ─────────────────────────────────────────────
print(bold("\n🗂️   5. Source files in src/"))

CORE_FILES = [
    "__init__.py",
    "config.py",
    "exceptions.py",
    "logger.py",
    "utils.py",
    "transcribe.py",
    "summarize.py",
    "pipeline.py",
]

OPTIONAL_FILES = [
    "diarize.py",
    "asr_metrics.py",
    "transcribe_v2.py",   # dead file — flag it
]

for fname in CORE_FILES:
    report(f"src/{fname}", (ROOT / "src" / fname).is_file())

print(bold("  Optional / extra files:"))
for fname in OPTIONAL_FILES:
    exists = (ROOT / "src" / fname).is_file()
    if fname == "transcribe_v2.py" and exists:
        print(f"  {WARN}  src/{fname}  — exists but is NOT used by anything (safe to delete)")
    else:
        tag = PASS if exists else yellow("⏭️  not present")
        print(f"  {tag}  src/{fname}")

# ─────────────────────────────────────────────
# 6.  IMPORT SMOKE-TEST — actually load every module
# ─────────────────────────────────────────────
print(bold("\n🔧  6. Module import smoke-test"))

# We need src/ on the path so relative imports resolve
sys.path.insert(0, str(ROOT))

MODULES_TO_TEST = [
    "src.exceptions",
    "src.logger",
    "src.utils",
    "src.config",
    "src.transcribe",
    "src.summarize",
    "src.pipeline",
]

# add optional modules only if the file is present
if (ROOT / "src" / "diarize.py").is_file():
    MODULES_TO_TEST.append("src.diarize")
if (ROOT / "src" / "asr_metrics.py").is_file():
    MODULES_TO_TEST.append("src.asr_metrics")

for mod in MODULES_TO_TEST:
    try:
        __import__(mod)
        report(f"import {mod}", True)
    except Exception as e:
        report(f"import {mod}", False, str(e))

# ─────────────────────────────────────────────
# 7.  DATA FILES
# ─────────────────────────────────────────────
print(bold("\n🎵  7. Audio files in data/"))
audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
audio_files = [f for f in (ROOT / "data").iterdir()
               if f.is_file() and f.suffix.lower() in audio_exts] if has_data else []

if audio_files:
    for af in sorted(audio_files):
        size_mb = af.stat().st_size / (1024 * 1024)
        print(f"  {green('🎤')}  {af.name}  ({size_mb:.1f} MB)")
else:
    print(f"  {WARN}  No audio files in data/  — add .mp3 or .wav files to test")
    issues.append(("Audio files", "add at least one .mp3 or .wav to data/"))

# ─────────────────────────────────────────────
# 8.  FFMPEG
# ─────────────────────────────────────────────
print(bold("\n🛠️   8. FFmpeg"))
try:
    subprocess.run(["ffmpeg", "-version"],
                   capture_output=True, check=True)
    report("ffmpeg is installed", True)
except (subprocess.CalledProcessError, FileNotFoundError):
    report("ffmpeg is installed", False, "sudo apt-get install ffmpeg")

# ─────────────────────────────────────────────
# 9.  OUTPUTS FOLDER
# ─────────────────────────────────────────────
print(bold("\n📂  9. outputs/"))
out_dir = ROOT / "outputs"
if out_dir.is_dir():
    out_files = list(out_dir.rglob("*"))
    out_files = [f for f in out_files if f.is_file()]
    print(f"  {PASS}  outputs/ exists  — {len(out_files)} file(s) inside")
else:
    print(f"  {WARN}  outputs/ does not exist yet — will be created on first run")

# ─────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
if not issues:
    print(bold(green("\n🎉  Everything looks good! You're ready to run the pipeline.\n")))
    print("    Quick test:")
    if audio_files:
        print(f'      python -m src.pipeline --audio "data/{audio_files[0].name}" --whisper-model tiny')
    else:
        print('      python -m src.pipeline --audio data/your_file.mp3 --whisper-model tiny')
else:
    print(bold(yellow(f"\n⚠️  {len(issues)} issue(s) to fix:\n")))
    for i, (label, detail) in enumerate(issues, 1):
        print(f"    {i}. {label}")
        if detail:
            print(f"       → {detail}")
    print()

print("─" * 60)
