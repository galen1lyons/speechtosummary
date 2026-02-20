"""
probe_malaysian_models.py — Can your machine actually run these models?

This script does NOT run any inference.  It checks:
  1. How much RAM / GPU memory you have
  2. How much free disk space you have
  3. Whether each target model can be downloaded (connectivity + token)
  4. Whether each model can be loaded into memory

Run it once.  It tells you exactly what's feasible and what isn't.

Usage:
    python probe_malaysian_models.py
"""

import sys
import os
import shutil
import time
from pathlib import Path

# ── colour helpers ──
def green(t):  return f"\033[92m{t}\033[0m"
def red(t):    return f"\033[91m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def bold(t):   return f"\033[1m{t}\033[0m"

# ─────────────────────────────────────────────
# 1.  HARDWARE INVENTORY
# ─────────────────────────────────────────────
print(bold("\n🖥️   1. Hardware inventory\n"))

import psutil
ram_total_gb = psutil.virtual_memory().total / (1024**3)
ram_free_gb  = psutil.virtual_memory().available / (1024**3)
print(f"  RAM total : {ram_total_gb:.1f} GB")
print(f"  RAM free  : {ram_free_gb:.1f} GB")

gpu_available = False
gpu_name      = "none"
gpu_mem_gb    = 0.0
try:
    import torch
    if torch.cuda.is_available():
        gpu_available = True
        gpu_name      = torch.cuda.get_device_name(0)
        gpu_mem_gb    = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  GPU       : {gpu_name}")
        print(f"  GPU VRAM  : {gpu_mem_gb:.1f} GB")
    else:
        print(f"  GPU       : {yellow('none — will run on CPU')}")
except ImportError:
    print(f"  {red('torch not installed — cannot check GPU')}")
    sys.exit(1)

# disk space in the HF cache directory (default)
hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
disk_total, disk_used, disk_free = shutil.disk_usage(str(hf_cache.parent))
disk_free_gb = disk_free / (1024**3)
print(f"  Disk free : {disk_free_gb:.1f} GB  (HF cache lives in {hf_cache})")

# ─────────────────────────────────────────────
# 2.  FEASIBILITY TABLE
# ─────────────────────────────────────────────
# Each entry: (label, hf_id, download_gb, ram_gb_cpu, vram_gb_gpu, purpose)
MODELS = [
    ("Malaysian Whisper base",
     "mesolitica/malaysian-whisper-base",
     0.5,   # ~500 MB download
     2.0,   # needs ~2 GB RAM on CPU
     0.5,   # needs ~0.5 GB VRAM on GPU
     "Transcription — drop-in alongside OpenAI Whisper"),

    ("Malaysian Whisper medium",
     "mesolitica/malaysian-whisper-medium",
     1.5,   # ~1.5 GB download
     4.0,   # needs ~4 GB RAM on CPU
     1.5,   # needs ~1.5 GB VRAM on GPU
     "Transcription — better accuracy, slower"),

    ("MaLLaM 1.1B instructions v2",
     "mesolitica/mallam-1.1b-20k-instructions-v2",
     2.2,   # ~2.2 GB download
     6.0,   # needs ~6 GB RAM on CPU (fp16)
     2.5,   # needs ~2.5 GB VRAM on GPU (fp16)
     "Summarisation — Malaysian-aware LLM"),

    ("MaLLaM 3B instructions",
     "mesolitica/mallam-3b-20k-instructions",
     6.0,   # ~6 GB download
     14.0,  # needs ~14 GB RAM on CPU
     6.5,   # needs ~6.5 GB VRAM on GPU
     "Summarisation — better quality, needs more RAM"),
]

print(bold("\n📊  2. Feasibility check\n"))
print(f"  {'Model':<38} {'Download':>9} {'Fits RAM?':>10} {'Fits VRAM?':>11}  Purpose")
print(f"  {'─'*38} {'─'*9} {'─'*10} {'─'*11}  {'─'*50}")

feasible = []
for label, hf_id, dl_gb, ram_gb, vram_gb, purpose in MODELS:
    dl_ok   = disk_free_gb > dl_gb
    ram_ok  = ram_free_gb  > ram_gb
    vram_ok = (not gpu_available) or (gpu_mem_gb > vram_gb)

    dl_tag   = green(f"{dl_gb:.1f} GB") if dl_ok   else red(f"{dl_gb:.1f} GB")
    ram_tag  = green("yes")            if ram_ok  else red("no")
    vram_tag = green("yes")            if vram_ok else red("no")
    if not gpu_available:
        vram_tag = yellow("CPU only")

    print(f"  {label:<38} {dl_tag:>18} {ram_tag:>10} {vram_tag:>11}  {purpose}")

    if dl_ok and ram_ok:
        feasible.append((label, hf_id, dl_gb, ram_gb, vram_gb, purpose))

# ─────────────────────────────────────────────
# 3.  CONNECTIVITY CHECK — can we actually reach HuggingFace?
# ─────────────────────────────────────────────
print(bold("\n🌐  3. HuggingFace connectivity\n"))

try:
    from huggingface_hub import HfApi
    api = HfApi()
    # just list the files for the smallest model — proves auth + network work
    files = api.list_repo_files("mesolitica/malaysian-whisper-base")
    print(f"  {green('✅')}  HuggingFace reachable — {len(files)} files in mesolitica/malaysian-whisper-base")
except Exception as e:
    print(f"  {red('❌')}  Cannot reach HuggingFace: {e}")
    print(f"       Check your network and that HF_TOKEN is set in .env")
    sys.exit(1)

# ─────────────────────────────────────────────
# 4.  TRIAL DOWNLOAD — grab just the config files
#     (tiny, proves the token + repo are accessible)
# ─────────────────────────────────────────────
print(bold("\n⬇️   4. Config-file download check (proves access, downloads < 10 KB each)\n"))

from huggingface_hub import hf_hub_download

for label, hf_id, *_ in feasible[:2]:   # only check the first two feasible ones
    try:
        t0 = time.time()
        path = hf_hub_download(repo_id=hf_id, filename="config.json")
        elapsed = time.time() - t0
        print(f"  {green('✅')}  {label}")
        print(f"       config.json downloaded in {elapsed:.2f}s → {path}")
    except Exception as e:
        print(f"  {red('❌')}  {label}: {e}")

# ─────────────────────────────────────────────
# 5.  RECOMMENDATION
# ─────────────────────────────────────────────
print(bold("\n📋  5. Recommended next steps\n"))

if not feasible:
    print(f"  {red('None of the models fit your current free RAM.')}")
    print(f"  Close other applications or free up disk space and try again.")
else:
    print(f"  These models are feasible on your machine:\n")
    for label, hf_id, dl_gb, *_, purpose in feasible:
        print(f"    • {bold(label)}")
        print(f"      {hf_id}")
        print(f"      {purpose}")
        print(f"      Download size: {dl_gb} GB")
        print()

    # pick the best two: one whisper, one mallam
    whisper_pick = None
    mallam_pick  = None
    for item in feasible:
        if "Whisper" in item[0] and whisper_pick is None:
            whisper_pick = item
        if "MaLLaM" in item[0] and mallam_pick is None:
            mallam_pick = item

    print("  " + bold("Suggested starting pair:"))
    if whisper_pick:
        print(f"    Transcription : {whisper_pick[1]}")
    if mallam_pick:
        print(f"    Summarisation : {mallam_pick[1]}")
    print()
    print("  To actually download and test one of them, come back here")
    print("  and I will write the integration code against your real src/ files.")

print("\n" + "─" * 60)