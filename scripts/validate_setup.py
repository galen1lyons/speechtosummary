"""
Validate Setup for Comprehensive Whisper Testing

This script checks if your environment is properly configured before running tests.

Usage:
    python scripts/validate_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking required imports...")

    errors = []

    # Check src imports
    try:
        from src import WhisperConfig, transcribe, setup_logger
        print("  ✅ src module imports successful")
    except ImportError as e:
        errors.append(f"  ❌ Failed to import from src: {e}")

    # Check whisper
    try:
        import whisper
        print("  ✅ whisper module found")
    except ImportError:
        errors.append("  ❌ whisper not installed (pip install openai-whisper)")

    # Check transformers
    try:
        import transformers
        print("  ✅ transformers module found")
    except ImportError:
        errors.append("  ❌ transformers not installed (pip install transformers)")

    # Check torch
    try:
        import torch
        print(f"  ✅ torch module found (version {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available ({torch.cuda.get_device_name(0)})")
        else:
            print("  ⚠️  CUDA not available (will use CPU)")
    except ImportError:
        errors.append("  ❌ torch not installed (pip install torch)")

    return errors

def check_audio_file():
    """Check if test audio file exists."""
    print("\nChecking test audio file...")

    audio_path = project_root / "data" / "mamak session scam.mp3"

    if audio_path.exists():
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Audio file found: {audio_path.name} ({size_mb:.1f} MB)")
        return []
    else:
        return [f"  ❌ Audio file not found: {audio_path}"]

def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")

    errors = []

    data_dir = project_root / "data"
    if data_dir.exists():
        print(f"  ✅ data/ directory exists")
    else:
        errors.append("  ❌ data/ directory not found")

    src_dir = project_root / "src"
    if src_dir.exists():
        print(f"  ✅ src/ directory exists")
    else:
        errors.append("  ❌ src/ directory not found")

    scripts_dir = project_root / "scripts"
    if scripts_dir.exists():
        print(f"  ✅ scripts/ directory exists")
    else:
        errors.append("  ❌ scripts/ directory not found")

    # Create output directories if they don't exist
    (project_root / "outputs" / "runs").mkdir(parents=True, exist_ok=True)

    print(f"  ✅ Output directories ready")

    return errors

def check_scripts():
    """Check if core scripts exist."""
    print("\nChecking core scripts...")

    errors = []

    scripts = [
        "transcribe.py",
        "check_project.py",
    ]

    for script in scripts:
        script_path = project_root / "scripts" / script
        if script_path.exists():
            print(f"  ✅ {script}")
        else:
            errors.append(f"  ❌ {script} not found")

    return errors

def main():
    """Run all validation checks."""
    print("="*70)
    print("  COMPREHENSIVE WHISPER TESTING - SETUP VALIDATION")
    print("="*70)
    print(f"\nProject root: {project_root}")
    print(f"Python version: {sys.version.split()[0]}")

    all_errors = []

    # Run checks
    all_errors.extend(check_directories())
    all_errors.extend(check_scripts())
    all_errors.extend(check_imports())
    all_errors.extend(check_audio_file())

    # Print summary
    print("\n" + "="*70)
    print("  VALIDATION SUMMARY")
    print("="*70)

    if all_errors:
        print("\n❌ Setup validation FAILED\n")
        print("Errors found:")
        for error in all_errors:
            print(error)
        print("\nPlease fix these issues before running tests.")
        print("\nQuick fixes:")
        print("  1. Activate virtual environment: source venv/bin/activate")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Verify audio file exists: ls -lh 'data/mamak session scam.mp3'")
        return 1
    else:
        print("\n✅ All checks passed! Your setup is ready.\n")
        print("You can now run:")
        print("  python scripts/transcribe.py transcribe --audio \"data/mamak session scam.mp3\" --config fw5_optimal")
        return 0

if __name__ == "__main__":
    exit(main())
