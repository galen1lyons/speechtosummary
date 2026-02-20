# Archive

One-time verification scripts that are no longer actively used but kept for reference.

---

## Scripts

### `verify_parameter_fix.py`

**Purpose:** One-time verification that Whisper parameters are correctly passed to models

**Usage:**
```bash
python scripts/archive/verify_parameter_fix.py
```

**What it does:**
- Runs 2 tests with different beam sizes (1 vs 10)
- Confirms that different parameters produce different outputs
- Validates parameter passing to HuggingFace models

**Status:** Verification complete - parameters confirmed working

---

### `verify_optimal_test_setup.sh`

**Purpose:** One-time setup verification for optimal whisper tests

**Usage:**
```bash
bash scripts/archive/verify_optimal_test_setup.sh
```

**What it does:**
- Checks virtual environment is activated
- Verifies audio file exists
- Validates disk space (>500MB)
- Confirms Python packages installed

**Status:** Replaced by `scripts/validate_setup.py` which is more comprehensive

---

### `create_stress_test.sh`

**Purpose:** Utility to create stress test audio files

**Usage:**
```bash
bash scripts/archive/create_stress_test.sh
```

**What it does:**
- Creates audio files for stress testing
- Purpose unclear - may have been for performance testing

**Status:** Unclear if still needed - kept for reference

---

## Why These Are Archived

These scripts were useful for one-time verification or specific development tasks but are not part of the active workflow:

- **verify_parameter_fix.py** - Verification complete, issue resolved
- **verify_optimal_test_setup.sh** - Replaced by better validation script
- **create_stress_test.sh** - Purpose unclear, not referenced in documentation

If you need to resurrect any of these scripts, they are still functional - just move them back to `scripts/` directory.

---

**Archived:** 2026-02-09
