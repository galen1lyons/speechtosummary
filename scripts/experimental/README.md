# Experimental Scripts

Experimental features and development tools not part of the main workflow.

---

## Scripts

### `summarize_mallam.py`

**Purpose:** Experimental Malaysian LLM summarization (alternative to mT5)

**Status:** Experimental - not integrated into main pipeline

**Usage (Standalone):**
```bash
python scripts/experimental/summarize_mallam.py \
  --transcript outputs/manglish_ho_lee_fak.json
```

**Usage (From Code):**
```python
from scripts.experimental.summarize_mallam import summarize_with_mallam
summary_md = summarize_with_mallam(transcript_json_path)
```

**What it does:**
- Uses MaLLaM (Malaysian LLM) for summarization
- Model: `mesolitica/mallam-1.1b-20k-instructions-v2`
- 1.1B parameters (~2.2GB RAM in float16)
- Generates summary in same format as existing summarizer

**Performance:**
- CPU generation: 30-120 seconds per summary
- Model loaded once and reused across calls
- Call `unload_model()` when done

**Why experimental:**
- Not benchmarked against mT5
- Slower than existing summarizer
- Requires additional 2GB RAM
- Malaysian LLM - may be better for Manglish but needs validation

---

### `sync_project.py`

**Purpose:** Development tool for auditing import statements in `src/`

**Status:** Development utility - not for end users

**Usage:**
```bash
python scripts/experimental/sync_project.py
```

**What it does:**
- Reads every `.py` file in `src/`
- Traces every `from .xxx import yyy` statement
- Checks if target files and names exist
- Reports exactly what is broken and what is in sync

**Does NOT:**
- Modify any files automatically
- Fix import issues
- Run tests

**Output:**
- Prints exact commands or edits needed to fix imports
- Helps identify missing modules or incorrect imports

**When to use:**
- After refactoring `src/` structure
- When adding new modules
- Debugging import errors
- Auditing code dependencies

**Why experimental:**
- Development tool, not for production use
- Rarely needed in normal workflow
- Useful for major refactoring only

---

## Guidelines

### When to Use Experimental Scripts

- You understand they are not production-ready
- You need the specific functionality they provide
- You're willing to debug if issues arise
- You're testing alternative approaches

### When NOT to Use

- For production pipelines
- When stability is critical
- If you need support/documentation
- When a stable alternative exists

---

## Moving to Production

If an experimental script proves valuable:

1. **Test thoroughly** on diverse inputs
2. **Benchmark** against existing solutions
3. **Document** usage and limitations
4. **Integrate** into main codebase
5. **Move** to `scripts/` main directory
6. **Update** documentation

---

**Last Updated:** 2026-02-09
