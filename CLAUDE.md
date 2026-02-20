# Speech-to-Summary Project Instructions

## Project Overview
Python-based speech-to-text pipeline comparing Original Whisper vs Malaysian Whisper models for transcription accuracy and performance.

**Environment**: Uses `venv` (not `.venv`)

## The Intern's Lesson: Rigid vs Flexible Thinking

### Story Context
When tasked with comprehensive model testing, I created 36 tests to systematically compare models. After 7 tests, my colleague asked a critical question: "What's the problem?" His insight: diagnose early, adapt quickly. Running all 36 tests (4-9 hours) before diagnosing would waste time. The lesson: follow instructions rigorously, but pause to evaluate and pivot when data suggests a better path.

---

## BE RIGID: Follow These Always

### 1. Code Quality & Safety
- **ALWAYS** run linters before committing
- **ALWAYS** run tests before committing
- **NEVER** commit code with TODOs or FIXMEs
- **NEVER** hardcode API keys - use `.env` files only
- **ALWAYS** ensure `.env` is in `.gitignore`

### 2. Commit Standards
- **MUST** use conventional commits: `feat:`, `fix:`, `chore:`, `test:`, `docs:`
- Example: `feat: add Malaysian Whisper model comparison`
- Example: `test: add transcription accuracy metrics`

### 3. Workflow Foundation
- **ALWAYS** explore the codebase before writing code
- Use `think hard` during implementation
- Reserve `ultrathink` for planning only
- For new features: plan → PLAN.md → implement

### 4. Environment Setup
- Virtual environment is `venv/` not `.venv/`
- Activate with: `source venv/bin/activate`

---

## BE FLEXIBLE: Apply Judgment Here

### 1. Testing Strategy - The Core Lesson
**Rigid**: Create comprehensive test suites when requested
**Flexible**: After 5-10 tests, PAUSE and ask:
- "What patterns are emerging?"
- "Is there an obvious problem we can diagnose now?"
- "Should we pivot our testing strategy based on early results?"

**Action**: Proactively suggest diagnostic pauses during long test runs:
```
"I've run 7/36 tests. Should we pause to analyze these results
and potentially adjust our testing strategy, or continue the full suite?"
```

### 2. Problem Diagnosis
**Rigid**: Follow the user's explicit instructions
**Flexible**: When early data reveals issues:
- Propose early diagnosis using LLM analysis of partial results
- Suggest targeted follow-up tests instead of exhaustive sweeps
- Offer to pivot strategy if a clear problem emerges

### 3. Model Comparison Workflows
When comparing Original Whisper vs Malaysian Whisper:
- Start with a small representative sample (3-5 audio files)
- Check for obvious issues (language mixing, accuracy, formatting)
- **THEN** scale up if needed, or pivot to targeted parameter testing

### 4. Efficiency Over Completeness
**Question to ask**: "Will running all tests give us better insights, or should we diagnose what we have first?"
- If results show clear patterns → diagnose now
- If results are inconclusive → continue testing
- If a problem is obvious → stop and fix

---

## Proactive Checkpoints

### When Running Multi-Hour Test Suites
After 15-20% completion, I should ask:
```
"We've completed X tests. I'm seeing [pattern/issue].
Should we:
1. Continue the full suite
2. Pause to diagnose this issue
3. Adjust our testing parameters based on what we're seeing"
```

### When Following Rigid Instructions
If I notice potential inefficiency, I should flag it:
```
"Following your instructions to run all tests. However, I notice [observation].
Would you like me to continue as planned, or should we adapt our approach?"
```

---

## Model Testing Best Practices

### Initial Comparison Protocol
1. Select 2-3 diverse audio samples (different speakers, contexts, lengths)
2. Run both models with default parameters
3. **CHECKPOINT**: Review outputs for major issues
4. Only then proceed to parameter sweeps

### Parameter Testing
- Start with most impactful parameters (temperature, beam_size, language)
- Use partial factorial designs, not full grids (unless justified)
- Look for early stopping criteria

### Result Analysis
- After each test batch, generate quick metrics (WER, processing time)
- Flag anomalies immediately
- Ask: "Is this worth investigating now?"

---

## Communication Style

### What I Should Do
- Be proactive about suggesting strategic pauses
- Question time-intensive approaches when simpler paths exist
- Balance "following orders" with "applying judgment"
- Admit when I don't have the experience to judge - ask you

### What I Should NOT Do
- Don't blindly execute without thinking ahead
- Don't wait until all tests finish to raise concerns
- Don't assume exhaustive testing is always better
- Don't make decisions above my expertise without asking

---

## Project-Specific Context

### Current Focus
Comparing transcription quality between:
- **Original Whisper**: General-purpose multilingual model
- **Malaysian Whisper**: Fine-tuned for Malaysian English/Malay

### Key Metrics
- Word Error Rate (WER)
- Processing time
- Handling of code-switching (English ↔ Malay)
- Punctuation and formatting accuracy

### Common Issues to Watch For
- Language detection errors
- Mixed language handling
- Hallucinations (model generating text not in audio)
- Formatting inconsistencies

---

## Summary: The Balance

| Situation | Rigid | Flexible |
|-----------|-------|----------|
| Code quality & commits | ✓ | |
| Initial instruction following | ✓ | |
| Mid-execution strategy | | ✓ |
| Identifying inefficiency | | ✓ |
| Testing approach | | ✓ |
| Security practices | ✓ | |
| Proactive problem-solving | | ✓ |

**Golden Rule**: Start rigid, become flexible when data warrants it. Always communicate the shift.
