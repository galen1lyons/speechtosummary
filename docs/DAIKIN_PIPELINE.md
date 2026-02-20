# Daikin Malaysia — Meeting Transcription Pipeline
## Phased Build Plan (IDE Agent Handoff Document)

> **Note:** This is a separate future project for Daikin customer engagement, distinct from the core speechtosummary pipeline.

---

> **How to read this document:**
> Each phase is a discrete unit of work. Each phase contains a **CONTEXT** block (why), a **CONSTRAINTS** block (hard rules the agent must not violate), and an **IMPLEMENTATION** block (what to actually do). Phases must be executed in order. Do not skip or reorder.

---

## Meta-Instruction (read before anything else)

```
You are building an accuracy-first, on-prem meeting transcription and summarisation
pipeline for a Malaysian-based Japanese company.

Priority order (non-negotiable):
  1. Accuracy — over speed, over elegance, over anything else.
  2. Stability — pinned dependencies, graceful degradation, no bleeding-edge experiments.
  3. Debuggability — engineers will maintain this, not ML researchers.

The pipeline must degrade gracefully at every stage. If diarization fails, transcription
still works. If the local LLM fails, the raw transcript is still outputted. Nothing
should silently produce bad output.
```

---

## Phase 0 — Language Hierarchy & Routing Decision

**CONTEXT:**
This is a Japanese company. Japanese is not an add-on language — it is a primary
language in management and HQ-level meetings. The ChatGPT plan treated Malay/English
as core and Japanese/Chinese as secondary. This is wrong for Daikin.

**CONSTRAINTS:**
```
Language priority (reflects actual meeting importance, not alphabetical order):
  1. English       — cross-functional lingua franca
  2. Japanese      — HQ / management decisions
  3. Malay/Manglish — local operational discussion
  4. Chinese       — engineering, suppliers, regional teams
                     (may be Mandarin OR Cantonese — see Phase 2)

Summarisation routing MUST follow this hierarchy.
Do not treat all four languages symmetrically.
```

**DECISION TO LOCK (no agent should re-litigate this):**
```
Japanese and Chinese segments in the final summary are NOT optional.
They are not "verbatim dump" fallback content.
They must be routed to a capable model for summarisation.

On-prem option (Phase 4 Option A): skip LLM rewriting for JA/ZH, output
  the raw transcription segments clearly labelled — but DO include them in
  the structured output with their speaker and timestamp.
Cloud option (Phase 4 Option B): route JA/ZH segments to a cloud LLM API
  for summarisation, keep everything else on-prem.

The agent must implement BOTH paths as a config toggle. Default to Option A
(on-prem only) for the CPU validation run.
```

---

## Phase 1 — Environment Setup (CPU-first validation)

**CONTEXT:**
The user wants to validate transcription accuracy on CPU before moving to the A4000.
This means the environment must be runnable without any GPU. The diarization step
(pyannote) is the most fragile dependency — it has known conflicts with modern
torch/torchaudio and runs painfully slow on CPU. The plan is to isolate it.

**CONSTRAINTS:**
```
- Do NOT try to make pyannote compatible with modern torch on the same env.
- Implement a two-environment split. The pipeline must run end-to-end even if
  the diarization environment is absent (graceful degradation).
- CPU compute_type must be int8 everywhere. Do not use float16 on CPU.
- Do not install any GPU-only packages (flash attention, onnxruntime-gpu) in
  the CPU validation environment.
```

**IMPLEMENTATION:**

Create two environment files:

```
env_core/
  └── requirements.txt          # core pipeline (transcription + post-processing)

env_diarization/
  └── requirements.txt          # pyannote isolated environment
```

**env_core/requirements.txt — pinned versions:**
```
# Note: Requires Python 3.10 or 3.11 (not 3.12+)
faster-whisper==1.0.0
whisperx==3.1.1
langdetect==1.0.9
torch>=2.1.0
torchaudio>=2.1.0
numpy>=1.24.0
ffmpeg-python==0.2.0
```

**env_diarization/requirements.txt — pinned versions:**
```
# Note: Requires Python 3.10 or 3.11 (not 3.12+)
torch==2.0.1
torchaudio==2.0.1
numpy==1.26.4
pyannote.audio==3.1.1
pyannote.core==5.0.0
pyannote.pipeline==3.0.1
```

**Subprocess bridge:**
The core pipeline calls diarization via subprocess. Communication is JSON only.
```
Input:  path to audio file (WAV, 16kHz mono)
Output: JSON array of diarization segments
        [{"speaker": "SPEAKER_00", "start": 0.0, "end": 12.4}, ...]
```

If the diarization subprocess fails, exits non-zero, or is not available, the
core pipeline falls back to transcribing the full audio as a single block with
no speaker labels. Log a warning. Do not crash.

---

## Phase 2 — ASR Transcription (Whisper large-v3 via faster-whisper, CPU int8)

**CONTEXT:**
Whisper large-v3 is the only open-source model that covers all four languages
(English, Japanese, Mandarin, Cantonese, Malay) at acceptable accuracy in one
model. Do not use Whisper turbo — it degrades on Cantonese specifically.
Do not use Malaysian Whisper — it only covers Malay/English.

**CONSTRAINTS:**
```
- Model: openai/whisper-large-v3 via faster-whisper backend
- Compute type: int8 (CPU validation path)
- Long-form algorithm: sequential (not chunked). Sequential is up to 0.5% WER
  more accurate. We are accuracy-first.
- Batch size: 1 on CPU. Do not batch on CPU.
- Do NOT set a global language flag. Language detection must run per-segment.
  Exception: see Cantonese handling below.
- VAD preprocessing: ENABLED. This is not optional. It prevents hallucination
  on silent/music sections — a known and serious issue with Whisper.
```

**CANTONESE HANDLING (critical — this was missing from the original plan):**
```
Whisper large-v3 auto-detection cannot reliably distinguish Cantonese from
Mandarin. It will label Cantonese audio as "zh" (Chinese), which causes it
to transcribe in written-language format instead of spoken-language format.

The pipeline must include a post-detection correction step:
  1. Run transcription with language=None (auto-detect).
  2. If detected language is "zh", run a secondary check:
     - Re-run the first 5 seconds of that segment with language="yue" forced.
     - Compare the two outputs. If the "yue" output is coherent, use it and
       relabel the segment as Cantonese.
  3. This is a heuristic. Log the decision for manual review.

Note: This only matters if your Malaysian Chinese speakers are actually
Cantonese. If they are Mandarin speakers, "zh" detection is fine. This step
can be toggled off via config if confirmed Mandarin-only.
```

**IMPLEMENTATION:**
```python
# Pseudocode for the agent — do not copy literally, implement properly

from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cpu", compute_type="int8")

def transcribe_segment(audio_path, start_time, end_time):
    # Extract segment from audio
    segment_audio = extract_audio(audio_path, start_time, end_time)

    # First pass: auto-detect language
    segments, info = model.transcribe(
        segment_audio,
        language=None,           # auto-detect
        vad_filter=True,         # REQUIRED — prevents silence hallucination
        beam_size=5,             # stability over speed
        repetition_penalty=1.2,  # mitigate repetitive text generation
    )

    detected_lang = info.language

    # Cantonese correction check
    if detected_lang == "zh":
        detected_lang = cantonese_check(segment_audio, segments)

    return segments, detected_lang

def cantonese_check(segment_audio, zh_segments):
    """
    Re-transcribe first 5 seconds with forced Cantonese to check if it's
    actually Cantonese audio misdetected as Mandarin.
    """
    # Extract first 5 seconds of audio
    short_clip = extract_first_n_seconds(segment_audio, duration=5)

    # Re-run with forced Cantonese
    yue_segments, _ = model.transcribe(
        short_clip,
        language="yue",  # force Cantonese
        vad_filter=True,
        beam_size=5,
    )

    # Heuristic: if Cantonese output is coherent (non-empty, reasonable length),
    # assume it's Cantonese. This is imperfect but better than nothing.
    yue_text = " ".join([s.text for s in yue_segments])
    zh_text = " ".join([s.text for s in zh_segments])[:len(yue_text)]

    # Simple heuristic: if yue output has similar length and isn't garbled
    if len(yue_text) > 10 and len(yue_text) > 0.5 * len(zh_text):
        logger.info(f"Cantonese detected (relabeled from zh to yue)")
        return "yue"

    return "zh"  # default to Mandarin
```

---

## Phase 3 — Failure-Mode Detection (degenerate output guard)

**CONTEXT:**
Whisper has two well-documented failure modes that produce silently bad output:
  1. Repetitive text — the seq2seq architecture repeats segments, especially on
     noisy or low-speech audio.
  2. Silence hallucination — on silent or music-only sections, Whisper generates
     plausible-sounding but completely fabricated text.

Both of these will end up in the summary if not caught. The original plan only
covered (1). This phase covers both.

**CONSTRAINTS:**
```
- This step runs AFTER transcription, BEFORE summarisation.
- It must flag segments, not silently delete them.
- Flagged segments are included in the output as "low_confidence" and are
  excluded from the summary extraction pass.
- Detection thresholds are configurable. Do not hardcode magic numbers.
```

**IMPLEMENTATION — detection rules:**
```
Rule 1: Repetitive text
  - Split segment text into overlapping n-grams (n=5 words).
  - If any n-gram appears more than 3 times in the segment: flag as repetitive.
  - Threshold: configurable, default 3.

Rule 2: Filler run
  - Tokenise segment. If ratio of filler tokens ("um", "uh", "ah", "er",
    equivalent tokens in JA/ZH/MS) exceeds 40%: flag as filler-heavy.
  - Filler lists are per-language. Must be loaded from a config file.

Rule 3: Silence hallucination
  - Cross-reference the VAD output from Phase 2 with the transcription
    timestamps. If a transcription segment falls entirely within a VAD-detected
    silence region: flag as hallucinated.
  - This is the most reliable check. Do not skip it.

Rule 4: Identical cross-segment duplication
  - If two consecutive segments have identical text: flag both, keep only
    the first occurrence.
```

**Output schema for flagged segments:**
```json
{
  "segment_id": 7,
  "text": "...",
  "flag": "repetitive",          // or "filler_heavy", "hallucinated", "duplicate"
  "confidence": "low",
  "include_in_summary": false
}
```

---

## Phase 4 — Language-Aware Summarisation (two-pass, routed)

**CONTEXT:**
Summarisation must be language-aware. MaLLaM 3B (or 1.1B on CPU) is strong on
Malay/English but weak on Japanese and Chinese. Running Japanese segments through
MaLLaM will produce degraded or hallucinated output. This must not happen.

On CPU validation: MaLLaM 1.1B quantized to GGUF Q4_K_M (~750MB) is the only
feasible local option. It will be slow but usable for post-meeting batch work.

**CONSTRAINTS:**
```
- Do not summarise or rewrite Japanese or Chinese segments using MaLLaM.
- Two summarisation paths must exist, toggled by config:
    Option A (on-prem only):  JA/ZH segments are included in output verbatim
                              with speaker + timestamp, clearly labelled as
                              "original language — not summarised".
    Option B (cloud fallback): JA/ZH segments are sent to a cloud LLM API
                              (endpoint configurable) for summarisation.
                              See Appendix D for API specification.
- Default for CPU validation run: Option A.
- The final executive summary is always in English.
- Malay/English segments are summarised locally via MaLLaM.
```

**TWO-PASS SUMMARISATION STRUCTURE:**

```
Pass 1 — Extraction (structured, per-language)
  Input:  Transcribed segments (filtered by Phase 3), tagged with language codes
  Output: Extracted items per category

  Categories:
    - decisions_made
    - action_items (owner, action, due_date, confidence)
    - open_questions
    - technical_notes

  Rules:
    - For EN/MS segments: extract using MaLLaM locally.
    - For JA/ZH segments: either preserve verbatim (Option A) or route to
      cloud API (Option B). Do not extract using MaLLaM.
    - Confidence scores must be included. "high" if extracted cleanly,
      "low" if the source segment was flagged in Phase 3.
    - Prefer omission over hallucination. If unsure whether something is
      a decision or an action item, omit it and flag as "unclassified".

Pass 2 — Executive Rewrite (English, unified)
  Input:  Extraction output from Pass 1
  Output: Single coherent English executive summary

  Rules:
    - For items sourced from EN/MS: rewrite into polished English.
    - For items sourced from JA/ZH (Option A): include as
      "Reported in [language] — see original transcript for details."
    - For items sourced from JA/ZH (Option B): include the cloud-summarised
      version directly.
    - Confidence levels must be surfaced: "High confidence" / "Low confidence —
      verify before acting on".
```

**OUTPUT JSON SCHEMA:**
```json
{
  "metadata": {
    "meeting_file": "string",
    "transcribed_at": "ISO-8601 timestamp",
    "languages_detected": ["en", "ms", "ja", "zh"],
    "cantonese_detected": false,
    "diarization_enabled": true,
    "summarisation_mode": "on_prem",          // or "cloud_fallback"
    "low_confidence_segments_count": 2
  },
  "executive_summary": "string — English, unified, polished",
  "action_items": [
    {
      "owner": "SPEAKER_01",
      "action": "string",
      "due_date": "string or null",
      "confidence": "high",
      "source_language": "en",
      "source_segment_ids": [3, 4]
    }
  ],
  "decisions_made": [
    {
      "description": "string",
      "confidence": "high",
      "source_language": "ja",
      "source_segment_ids": [7],
      "note": "Reported in Japanese — see original transcript for details."
    }
  ],
  "open_questions": ["string"],
  "technical_notes": ["string"],
  "original_language_segments": {
    "ja": [
      {
        "speaker": "SPEAKER_02",
        "start": 145.2,
        "end": 162.8,
        "text": "original Japanese transcription text"
      }
    ],
    "zh": []
  },
  "low_confidence_segments": [
    {
      "segment_id": 12,
      "flag": "repetitive",
      "text": "..."
    }
  ]
}
```

---

## Phase 5 — CPU Validation Run (accuracy check before GPU)

**CONTEXT:**
This is the phase that was missing entirely from the original ChatGPT plan.
Before touching the A4000, we need to confirm that the transcription output is
actually accurate on a real sample. This is the "does it work" checkpoint.

**CONSTRAINTS:**
```
- Run on CPU with int8 compute. This will be slow. That is expected.
- Use a real or realistic test audio clip. Ideally a short (~5-10 min)
  Zoom/Teams recording that contains at least two of: Malay/Manglish, Japanese,
  English. If you don't have one, create a synthetic test case with known
  ground truth text.
- Do NOT measure WER as the primary metric. WER is misleading for multilingual
  code-switched audio. Instead, do a qualitative review:
    1. Are the language boundaries detected correctly?
    2. Is the Cantonese/Mandarin distinction working (if applicable)?
    3. Are hallucinated segments being caught by Phase 3?
    4. Is the speaker diarization (if enabled) assigning segments to the
       right speakers?
    5. Does the executive summary make sense and accurately reflect what
       was said?
- Document the results. This is what you show to stakeholders internally.
  "Qualitative improvements" are more defensible than raw WER numbers for
  this use case.
```

**IMPLEMENTATION:**
```
Create a test script: run_validation.py

It must:
  1. Accept a path to a test audio file and an optional ground-truth transcript.
  2. Run the full pipeline end-to-end (Phase 2 → Phase 3 → Phase 4).
  3. Output:
     - The raw transcription (with language tags per segment)
     - The flagged segments report (Phase 3)
     - The structured JSON summary (Phase 4)
     - A side-by-side comparison if ground truth is provided
  4. Print timing information for each stage (for later GPU benchmarking).
  5. Log everything. Verbose by default on CPU validation.
```

---

## Phase 6 — Output, Handoff & README

**CONTEXT:**
The pipeline will be handed off to engineers. It must be self-documenting.

**CONSTRAINTS:**
```
- Pipeline must run end-to-end even if diarization is disabled.
- All dependencies must be frozen in lockfiles (requirements.txt is not enough —
  use pip freeze > requirements.lock for both environments).
- A README must exist and must be accurate.
```

**README must include:**
```
1. What this pipeline does (one paragraph, plain English)
2. Hardware assumptions:
   - CPU validation: any modern x86 CPU, 8GB+ RAM, ~4GB free disk
   - Production: RTX A4000 (16GB VRAM), 64GB RAM, 250GB storage
3. Language support matrix:
   - Which languages are transcribed (EN, MS, JA, ZH, Cantonese)
   - Which languages are summarised locally vs routed to cloud
4. How to run:
   - CPU validation: exact commands
   - Production (A4000): noted as "Phase 7 — not yet implemented"
5. Configuration:
   - List every config toggle and its default
   - Cantonese check on/off
   - Diarization on/off
   - Summarisation mode (on_prem / cloud_fallback)
6. Expected runtimes:
   - CPU: "approximately X minutes for a 10-minute meeting" (fill in after
     Phase 5 validation)
   - GPU: "to be benchmarked"
7. Known limitations:
   - Cantonese auto-detection is a heuristic, not reliable
   - Overlapping speech is not handled well
   - Speaker labels are generic (SPEAKER_00, etc.) — manual name assignment needed
   - MaLLaM is weak on Japanese/Chinese — do not route those segments to it
8. Dependencies:
   - HuggingFace token required for pyannote models
   - User agreements required: pyannote/segmentation-3.0, pyannote/speaker-diarization-3.1
   - FFmpeg must be installed system-wide
```

---

## Phase 7 — GPU Migration (A4000) — DO NOT IMPLEMENT YET

**CONTEXT:**
This phase is locked until Phase 5 validation is complete and results are reviewed.
It is documented here so the agent knows it exists and does not try to optimise
for GPU prematurely.

**What changes when moving to A4000:**
```
- compute_type: int8 → float16
- batch_size: 1 → 8 (tunable)
- MaLLaM: 1.1B Q4_K_M (GGUF, llama.cpp) → 3B NF4 (bitsandbytes, HuggingFace)
- Diarization: can run in-process on GPU instead of subprocess (faster, but
  re-test the dependency conflicts first)
- Sequential model loading to manage VRAM:
    Whisper large-v3 float16 (~6GB) → unload
    pyannote + wav2vec2 (~2-3GB) → unload
    MaLLaM 3B NF4 (~2GB) → unload
- Colab is kept as a backup option but is NOT the primary target.
  The A4000 is a persistent company server — prefer that.
```

---

## Appendix A — Model Registry

| Model | Role | Size (disk) | CPU feasible? | Notes |
|---|---|---|---|---|
| openai/whisper-large-v3 | ASR (all languages) | ~3GB (int8) | Yes, slow | via faster-whisper. Do NOT use turbo. |
| pyannote/speaker-diarization-3.1 | Speaker diarization | ~200MB | Yes, very slow | Isolated environment. Requires HF token + agreement. |
| mesolitica/mallam-1.1B-4096 | Post-processing (EN/MS only) | ~750MB (Q4_K_M GGUF) | Yes | via llama.cpp. CPU validation path. |
| mesolitica/mallam-3B-4096 | Post-processing (EN/MS only) | ~2GB (NF4) | No | GPU only. Phase 7. |
| wav2vec2 (alignment) | Word-level timestamps | ~300MB | Yes | Language-specific models. Auto-selected by WhisperX. |

---

## Appendix B — Config File Template

```yaml
# pipeline_config.yaml
# All toggles are here. The agent must not hardcode any of these.

transcription:
  model: "large-v3"
  backend: "faster-whisper"
  compute_type: "int8"              # int8 for CPU, float16 for GPU
  device: "cpu"                     # cpu or cuda
  batch_size: 1                     # 1 for CPU, 8 for GPU
  long_form_algorithm: "sequential" # sequential or chunked. Use sequential.
  vad_filter: true                  # NEVER set to false

cantonese:
  detection_enabled: true           # set to false if confirmed Mandarin-only
  check_duration_seconds: 5         # how long to re-run for yue check

diarization:
  enabled: true                     # set to false if env_diarization is unavailable
  subprocess_timeout_seconds: 300   # fail gracefully after this

failure_detection:
  repetition_ngram_size: 5
  repetition_max_count: 3
  filler_ratio_threshold: 0.40
  filler_lists_path: "./config/filler_words.json"  # See Appendix E for schema

summarisation:
  mode: "on_prem"                   # on_prem or cloud_fallback
  cloud_api_endpoint: ""            # required if mode is cloud_fallback (see Appendix D)
  cloud_api_key: "${CLOUD_API_KEY}" # loaded from environment
  cloud_api_timeout: 30             # seconds
  local_llm:
    model: "mesolitica/mallam-1.1B-4096"  # or path to GGUF: mallam-1.1B-Q4_K_M.gguf
    backend: "llama_cpp"                   # llama_cpp for CPU, bitsandbytes for GPU

output:
  format: "json"
  include_raw_transcript: true
  include_low_confidence_segments: true
```

---

## Appendix C — Filler Words Configuration

**File: `config/filler_words.json`**

This file defines language-specific filler words used in Phase 3 failure detection.

```json
{
  "en": [
    "um", "uh", "ah", "er", "hmm", "like", "you know", "actually", "basically",
    "literally", "sort of", "kind of", "i mean", "well", "so", "right"
  ],
  "ms": [
    "em", "ah", "er", "hmm", "kan", "lah", "mah", "lor", "leh",
    "apa", "macam", "tu", "ni", "itu", "ini"
  ],
  "ja": [
    "えと", "あの", "ええ", "まあ", "その", "なんか", "ちょっと",
    "eto", "ano", "ee", "maa", "sono", "nanka", "chotto"
  ],
  "zh": [
    "嗯", "啊", "呃", "那个", "就是", "然后", "这个",
    "en", "a", "e", "nage", "jiushi", "ranhou", "zhege"
  ],
  "yue": [
    "咁", "啦", "呢", "嘅", "嗰", "喎", "啫",
    "gam", "la", "ne", "ge", "go", "wo", "je"
  ]
}
```

**Usage notes:**
- Include both native script and romanization where applicable
- Thresholds are configurable in `pipeline_config.yaml`
- These lists are not exhaustive — extend based on observed patterns

---

## Appendix D — Cloud API Specification (Option B Summarisation)

When `summarisation.mode` is set to `cloud_fallback`, Japanese and Chinese segments
are routed to an external API for summarisation.

**Required configuration:**
```yaml
summarisation:
  mode: "cloud_fallback"
  cloud_api_endpoint: "https://api.example.com/v1/summarize"
  cloud_api_key: "${CLOUD_API_KEY}"  # loaded from environment variable
  cloud_api_timeout: 30               # seconds
```

**Request format (POST):**
```json
{
  "segments": [
    {
      "text": "original Japanese or Chinese transcription text",
      "language": "ja",
      "speaker": "SPEAKER_02",
      "start": 145.2,
      "end": 162.8
    }
  ],
  "task": "extract_structured",
  "categories": ["decisions_made", "action_items", "open_questions", "technical_notes"],
  "output_language": "en"
}
```

**Expected response format:**
```json
{
  "status": "success",
  "extractions": {
    "decisions_made": [
      {
        "description": "English summary of the decision",
        "confidence": "high",
        "source_segment_indices": [0]
      }
    ],
    "action_items": [
      {
        "owner": "SPEAKER_02",
        "action": "English description of action",
        "due_date": null,
        "confidence": "medium",
        "source_segment_indices": [0]
      }
    ],
    "open_questions": [],
    "technical_notes": []
  }
}
```

**Error handling:**
- If the API returns non-200 status, fall back to Option A (verbatim inclusion)
- Log the error with full request/response for debugging
- Include a note in the final output: "Cloud summarisation failed — original text preserved"
- Never block pipeline execution on API failures

**Rate limiting:**
- Batch multiple segments per request when possible (max 10 segments per request)
- Implement exponential backoff on 429 responses
- Configurable max retries (default: 3)

---

## Appendix E — What NOT to do (guardrails for the agent)

```
1. Do not use Whisper turbo. It degrades on Cantonese.
2. Do not use Malaysian Whisper. It only covers Malay/English.
3. Do not route Japanese or Chinese segments to MaLLaM. It will hallucinate.
4. Do not set a global language flag on Whisper. Per-segment detection is required.
5. Do not disable VAD filtering. Silence hallucination is a real and common problem.
6. Do not use float16 on CPU. Use int8.
7. Do not try to make pyannote and modern torch coexist in one environment.
8. Do not optimise for speed at this stage. Accuracy first.
9. Do not use flash attention on CPU.
10. Do not assume "zh" means Mandarin. It might be Cantonese.
```