"""
summarize_mallam.py — Summarise a transcript using MaLLaM (Malaysian LLM).

This is a standalone module.  It reads the same transcript JSON that
your transcribe.py produces and writes a .summary.md in the same format
your existing summarize.py produces.  You can use it instead of, or
alongside, the mT5 summariser.

Usage (standalone):
    python summarize_mallam.py --transcript outputs/manglish_ho_lee_fak.json

Usage (from pipeline — see bottom of file):
    from summarize_mallam import summarize_with_mallam
    summary_md = summarize_with_mallam(transcript_json_path)

RAM notes (your machine: 7.6 GB total, 6.7 GB free, CPU only):
  - The 1.1B model in float16 is ~2.2 GB in memory.
  - Generation on CPU is slow: expect 30-120 seconds per summary.
  - The model is loaded once and kept alive if you call the function
    multiple times.  Call unload_model() when you are done.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch


# ─────────────────────────────────────────────
# module-level model cache (load once, reuse)
# ─────────────────────────────────────────────
_tokenizer = None
_model     = None
_MODEL_ID  = "mesolitica/mallam-1.1b-20k-instructions-v2"


def _load_model():
    """Load tokenizer + model if not already loaded."""
    global _tokenizer, _model
    if _model is not None:
        return                          # already loaded

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading {_MODEL_ID} …  (first time is slow, ~30s on CPU)")
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    _model     = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16,      # float16 works on CPU, bfloat16 does not
        low_cpu_mem_usage=True,         # streams weights instead of loading all at once
    )
    _model.eval()
    print(f"  Model loaded.")


def unload_model():
    """Free the model from memory.  Call this when you are done."""
    global _tokenizer, _model
    del _model, _tokenizer
    _model = _tokenizer = None
    gc.collect()


# ─────────────────────────────────────────────
# core: generate text from a prompt
# ─────────────────────────────────────────────
def _generate(prompt: str, max_new_tokens: int = 400) -> str:
    """Run inference on CPU.  Returns generated text only (no prompt)."""
    _load_model()

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    # move to same device as model (cpu)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # greedy — deterministic, faster on CPU
            temperature=1.0,
            repetition_penalty=1.15,    # prevent looping
            pad_token_id=_tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # decode only the NEW tokens (strip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"  Generated {len(new_tokens)} tokens in {elapsed:.1f}s")
    return text


# ─────────────────────────────────────────────
# load transcript (same format as your transcribe.py output)
# ─────────────────────────────────────────────
def load_transcript_text(json_path: Path) -> str:
    """Extract plain text from a Whisper-format transcript JSON."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # prefer the top-level "text" key if present
    if "text" in data and data["text"]:
        return data["text"].strip()

    # otherwise concatenate segments
    segments = data.get("segments", [])
    return " ".join(seg.get("text", "").strip() for seg in segments).strip()


# ─────────────────────────────────────────────
# prompt engineering
# ─────────────────────────────────────────────
def _build_prompt(transcript_text: str) -> str:
    """
    Build a prompt that MaLLaM understands.

    MaLLaM 1.1B instructions was tuned on simple instruction-following.
    We keep the prompt short and direct — long system prompts confuse
    small models on CPU.
    """
    # Truncate transcript if it's very long — 1.1B context is 4096 tokens,
    # and we need room for the output.  ~1500 words is a safe input limit.
    words = transcript_text.split()
    if len(words) > 1500:
        transcript_text = " ".join(words[:1500])
        transcript_text += " [...]"

    prompt = (
        "<s>[INST] Ringkas mesyuarat berikut dan kenal pasti "
        "titik-titik utama, item tindakan, dan keputusan yang dibuat.\n\n"
        f"Transkip:\n{transcript_text}\n\n"
        "Berikan ringkasan dalam format berikut:\n"
        "1. Ringkasan Eksekutif\n"
        "2. Titik-Titik Utama\n"
        "3. Item Tindakan\n"
        "4. Keputusan [/INST]\n"
    )
    return prompt


# ─────────────────────────────────────────────
# main entry point
# ─────────────────────────────────────────────
def summarize_with_mallam(transcript_json_path: Path) -> str:
    """
    Summarise a transcript using MaLLaM.

    Args:
        transcript_json_path: Path to the .json file from your transcribe.py

    Returns:
        Summary as a markdown string (same structure as your summarize.py output)
    """
    print(f"\n  Reading transcript: {transcript_json_path.name}")
    text = load_transcript_text(transcript_json_path)

    if not text:
        return "No content to summarise."

    print(f"  Transcript length: {len(text.split())} words")

    # build prompt and generate
    prompt   = _build_prompt(text)
    raw_output = _generate(prompt, max_new_tokens=400)

    # ── format into markdown ──
    # The model output is free-form.  We wrap it in our standard markdown
    # structure so it looks like what your existing summarize.py produces.
    md = _format_as_markdown(raw_output, text)
    return md


def _format_as_markdown(model_output: str, original_text: str) -> str:
    """
    Wrap the model's raw output into the same markdown structure
    that your existing summarize.py uses.
    """
    # If the model already produced numbered sections, use them directly.
    # Otherwise, put the whole output under Executive Summary and do
    # keyword extraction for the structured parts (same as your summarize.py).

    lines = []
    lines.append("## Executive Summary\n")

    # check if model output has our section markers
    has_sections = any(marker in model_output for marker in
                       ["Ringkasan Eksekutif", "Titik-Titik", "Item Tindakan", "Keputusan",
                        "Executive Summary", "Key Points", "Action Items", "Decisions"])

    if has_sections:
        # model produced structured output — use it as-is under the header
        lines.append(model_output)
    else:
        # model produced free-form text — put it as the summary,
        # then do keyword extraction for the rest
        lines.append(model_output)
        lines.append("\n")

        # ── keyword extraction (mirrors your summarize.py logic) ──
        import re
        sentences = re.split(r'[.!?]+', original_text)

        # action items
        action_keywords = ["need to", "should", "must", "will", "going to",
                           "have to", "plan to", "todo", "task", "follow up",
                           "perlu", "harus", "akan", "rancang"]
        actions = []
        for s in sentences:
            s = s.strip()
            if s and len(s) > 15 and any(k in s.lower() for k in action_keywords):
                actions.append(s)
        actions = actions[:8]

        # decisions
        decision_keywords = ["decided", "agreed", "concluded", "determined",
                             "putus", "setuju", "kesimpulan"]
        decisions = []
        for s in sentences:
            s = s.strip()
            if s and len(s) > 15 and any(k in s.lower() for k in decision_keywords):
                decisions.append(s)
        decisions = decisions[:5]

        lines.append("\n## Action Items\n")
        if actions:
            for a in actions:
                lines.append(f"- [ ] {a}")
        else:
            lines.append("- [ ] No specific action items identified")

        lines.append("\n\n## Decisions Made\n")
        if decisions:
            for d in decisions:
                lines.append(f"- {d}")
        else:
            lines.append("- No specific decisions identified")

    lines.append("\n\n---")
    lines.append("*Summary generated by MaLLaM (mesolitica/mallam-1.1b-20k-instructions-v2)*")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise a transcript using MaLLaM (Malaysian LLM)"
    )
    parser.add_argument("--transcript", required=True,
                        help="Path to transcript JSON (output of transcribe.py)")
    parser.add_argument("--out", default=None,
                        help="Output path for summary .md file. "
                             "Default: same name as transcript with .mallam.summary.md")
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    transcript_path = Path(args.transcript).expanduser().resolve()
    if not transcript_path.exists():
        print(f"\n❌  Transcript not found: {transcript_path}")
        sys.exit(1)

    # determine output path
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        # e.g.  outputs/manglish_ho_lee_fak.json  →  outputs/manglish_ho_lee_fak.mallam.summary.md
        out_path = transcript_path.with_suffix(".mallam.summary.md")

    print(f"\n{'═' * 60}")
    print(f"  MaLLaM Summariser")
    print(f"  Input : {transcript_path.name}")
    print(f"  Output: {out_path.name}")
    print(f"{'═' * 60}\n")

    summary = summarize_with_mallam(transcript_path)

    out_path.write_text(summary, encoding="utf-8")
    print(f"\n✅  Summary written to {out_path}")
    print(f"\n{'─' * 60}")
    print(summary)
    print(f"{'─' * 60}\n")

    # free memory
    unload_model()


if __name__ == "__main__":
    main()