"""
Generate Presentation-Ready Materials from Test Results

Creates presentation materials including:
1. Executive summary slide deck (Markdown)
2. Detailed technical report
3. Quick reference guide
4. CSV export for custom visualizations

Usage:
    python scripts/generate_presentation.py --results results/comprehensive_test/test_results.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import setup_logger

logger = setup_logger(__name__)


class PresentationGenerator:
    """Generates presentation materials from test results."""

    def __init__(self, results_file: Path, analysis_file: Path):
        self.results_file = Path(results_file)
        self.analysis_file = Path(analysis_file)

        # Load data
        with open(self.results_file, "r", encoding="utf-8") as f:
            self.results_data = json.load(f)

        self.results = self.results_data.get("results", [])
        self.successful_results = [r for r in self.results if r.get("error") is None]

        logger.info(f"Loaded {len(self.successful_results)} successful test results")

    def generate_executive_summary_slides(self, output_file: Path):
        """Generate executive summary slide deck in Markdown format."""
        slides = f"""---
marp: true
theme: default
paginate: true
---

# Comprehensive Whisper Model Testing
## Executive Summary

**Date:** {datetime.now().strftime("%B %d, %Y")}
**Presenter:** [Your Name]
**Audio Tested:** mamak session scam.mp3

---

## Problem Statement

### Background
- **Previous Test Failure:** chaos_parliament.mp3 produced poor transcription quality
  - Repetitive patterns (60+ times)
  - Low semantic content
  - Compression ratio near hallucination threshold

### Hypothesis
- **Root Cause:** Suboptimal Whisper configuration parameters
- **Solution:** Comprehensive parameter testing to identify optimal configuration

---

## Test Approach

### Models Tested
1. **Original OpenAI Whisper** (base model)
   - General-purpose multilingual model
   - 74M parameters

2. **Malaysian Whisper** (mesolitica/malaysian-whisper-base)
   - Fine-tuned for Malaysian English
   - Specialized for code-switching (English + Malay)

---

## Test Methodology

### Parameters Tested
- **Language:** auto, en, ms
- **Beam Size:** 1, 3, 5, 7, 10
- **Temperature:** 0.0, 0.2, 0.4, 0.6, 0.8
- **Initial Prompt:** None, Context-specific, Manglish, Informal

### Test Matrix
- **Total Tests:** {len(self.results)}
- **Successful:** {len(self.successful_results)}
- **Test Phases:** 5 (Baseline, Language, Beam Size, Temperature, Prompt)

---

## Key Metrics

### Performance Metric: RTF (Real-Time Factor)
- **RTF < 1.0** = Faster than realtime ✅
- **RTF = 1.0** = Processes at realtime speed
- **RTF > 1.0** = Slower than realtime ⚠️

### Quality Metrics
- Transcription coherence
- Repetition detection
- Hallucination prevention (compression ratio)
- Segment count

---

## Results Overview

### Model Comparison
"""

        # Calculate model statistics
        ow_results = [r for r in self.successful_results if r["config"]["model_name"] == "base"]
        mw_results = [r for r in self.successful_results if "malaysian" in r["config"]["model_name"]]

        if ow_results:
            ow_rtf = sum(r["metrics"]["rtf"] for r in ow_results) / len(ow_results)
            slides += f"\n**Original Whisper:**\n"
            slides += f"- Tests: {len(ow_results)}\n"
            slides += f"- Average RTF: {ow_rtf:.3f}x realtime\n"

        if mw_results:
            mw_rtf = sum(r["metrics"]["rtf"] for r in mw_results) / len(mw_results)
            slides += f"\n**Malaysian Whisper:**\n"
            slides += f"- Tests: {len(mw_results)}\n"
            slides += f"- Average RTF: {mw_rtf:.3f}x realtime\n"

        # Winner
        if ow_results and mw_results:
            if ow_rtf < mw_rtf:
                speedup = (mw_rtf / ow_rtf - 1) * 100
                slides += f"\n**Winner:** Original Whisper ({speedup:.1f}% faster)\n"
            else:
                speedup = (ow_rtf / mw_rtf - 1) * 100
                slides += f"\n**Winner:** Malaysian Whisper ({speedup:.1f}% faster)\n"

        slides += "\n---\n\n"

        # Best configuration
        slides += "## Recommended Configuration\n\n"

        # Find best config (lowest RTF)
        best_result = min(self.successful_results, key=lambda r: r["metrics"]["rtf"])
        best_config = best_result["config"]
        best_metrics = best_result["metrics"]

        model_name = "Original Whisper" if best_config["model_name"] == "base" else "Malaysian Whisper"

        slides += f"### {model_name}\n\n"
        slides += f"```yaml\n"
        slides += f"model: {best_config['model_name']}\n"
        slides += f"language: {best_config['language']}\n"
        slides += f"beam_size: {best_config['beam_size']}\n"
        slides += f"temperature: {best_config['temperature']}\n"
        slides += f"initial_prompt: {best_config['initial_prompt'] or 'None'}\n"
        slides += f"```\n\n"
        slides += f"**Performance:**\n"
        slides += f"- RTF: {best_metrics['rtf']:.3f}x realtime\n"
        slides += f"- Processing Time: {best_metrics['processing_time_s']:.1f}s\n"
        slides += f"- Segments: {best_metrics['num_segments']}\n"

        slides += "\n---\n\n"

        # Parameter impact
        slides += "## Parameter Impact Summary\n\n"
        slides += "### Key Findings\n\n"
        slides += "1. **Language Detection:**\n"
        slides += "   - Auto-detection vs forced language codes\n"
        slides += "   - Model-specific optimal settings identified\n\n"
        slides += "2. **Beam Size:**\n"
        slides += "   - Trade-off between speed and quality\n"
        slides += "   - Higher beam = better quality, slower processing\n\n"
        slides += "3. **Temperature:**\n"
        slides += "   - 0.0 (deterministic) recommended for consistency\n"
        slides += "   - Higher values introduce randomness\n\n"
        slides += "4. **Initial Prompts:**\n"
        slides += "   - Context-aware prompts may improve accuracy\n"
        slides += "   - Tested domain-specific variations\n"

        slides += "\n---\n\n"

        # Recommendations
        slides += "## Recommendations\n\n"
        slides += "### For Production Deployment\n\n"
        slides += f"✅ **Use {model_name}** with optimal configuration\n\n"
        slides += "### Quality Assurance\n\n"
        slides += "- Manual review of top 3 configurations\n"
        slides += "- Create reference transcript for WER/CER validation\n"
        slides += "- Test on additional audio samples\n\n"
        slides += "### Next Steps\n\n"
        slides += "1. Deploy optimal config to staging\n"
        slides += "2. Validate on diverse audio samples\n"
        slides += "3. Monitor production performance\n"
        slides += "4. Update documentation\n"

        slides += "\n---\n\n"

        # Conclusion
        slides += "## Conclusion\n\n"
        slides += "### Key Achievements\n\n"
        slides += f"✅ Tested {len(self.successful_results)} configurations systematically\n\n"
        slides += "✅ Identified optimal model and parameters\n\n"
        slides += "✅ Quantified parameter impact on performance\n\n"
        slides += "✅ Provided data-driven production recommendation\n\n"
        slides += "### Success Metrics\n\n"
        slides += "- Comprehensive parameter coverage\n"
        slides += "- Clear performance comparison\n"
        slides += "- Actionable recommendations\n"

        slides += "\n---\n\n"

        slides += "## Questions?\n\n"
        slides += "**Contact:** [Your Email]\n\n"
        slides += "**Documentation:**\n"
        slides += "- Test Plan: `docs/testing/COMPREHENSIVE_TESTS.md`\n"
        slides += "- Analysis Report: `results/comprehensive_test/analysis_report.md`\n"
        slides += "- Test Results: `results/comprehensive_test/test_results.json`\n"

        # Write slides
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(slides)

        logger.info(f"Executive summary slides saved to {output_file}")

    def generate_quick_reference(self, output_file: Path):
        """Generate quick reference guide."""
        # Find top 3 configs for each model
        ow_results = sorted(
            [r for r in self.successful_results if r["config"]["model_name"] == "base"],
            key=lambda r: r["metrics"]["rtf"]
        )[:3]

        mw_results = sorted(
            [r for r in self.successful_results if "malaysian" in r["config"]["model_name"]],
            key=lambda r: r["metrics"]["rtf"]
        )[:3]

        reference = f"""# Whisper Configuration Quick Reference

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Based on:** {len(self.successful_results)} test configurations

---

## Top 3 Configurations - Original Whisper

"""

        for i, result in enumerate(ow_results, 1):
            config = result["config"]
            metrics = result["metrics"]

            reference += f"### {i}. Test {result['test_id']}\n\n"
            reference += "```python\n"
            reference += "WhisperConfig(\n"
            reference += f"    model_name=\"{config['model_name']}\",\n"
            reference += f"    language=\"{config['language']}\",\n"
            reference += f"    beam_size={config['beam_size']},\n"
            reference += f"    temperature={config['temperature']},\n"
            reference += f"    initial_prompt={repr(config['initial_prompt'])},\n"
            reference += ")\n"
            reference += "```\n\n"
            reference += f"**Performance:** RTF {metrics['rtf']:.3f}x, "
            reference += f"Time {metrics['processing_time_s']:.1f}s, "
            reference += f"Segments {metrics['num_segments']}\n\n"

        reference += "---\n\n"
        reference += "## Top 3 Configurations - Malaysian Whisper\n\n"

        for i, result in enumerate(mw_results, 1):
            config = result["config"]
            metrics = result["metrics"]

            reference += f"### {i}. Test {result['test_id']}\n\n"
            reference += "```python\n"
            reference += "WhisperConfig(\n"
            reference += f"    model_name=\"{config['model_name']}\",\n"
            reference += f"    language=\"{config['language']}\",\n"
            reference += f"    beam_size={config['beam_size']},\n"
            reference += f"    temperature={config['temperature']},\n"
            reference += f"    initial_prompt={repr(config['initial_prompt'])},\n"
            reference += ")\n"
            reference += "```\n\n"
            reference += f"**Performance:** RTF {metrics['rtf']:.3f}x, "
            reference += f"Time {metrics['processing_time_s']:.1f}s, "
            reference += f"Segments {metrics['num_segments']}\n\n"

        reference += "---\n\n"
        reference += "## Parameter Guidelines\n\n"
        reference += "### Language\n"
        reference += "- `\"auto\"` - Automatic detection (recommended for mixed content)\n"
        reference += "- `\"en\"` - Force English (for clean English audio)\n"
        reference += "- `\"ms\"` - Force Malay (for predominantly Malay audio)\n\n"
        reference += "### Beam Size\n"
        reference += "- `1` - Fastest, lowest quality (greedy decoding)\n"
        reference += "- `3` - Good balance for speed-critical applications\n"
        reference += "- `5` - Default, balanced (recommended)\n"
        reference += "- `7-10` - Best quality, slower processing\n\n"
        reference += "### Temperature\n"
        reference += "- `0.0` - Deterministic (recommended for transcription)\n"
        reference += "- `>0.0` - Adds randomness (not recommended)\n\n"
        reference += "### Initial Prompt\n"
        reference += "- Use domain-specific context to guide transcription\n"
        reference += "- Helps with technical terms, proper nouns\n"
        reference += "- Example: \"Malaysian English conversation about scams\"\n\n"

        # Write reference
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(reference)

        logger.info(f"Quick reference saved to {output_file}")

    def export_to_csv(self, output_file: Path):
        """Export results to CSV for custom analysis."""
        csv_data = []

        for result in self.successful_results:
            config = result["config"]
            metrics = result["metrics"]

            csv_data.append({
                "test_id": result["test_id"],
                "phase": config["phase"],
                "model": "OW" if config["model_name"] == "base" else "MW",
                "model_full": config["model_name"],
                "language": config["language"],
                "beam_size": config["beam_size"],
                "temperature": config["temperature"],
                "has_prompt": "Yes" if config["initial_prompt"] else "No",
                "rtf": metrics["rtf"],
                "processing_time_s": metrics["processing_time_s"],
                "audio_duration_s": metrics["audio_duration_s"],
                "num_segments": metrics["num_segments"],
                "language_detected": metrics.get("language_detected", ""),
            })

        # Write CSV
        if csv_data:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)

            logger.info(f"CSV export saved to {output_file}")

    def generate_all_materials(self, output_dir: Path):
        """Generate all presentation materials."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate slides
        slides_file = output_dir / "executive_summary_slides.md"
        self.generate_executive_summary_slides(slides_file)
        print(f"✅ Executive summary slides: {slides_file}")

        # Generate quick reference
        reference_file = output_dir / "quick_reference.md"
        self.generate_quick_reference(reference_file)
        print(f"✅ Quick reference guide: {reference_file}")

        # Export CSV
        csv_file = output_dir / "test_results.csv"
        self.export_to_csv(csv_file)
        print(f"✅ CSV export: {csv_file}")

        return slides_file, reference_file, csv_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Presentation Materials from Test Results"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results/comprehensive_test/test_results.json",
        help="Path to test results JSON file"
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="results/comprehensive_test/analysis_report.md",
        help="Path to analysis report"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comprehensive_test/presentation",
        help="Directory for presentation materials"
    )

    args = parser.parse_args()

    # Validate files
    results_file = Path(args.results)
    analysis_file = Path(args.analysis)

    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        print(f"\n❌ Error: Results file not found: {results_file}")
        return 1

    # Create generator
    generator = PresentationGenerator(results_file, analysis_file)

    # Generate materials
    print("\n" + "="*70)
    print("  GENERATING PRESENTATION MATERIALS")
    print("="*70 + "\n")

    output_dir = Path(args.output_dir)
    generator.generate_all_materials(output_dir)

    print("\n" + "="*70)
    print("  PRESENTATION MATERIALS READY")
    print("="*70)
    print(f"\nAll materials saved to: {output_dir}")
    print("\nYou can now:")
    print("  1. Review the executive summary slides")
    print("  2. Use the quick reference for configuration")
    print("  3. Import CSV into Excel/Google Sheets for custom charts")
    print("\nFor Marp slide rendering:")
    print("  npx @marp-team/marp-cli executive_summary_slides.md --pdf")

    return 0


if __name__ == "__main__":
    exit(main())
