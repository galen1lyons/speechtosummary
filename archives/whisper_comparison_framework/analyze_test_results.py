"""
Analyze Comprehensive Whisper Test Results

This script analyzes the results from comprehensive_whisper_test.py and generates:
1. Comparative analysis tables
2. Statistical summaries
3. Parameter impact analysis
4. Model comparison
5. Visualization-ready data

Usage:
    python scripts/analyze_test_results.py --results results/comprehensive_test/test_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import setup_logger

logger = setup_logger(__name__)


class TestResultsAnalyzer:
    """Analyzes comprehensive test results."""

    def __init__(self, results_file: Path):
        self.results_file = Path(results_file)
        self.results_data = self._load_results()
        self.results = self.results_data.get("results", [])

        # Filter successful tests only
        self.successful_results = [
            r for r in self.results if r.get("error") is None
        ]

        logger.info(f"Loaded {len(self.results)} tests ({len(self.successful_results)} successful)")

    def _load_results(self) -> Dict:
        """Load test results from JSON file."""
        with open(self.results_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def analyze_by_phase(self) -> Dict:
        """Analyze results grouped by test phase."""
        phase_analysis = defaultdict(lambda: {
            "tests": [],
            "avg_rtf": [],
            "avg_segments": [],
        })

        for result in self.successful_results:
            phase = result["config"]["phase"]
            rtf = result["metrics"].get("rtf", 0)
            segments = result["metrics"].get("num_segments", 0)

            phase_analysis[phase]["tests"].append(result)
            phase_analysis[phase]["avg_rtf"].append(rtf)
            phase_analysis[phase]["avg_segments"].append(segments)

        # Calculate averages
        summary = {}
        for phase, data in phase_analysis.items():
            summary[phase] = {
                "count": len(data["tests"]),
                "avg_rtf": statistics.mean(data["avg_rtf"]) if data["avg_rtf"] else 0,
                "min_rtf": min(data["avg_rtf"]) if data["avg_rtf"] else 0,
                "max_rtf": max(data["avg_rtf"]) if data["avg_rtf"] else 0,
                "avg_segments": statistics.mean(data["avg_segments"]) if data["avg_segments"] else 0,
            }

        return summary

    def analyze_by_model(self) -> Dict:
        """Compare Original Whisper vs Malaysian Whisper."""
        model_analysis = {
            "base": {"tests": [], "rtf": [], "segments": []},
            "mesolitica/malaysian-whisper-base": {"tests": [], "rtf": [], "segments": []},
        }

        for result in self.successful_results:
            model = result["config"]["model_name"]
            if model in model_analysis:
                model_analysis[model]["tests"].append(result)
                model_analysis[model]["rtf"].append(result["metrics"].get("rtf", 0))
                model_analysis[model]["segments"].append(result["metrics"].get("num_segments", 0))

        # Calculate statistics
        summary = {}
        for model, data in model_analysis.items():
            model_name = "Original Whisper" if model == "base" else "Malaysian Whisper"

            if data["rtf"]:
                summary[model_name] = {
                    "count": len(data["tests"]),
                    "avg_rtf": statistics.mean(data["rtf"]),
                    "median_rtf": statistics.median(data["rtf"]),
                    "min_rtf": min(data["rtf"]),
                    "max_rtf": max(data["rtf"]),
                    "stdev_rtf": statistics.stdev(data["rtf"]) if len(data["rtf"]) > 1 else 0,
                    "avg_segments": statistics.mean(data["segments"]),
                }
            else:
                summary[model_name] = {
                    "count": 0,
                    "avg_rtf": 0,
                    "median_rtf": 0,
                    "min_rtf": 0,
                    "max_rtf": 0,
                    "stdev_rtf": 0,
                    "avg_segments": 0,
                }

        return summary

    def analyze_language_impact(self) -> Dict:
        """Analyze impact of language parameter."""
        language_analysis = defaultdict(lambda: defaultdict(list))

        for result in self.successful_results:
            if result["config"]["phase"] == "Language":
                model = "OW" if result["config"]["model_name"] == "base" else "MW"
                language = result["config"]["language"]
                rtf = result["metrics"].get("rtf", 0)

                language_analysis[model][language].append(rtf)

        # Calculate averages
        summary = {}
        for model, lang_data in language_analysis.items():
            summary[model] = {
                lang: {
                    "avg_rtf": statistics.mean(rtfs),
                    "count": len(rtfs)
                }
                for lang, rtfs in lang_data.items()
            }

        return summary

    def analyze_beam_size_impact(self) -> Dict:
        """Analyze impact of beam size parameter."""
        beam_analysis = defaultdict(lambda: defaultdict(list))

        for result in self.successful_results:
            if result["config"]["phase"] == "BeamSize":
                model = "OW" if result["config"]["model_name"] == "base" else "MW"
                beam_size = result["config"]["beam_size"]
                rtf = result["metrics"].get("rtf", 0)

                beam_analysis[model][beam_size].append(rtf)

        # Calculate averages
        summary = {}
        for model, beam_data in beam_analysis.items():
            summary[model] = {
                beam: {
                    "avg_rtf": statistics.mean(rtfs),
                    "count": len(rtfs)
                }
                for beam, rtfs in beam_data.items()
            }

        return summary

    def analyze_temperature_impact(self) -> Dict:
        """Analyze impact of temperature parameter."""
        temp_analysis = defaultdict(lambda: defaultdict(list))

        for result in self.successful_results:
            if result["config"]["phase"] == "Temperature":
                model = "OW" if result["config"]["model_name"] == "base" else "MW"
                temperature = result["config"]["temperature"]
                rtf = result["metrics"].get("rtf", 0)

                temp_analysis[model][temperature].append(rtf)

        # Calculate averages
        summary = {}
        for model, temp_data in temp_analysis.items():
            summary[model] = {
                temp: {
                    "avg_rtf": statistics.mean(rtfs),
                    "count": len(rtfs)
                }
                for temp, rtfs in temp_data.items()
            }

        return summary

    def analyze_prompt_impact(self) -> Dict:
        """Analyze impact of initial prompt parameter."""
        prompt_analysis = defaultdict(lambda: defaultdict(list))

        prompt_labels = {
            None: "No Prompt",
            "Malaysian English conversation about scams at a mamak stall.": "Context Prompt",
            "Manglish. Code-switching between English and Malay.": "Manglish Prompt",
            "Informal Malaysian conversation.": "Informal Prompt",
        }

        for result in self.successful_results:
            if result["config"]["phase"] == "Prompt":
                model = "OW" if result["config"]["model_name"] == "base" else "MW"
                prompt = result["config"]["initial_prompt"]
                prompt_label = prompt_labels.get(prompt, "Unknown")
                rtf = result["metrics"].get("rtf", 0)

                prompt_analysis[model][prompt_label].append(rtf)

        # Calculate averages
        summary = {}
        for model, prompt_data in prompt_analysis.items():
            summary[model] = {
                prompt: {
                    "avg_rtf": statistics.mean(rtfs),
                    "count": len(rtfs)
                }
                for prompt, rtfs in prompt_data.items()
            }

        return summary

    def find_best_configs(self, top_n: int = 5) -> List[Dict]:
        """Find top N configurations by RTF (fastest)."""
        # Sort by RTF (lower is better)
        sorted_results = sorted(
            self.successful_results,
            key=lambda r: r["metrics"].get("rtf", float("inf"))
        )

        best_configs = []
        for result in sorted_results[:top_n]:
            config = result["config"]
            metrics = result["metrics"]

            best_configs.append({
                "test_id": result["test_id"],
                "model": "Original Whisper" if config["model_name"] == "base" else "Malaysian Whisper",
                "language": config["language"],
                "beam_size": config["beam_size"],
                "temperature": config["temperature"],
                "prompt": config["initial_prompt"][:30] + "..." if config["initial_prompt"] else "None",
                "rtf": metrics.get("rtf", 0),
                "processing_time": metrics.get("processing_time_s", 0),
                "segments": metrics.get("num_segments", 0),
            })

        return best_configs

    def generate_analysis_report(self, output_file: Path):
        """Generate comprehensive analysis report."""
        report = f"""# Comprehensive Whisper Test Analysis

**Analysis Date:** {self.results_data.get("test_date", "Unknown")}
**Audio File:** {self.results_data.get("audio_file", "Unknown")}
**Total Tests:** {len(self.results)}
**Successful Tests:** {len(self.successful_results)}

---

## 1. Model Comparison

"""

        # Model analysis
        model_stats = self.analyze_by_model()

        report += "### Overall Performance\n\n"
        report += "| Model | Tests | Avg RTF | Median RTF | Min RTF | Max RTF | Std Dev |\n"
        report += "|-------|-------|---------|------------|---------|---------|----------|\n"

        for model, stats in model_stats.items():
            report += (
                f"| {model} | {stats['count']} | "
                f"{stats['avg_rtf']:.3f}x | {stats['median_rtf']:.3f}x | "
                f"{stats['min_rtf']:.3f}x | {stats['max_rtf']:.3f}x | "
                f"{stats['stdev_rtf']:.3f} |\n"
            )

        # Winner determination
        ow_rtf = model_stats.get("Original Whisper", {}).get("avg_rtf", float("inf"))
        mw_rtf = model_stats.get("Malaysian Whisper", {}).get("avg_rtf", float("inf"))

        if ow_rtf < mw_rtf:
            winner = "Original Whisper"
            speedup = (mw_rtf / ow_rtf - 1) * 100
            report += f"\n**Winner:** {winner} is **{speedup:.1f}% faster** on average.\n"
        elif mw_rtf < ow_rtf:
            winner = "Malaysian Whisper"
            speedup = (ow_rtf / mw_rtf - 1) * 100
            report += f"\n**Winner:** {winner} is **{speedup:.1f}% faster** on average.\n"
        else:
            report += "\n**Winner:** Tie - both models have similar performance.\n"

        report += "\n---\n\n"

        # Phase analysis
        report += "## 2. Analysis by Test Phase\n\n"
        phase_stats = self.analyze_by_phase()

        report += "| Phase | Tests | Avg RTF | Min RTF | Max RTF |\n"
        report += "|-------|-------|---------|---------|----------|\n"

        for phase, stats in sorted(phase_stats.items()):
            report += (
                f"| {phase} | {stats['count']} | "
                f"{stats['avg_rtf']:.3f}x | {stats['min_rtf']:.3f}x | "
                f"{stats['max_rtf']:.3f}x |\n"
            )

        report += "\n---\n\n"

        # Language impact
        report += "## 3. Language Parameter Impact\n\n"
        lang_stats = self.analyze_language_impact()

        for model, languages in lang_stats.items():
            report += f"### {model}\n\n"
            report += "| Language | Avg RTF | Tests |\n"
            report += "|----------|---------|-------|\n"

            for lang, stats in sorted(languages.items()):
                report += f"| {lang} | {stats['avg_rtf']:.3f}x | {stats['count']} |\n"

            # Find best language
            best_lang = min(languages.items(), key=lambda x: x[1]["avg_rtf"])
            report += f"\n**Best Language for {model}:** `{best_lang[0]}` (RTF: {best_lang[1]['avg_rtf']:.3f}x)\n\n"

        report += "---\n\n"

        # Beam size impact
        report += "## 4. Beam Size Parameter Impact\n\n"
        beam_stats = self.analyze_beam_size_impact()

        for model, beams in beam_stats.items():
            report += f"### {model}\n\n"
            report += "| Beam Size | Avg RTF | Tests |\n"
            report += "|-----------|---------|-------|\n"

            for beam, stats in sorted(beams.items()):
                report += f"| {beam} | {stats['avg_rtf']:.3f}x | {stats['count']} |\n"

            # Find best beam size
            best_beam = min(beams.items(), key=lambda x: x[1]["avg_rtf"])
            report += f"\n**Best Beam Size for {model}:** `{best_beam[0]}` (RTF: {best_beam[1]['avg_rtf']:.3f}x)\n\n"

        report += "---\n\n"

        # Temperature impact
        report += "## 5. Temperature Parameter Impact\n\n"
        temp_stats = self.analyze_temperature_impact()

        for model, temps in temp_stats.items():
            report += f"### {model}\n\n"
            report += "| Temperature | Avg RTF | Tests |\n"
            report += "|-------------|---------|-------|\n"

            for temp, stats in sorted(temps.items()):
                report += f"| {temp} | {stats['avg_rtf']:.3f}x | {stats['count']} |\n"

            # Find best temperature
            best_temp = min(temps.items(), key=lambda x: x[1]["avg_rtf"])
            report += f"\n**Best Temperature for {model}:** `{best_temp[0]}` (RTF: {best_temp[1]['avg_rtf']:.3f}x)\n\n"

        report += "---\n\n"

        # Prompt impact
        report += "## 6. Initial Prompt Parameter Impact\n\n"
        prompt_stats = self.analyze_prompt_impact()

        for model, prompts in prompt_stats.items():
            report += f"### {model}\n\n"
            report += "| Prompt Type | Avg RTF | Tests |\n"
            report += "|-------------|---------|-------|\n"

            for prompt, stats in sorted(prompts.items(), key=lambda x: x[1]["avg_rtf"]):
                report += f"| {prompt} | {stats['avg_rtf']:.3f}x | {stats['count']} |\n"

            # Find best prompt
            best_prompt = min(prompts.items(), key=lambda x: x[1]["avg_rtf"])
            report += f"\n**Best Prompt for {model}:** `{best_prompt[0]}` (RTF: {best_prompt[1]['avg_rtf']:.3f}x)\n\n"

        report += "---\n\n"

        # Top configurations
        report += "## 7. Top 5 Fastest Configurations\n\n"
        best_configs = self.find_best_configs(top_n=5)

        report += "| Rank | Test ID | Model | Language | Beam | Temp | RTF | Time (s) |\n"
        report += "|------|---------|-------|----------|------|------|-----|----------|\n"

        for i, config in enumerate(best_configs, 1):
            report += (
                f"| {i} | {config['test_id']} | {config['model'][:10]}... | "
                f"{config['language']} | {config['beam_size']} | {config['temperature']} | "
                f"{config['rtf']:.3f}x | {config['processing_time']:.1f}s |\n"
            )

        report += "\n---\n\n"

        # Recommendations
        report += "## 8. Recommendations\n\n"
        report += "### For Production Use\n\n"

        # Get best overall config
        if best_configs:
            best = best_configs[0]
            report += f"**Recommended Configuration:**\n\n"
            report += f"- **Model:** {best['model']}\n"
            report += f"- **Language:** `{best['language']}`\n"
            report += f"- **Beam Size:** `{best['beam_size']}`\n"
            report += f"- **Temperature:** `{best['temperature']}`\n"
            report += f"- **Initial Prompt:** {best['prompt']}\n"
            report += f"- **Expected RTF:** {best['rtf']:.3f}x realtime\n"
            report += f"- **Processing Time (for ~10min audio):** {best['processing_time']:.1f}s\n\n"

        report += "### Trade-offs\n\n"
        report += "- **Speed vs Quality:** Lower beam sizes (1-3) are faster but may reduce quality\n"
        report += "- **Determinism:** Temperature 0.0 is recommended for consistent transcription\n"
        report += "- **Language Detection:** Test results show optimal language setting per model\n"
        report += "- **Initial Prompts:** Context-aware prompts may improve domain-specific accuracy\n\n"

        report += "---\n\n"

        report += "## 9. Next Steps\n\n"
        report += "1. **Manual Quality Review:** Assess transcription quality for top 5 configurations\n"
        report += "2. **Validate on Additional Audio:** Test best configs on other audio files\n"
        report += "3. **WER/CER Calculation:** Create reference transcript for quantitative evaluation\n"
        report += "4. **Production Testing:** Deploy recommended config in staging environment\n"
        report += "5. **Documentation:** Update configuration guidelines based on findings\n\n"

        # Write report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Analysis report saved to {output_file}")

        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze Comprehensive Whisper Test Results"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results/comprehensive_test/test_results.json",
        help="Path to test results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comprehensive_test/analysis_report.md",
        help="Path to output analysis report"
    )

    args = parser.parse_args()

    # Validate results file
    results_file = Path(args.results)
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        print(f"\n❌ Error: Results file not found: {results_file}")
        print("\nPlease run the comprehensive test first:")
        print('  python scripts/comprehensive_whisper_test.py --audio "data/mamak session scam.mp3"')
        return 1

    # Create analyzer
    analyzer = TestResultsAnalyzer(results_file)

    # Generate report
    print("\n" + "="*70)
    print("  ANALYZING TEST RESULTS")
    print("="*70)

    output_file = Path(args.output)
    report = analyzer.generate_analysis_report(output_file)

    print(f"\n✅ Analysis complete!")
    print(f"\nReport saved to: {output_file}")
    print("\nYou can now:")
    print("  1. Review the analysis report")
    print("  2. Generate presentation: python scripts/generate_presentation.py")

    return 0


if __name__ == "__main__":
    exit(main())
