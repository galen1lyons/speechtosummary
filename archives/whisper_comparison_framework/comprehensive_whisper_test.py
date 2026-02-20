"""
Comprehensive Whisper Model Testing Suite

Tests multiple Whisper models across various configuration parameters:
- Model: OpenAI Whisper base vs Malaysian Whisper
- Language: auto, en, ms
- Beam size: 1, 3, 5, 7, 10
- Temperature: 0.0, 0.2, 0.4, 0.6, 0.8
- Initial prompt: None, Context-specific prompts

Usage:
    python scripts/comprehensive_whisper_test.py --audio "data/mamak session scam.mp3"
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import WhisperConfig, transcribe, setup_logger

# Setup logging
logger = setup_logger(__name__)


@dataclass
class TestConfig:
    """Configuration for a single test."""
    test_id: str
    phase: str
    model_name: str
    language: str
    beam_size: int
    temperature: float
    initial_prompt: Optional[str]
    description: str


@dataclass
class TestResult:
    """Results from a single test."""
    test_id: str
    config: Dict
    transcription_path: str
    text_path: str
    metrics: Dict
    quality_notes: str = ""
    manual_quality_score: Optional[int] = None  # 1-5 scale
    error: Optional[str] = None
    timestamp: str = ""


class ComprehensiveWhisperTest:
    """Manages comprehensive Whisper testing."""

    def __init__(self, audio_path: Path, output_dir: Path, results_dir: Path):
        self.audio_path = Path(audio_path)
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Storage for results
        self.test_results: List[TestResult] = []
        self.load_results()

        # Test configurations
        self.test_configs: List[TestConfig] = []

    def load_results(self):
        """Load existing results from JSON if they exist."""
        results_file = self.results_dir / "test_results.json"
        if not results_file.exists():
            return

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            for r_data in data.get("results", []):
                config_data = r_data.get("config", {})
                
                # Create TestConfig from loaded data
                test_config = TestConfig(
                    test_id=r_data.get("test_id", ""),
                    phase=config_data.get("phase", ""),
                    model_name=config_data.get("model_name", ""),
                    language=config_data.get("language", ""),
                    beam_size=config_data.get("beam_size", 5),
                    temperature=config_data.get("temperature", 0.0),
                    initial_prompt=config_data.get("initial_prompt"),
                    description=config_data.get("description", "")
                )
                
                result = TestResult(
                    test_id=r_data.get("test_id", ""),
                    config=asdict(test_config),
                    transcription_path=r_data.get("transcription_path", ""),
                    text_path=r_data.get("text_path", ""),
                    metrics=r_data.get("metrics", {}),
                    quality_notes=r_data.get("quality_notes", ""),
                    manual_quality_score=r_data.get("manual_quality_score"),
                    error=r_data.get("error"),
                    timestamp=r_data.get("timestamp", "")
                )
                self.test_results.append(result)
            logger.info(f"Loaded {len(self.test_results)} existing test results")
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")

    def generate_test_matrix(self) -> List[TestConfig]:
        """Generate all test configurations."""
        configs = []

        # Phase 1: Baseline Tests
        configs.extend(self._generate_baseline_tests())

        # Phase 2: Language Tests
        configs.extend(self._generate_language_tests())

        # Phase 3: Beam Size Tests
        configs.extend(self._generate_beam_size_tests())

        # Phase 4: Temperature Tests
        configs.extend(self._generate_temperature_tests())

        # Phase 5: Initial Prompt Tests
        configs.extend(self._generate_prompt_tests())

        logger.info(f"Generated {len(configs)} test configurations")
        return configs

    def _generate_baseline_tests(self) -> List[TestConfig]:
        """Phase 1: Baseline tests with default parameters."""
        return [
            TestConfig(
                test_id="B1",
                phase="Baseline",
                model_name="base",
                language="auto",
                beam_size=5,
                temperature=0.0,
                initial_prompt=None,
                description="Original Whisper - Default config"
            ),
            TestConfig(
                test_id="B2",
                phase="Baseline",
                model_name="mesolitica/malaysian-whisper-base",
                language="auto",
                beam_size=5,
                temperature=0.0,
                initial_prompt=None,
                description="Malaysian Whisper - Default config"
            ),
        ]

    def _generate_language_tests(self) -> List[TestConfig]:
        """Phase 2: Language detection tests."""
        configs = []
        test_num = 1

        for model in ["base", "mesolitica/malaysian-whisper-base"]:
            model_short = "OW" if model == "base" else "MW"

            for lang in ["auto", "en", "ms"]:
                configs.append(TestConfig(
                    test_id=f"L{test_num}",
                    phase="Language",
                    model_name=model,
                    language=lang,
                    beam_size=5,
                    temperature=0.0,
                    initial_prompt=None,
                    description=f"{model_short} - Language: {lang}"
                ))
                test_num += 1

        return configs

    def _generate_beam_size_tests(self) -> List[TestConfig]:
        """Phase 3: Beam size tests."""
        configs = []
        test_num = 1

        for model in ["base", "mesolitica/malaysian-whisper-base"]:
            model_short = "OW" if model == "base" else "MW"

            for beam in [1, 3, 5, 7, 10]:
                configs.append(TestConfig(
                    test_id=f"BS{test_num}",
                    phase="BeamSize",
                    model_name=model,
                    language="auto",
                    beam_size=beam,
                    temperature=0.0,
                    initial_prompt=None,
                    description=f"{model_short} - Beam size: {beam}"
                ))
                test_num += 1

        return configs

    def _generate_temperature_tests(self) -> List[TestConfig]:
        """Phase 4: Temperature tests."""
        configs = []
        test_num = 1

        for model in ["base", "mesolitica/malaysian-whisper-base"]:
            model_short = "OW" if model == "base" else "MW"

            for temp in [0.0, 0.2, 0.4, 0.6, 0.8]:
                configs.append(TestConfig(
                    test_id=f"T{test_num}",
                    phase="Temperature",
                    model_name=model,
                    language="auto",
                    beam_size=5,
                    temperature=temp,
                    initial_prompt=None,
                    description=f"{model_short} - Temperature: {temp}"
                ))
                test_num += 1

        return configs

    def _generate_prompt_tests(self) -> List[TestConfig]:
        """Phase 5: Initial prompt tests."""
        configs = []
        test_num = 1

        prompts = [
            None,
            "Malaysian English conversation about scams at a mamak stall.",
            "Manglish. Code-switching between English and Malay.",
            "Informal Malaysian conversation.",
        ]

        prompt_descriptions = [
            "No prompt",
            "Context prompt (scam/mamak)",
            "Manglish prompt",
            "Informal prompt",
        ]

        for model in ["base", "mesolitica/malaysian-whisper-base"]:
            model_short = "OW" if model == "base" else "MW"

            for prompt, prompt_desc in zip(prompts, prompt_descriptions):
                configs.append(TestConfig(
                    test_id=f"P{test_num}",
                    phase="Prompt",
                    model_name=model,
                    language="auto",
                    beam_size=5,
                    temperature=0.0,
                    initial_prompt=prompt,
                    description=f"{model_short} - {prompt_desc}"
                ))
                test_num += 1

        return configs

    def run_single_test(self, test_config: TestConfig) -> TestResult:
        """Run a single test configuration."""
        logger.info(f"Running test {test_config.test_id}: {test_config.description}")

        # Create output filename
        model_short = "base" if test_config.model_name == "base" else "malaysian"
        prompt_short = "none" if test_config.initial_prompt is None else f"prompt{test_config.test_id}"

        filename = (
            f"{test_config.test_id}_"
            f"{model_short}_"
            f"{test_config.language}_"
            f"beam{test_config.beam_size}_"
            f"temp{test_config.temperature}_"
            f"{prompt_short}"
        )

        out_base = self.output_dir / filename

        # Create WhisperConfig
        whisper_config = WhisperConfig(
            model_name=test_config.model_name,
            language=test_config.language,
            device="auto",
            beam_size=test_config.beam_size,
            temperature=test_config.temperature,
            initial_prompt=test_config.initial_prompt,
        )

        # Run transcription
        start_time = time.time()

        try:
            # Only pass parameters that transcribe() accepts
            json_path, txt_path, metrics = transcribe(
                audio_path=self.audio_path,
                out_base=out_base,
                model_name=whisper_config.model_name,
                language=whisper_config.language,
                beam_size=whisper_config.beam_size,
                temperature=whisper_config.temperature,
                initial_prompt=whisper_config.initial_prompt,
                device=whisper_config.device,
            )

            elapsed_time = time.time() - start_time

            # Create result
            result = TestResult(
                test_id=test_config.test_id,
                config=asdict(test_config),
                transcription_path=str(json_path),
                text_path=str(txt_path),
                metrics=metrics,
                timestamp=datetime.now().isoformat(),
            )

            logger.info(
                f"✅ Test {test_config.test_id} complete - "
                f"RTF: {metrics['rtf']:.2f}x, "
                f"Time: {elapsed_time:.1f}s, "
                f"Segments: {metrics['num_segments']}"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Test {test_config.test_id} failed: {e}")

            result = TestResult(
                test_id=test_config.test_id,
                config=asdict(test_config),
                transcription_path="",
                text_path="",
                metrics={},
                error=str(e),
                timestamp=datetime.now().isoformat(),
            )

            return result

    def run_all_tests(self, start_from: Optional[str] = None, limit: Optional[int] = None):
        """Run all test configurations."""
        self.test_configs = self.generate_test_matrix()

        # Apply start_from filter
        if start_from:
            start_idx = next(
                (i for i, cfg in enumerate(self.test_configs) if cfg.test_id == start_from),
                0
            )
            self.test_configs = self.test_configs[start_idx:]
            logger.info(f"Starting from test {start_from}")

        # Apply limit
        if limit:
            self.test_configs = self.test_configs[:limit]
            logger.info(f"Limited to {limit} tests")

        total_tests = len(self.test_configs)
        logger.info(f"Running {total_tests} tests")

        # Run tests
        for i, test_config in enumerate(self.test_configs, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Test {i}/{total_tests}: {test_config.test_id}")
            logger.info(f"{'='*70}")

            result = self.run_single_test(test_config)
            self.test_results.append(result)

            # Save intermediate results
            self.save_results()

            logger.info(f"Progress: {i}/{total_tests} tests complete")

        logger.info(f"\n🎉 All {total_tests} tests complete!")

    def save_results(self):
        """Save test results to JSON."""
        results_file = self.results_dir / "test_results.json"

        # If file exists, we've already loaded them in __init__
        # self.test_results contains both old (loaded) and new (ran) results
        
        # We want to ensure we don't have duplicates and keep the latest
        results_dict = {}
        for r in self.test_results:
            results_dict[r.test_id] = r
            
        results_list = list(results_dict.values())

        results_data = {
            "audio_file": str(self.audio_path),
            "test_date": datetime.now().isoformat(),
            "total_tests": len(results_list),
            "results": [asdict(r) for r in results_list],
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {results_file}")

    def generate_summary_report(self):
        """Generate a summary report of all tests."""
        report_file = self.results_dir / "test_summary.md"

        # Calculate statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.error is None)
        failed_tests = total_tests - successful_tests

        # Group by phase
        phase_stats = {}
        for result in self.test_results:
            phase = result.config["phase"]
            if phase not in phase_stats:
                phase_stats[phase] = {"total": 0, "success": 0, "failed": 0}

            phase_stats[phase]["total"] += 1
            if result.error is None:
                phase_stats[phase]["success"] += 1
            else:
                phase_stats[phase]["failed"] += 1

        # Generate report
        report = f"""# Comprehensive Whisper Test Summary

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Audio File:** {self.audio_path.name}

## Overall Results

- **Total Tests:** {total_tests}
- **Successful:** {successful_tests}
- **Failed:** {failed_tests}
- **Success Rate:** {(successful_tests/total_tests*100):.1f}%

## Phase Breakdown

| Phase | Total | Success | Failed |
|-------|-------|---------|--------|
"""

        for phase, stats in phase_stats.items():
            report += f"| {phase} | {stats['total']} | {stats['success']} | {stats['failed']} |\n"

        report += "\n## Test Results\n\n"

        # Add successful tests table
        report += "### Successful Tests\n\n"
        report += "| Test ID | Phase | Model | Language | Beam | Temp | RTF | Segments |\n"
        report += "|---------|-------|-------|----------|------|------|-----|----------|\n"

        for result in self.test_results:
            if result.error is None:
                config = result.config
                model_short = "OW" if "base" == config["model_name"] else "MW"
                rtf = result.metrics.get("rtf", 0)
                segments = result.metrics.get("num_segments", 0)

                report += (
                    f"| {result.test_id} | {config['phase']} | {model_short} | "
                    f"{config['language']} | {config['beam_size']} | "
                    f"{config['temperature']} | {rtf:.2f}x | {segments} |\n"
                )

        # Add failed tests if any
        if failed_tests > 0:
            report += "\n### Failed Tests\n\n"
            report += "| Test ID | Phase | Model | Error |\n"
            report += "|---------|-------|-------|-------|\n"

            for result in self.test_results:
                if result.error is not None:
                    config = result.config
                    model_short = "OW" if "base" == config["model_name"] else "MW"
                    report += (
                        f"| {result.test_id} | {config['phase']} | {model_short} | "
                        f"{result.error[:50]}... |\n"
                    )

        report += "\n## Next Steps\n\n"
        report += "1. Review transcription quality for each test\n"
        report += "2. Assign manual quality scores (1-5)\n"
        report += "3. Run analysis script to identify best configurations\n"
        report += "4. Generate visualizations\n"

        # Write report
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Summary report saved to {report_file}")
        print(f"\n{report}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Whisper Model Testing Suite"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file to test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/comprehensive_test",
        help="Directory for transcription outputs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/comprehensive_test",
        help="Directory for test results"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        help="Start from specific test ID (e.g., BS1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tests to run"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1

    # Create test suite
    test_suite = ComprehensiveWhisperTest(
        audio_path=audio_path,
        output_dir=Path(args.output_dir),
        results_dir=Path(args.results_dir),
    )

    # Print header
    print("\n" + "="*70)
    print("  COMPREHENSIVE WHISPER MODEL TESTING SUITE")
    print("="*70)
    print(f"\nAudio File: {audio_path.name}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Results Directory: {args.results_dir}")

    # Generate test matrix
    test_suite.generate_test_matrix()

    # Print test plan
    print(f"\nTest Plan:")
    print(f"  Phase 1: Baseline Tests (2 tests)")
    print(f"  Phase 2: Language Tests (6 tests)")
    print(f"  Phase 3: Beam Size Tests (10 tests)")
    print(f"  Phase 4: Temperature Tests (10 tests)")
    print(f"  Phase 5: Initial Prompt Tests (8 tests)")
    print(f"  Total: {len(test_suite.test_configs)} tests")

    if args.limit:
        print(f"\n⚠️  Limited to {args.limit} tests")

    if args.start_from:
        print(f"\n⚠️  Starting from test {args.start_from}")

    # Confirm before starting (unless --yes flag is set)
    if not args.yes:
        print("\n" + "="*70)
        input("Press ENTER to start tests (or Ctrl+C to cancel)...")

    # Run tests
    start_time = time.time()
    test_suite.run_all_tests(start_from=args.start_from, limit=args.limit)
    total_time = time.time() - start_time

    # Generate summary
    test_suite.generate_summary_report()

    # Print completion message
    print("\n" + "="*70)
    print(f"  TESTING COMPLETE")
    print("="*70)
    print(f"\nTotal Time: {total_time/60:.1f} minutes")
    print(f"Results Directory: {args.results_dir}")
    print("\nNext Steps:")
    print("  1. Review test results in results/comprehensive_test/")
    print("  2. Run analysis script: python scripts/analyze_test_results.py")
    print("  3. Generate presentation: python scripts/generate_presentation.py")

    return 0


if __name__ == "__main__":
    exit(main())
