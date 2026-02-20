"""
Optimal Whisper Configuration Testing Suite

Based on diagnosis of L3 (catastrophic failure) and L4 (functional but flawed),
this script tests 8 targeted configurations to find the optimal setup for
transcribing Manglish (Malaysian English + Malay code-switching) audio.

Key Findings from Diagnosis:
- L3 (base + language=ms): Complete hallucination loop ("hello" repeated)
- L4 (Malaysian Whisper + language=auto): Works but has localized hallucinations

Test Strategy:
- Focus on Malaysian Whisper model (proven to handle code-switching)
- Test beam_size variations (7, 10)
- Test temperature fallback (0.0, and (0.0, 0.2, 0.4))
- Test initial prompts (None vs context-aware)
- Test hallucination suppression thresholds

Usage:
    python scripts/optimal_whisper_test.py --audio "data/mamak session scam.mp3"
    python scripts/optimal_whisper_test.py --audio "data/mamak session scam.mp3" --resume
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import WhisperConfig, transcribe, setup_logger

# Setup logging
logger = setup_logger(__name__)


@dataclass
class OptimalTestConfig:
    """Configuration for a single optimal test."""
    test_id: str
    description: str
    model_name: str
    language: str
    beam_size: int
    temperature: Union[float, Tuple[float, ...]]
    initial_prompt: Optional[str]

    # Advanced Whisper parameters for hallucination control
    compression_ratio_threshold: float
    logprob_threshold: float
    no_speech_threshold: float


@dataclass
class TestResult:
    """Results from a single test."""
    test_id: str
    config: Dict
    transcription_path: str
    text_path: str
    metrics: Dict
    quality_notes: str = ""
    hallucination_count: Optional[int] = None
    segment_count: Optional[int] = None
    avg_segment_length: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = ""


class OptimalWhisperTest:
    """Manages optimal Whisper configuration testing."""

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
        self.test_configs: List[OptimalTestConfig] = []

    def load_results(self):
        """Load existing results from JSON if they exist."""
        results_file = self.results_dir / "optimal_test_results.json"
        if not results_file.exists():
            return

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for r_data in data.get("results", []):
                config_data = r_data.get("config", {})

                result = TestResult(
                    test_id=r_data.get("test_id", ""),
                    config=config_data,
                    transcription_path=r_data.get("transcription_path", ""),
                    text_path=r_data.get("text_path", ""),
                    metrics=r_data.get("metrics", {}),
                    quality_notes=r_data.get("quality_notes", ""),
                    hallucination_count=r_data.get("hallucination_count"),
                    segment_count=r_data.get("segment_count"),
                    avg_segment_length=r_data.get("avg_segment_length"),
                    error=r_data.get("error"),
                    timestamp=r_data.get("timestamp", "")
                )
                self.test_results.append(result)
            logger.info(f"Loaded {len(self.test_results)} existing test results")
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")

    def generate_test_matrix(self) -> List[OptimalTestConfig]:
        """Generate 8 optimal test configurations based on diagnosis."""
        configs = [
            # OPT1: Baseline with modest improvements
            OptimalTestConfig(
                test_id="OPT1",
                description="Baseline enhanced (beam=7, temp=0.0, no prompt)",
                model_name="mesolitica/malaysian-whisper-base",
                language="auto",
                beam_size=7,
                temperature=0.0,
                initial_prompt=None,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            ),

            # OPT2: High beam size
            OptimalTestConfig(
                test_id="OPT2",
                description="High beam size (beam=10, temp=0.0, no prompt)",
                model_name="mesolitica/malaysian-whisper-base",
                language="auto",
                beam_size=10,
                temperature=0.0,
                initial_prompt=None,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            ),

            # OPT3: Context prompt
            OptimalTestConfig(
                test_id="OPT3",
                description="Context prompt (beam=7, temp=0.0, with prompt)",
                model_name="mesolitica/malaysian-whisper-base",
                language="auto",
                beam_size=7,
                temperature=0.0,
                initial_prompt="This is a podcast about a BigPay scam phone call, spoken in Malaysian English and Malay.",
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            ),

            # OPT4: High beam + prompt
            OptimalTestConfig(
                test_id="OPT4",
                description="High beam + prompt (beam=10, temp=0.0, with prompt)",
                model_name="mesolitica/malaysian-whisper-base",
                language="auto",
                beam_size=10,
                temperature=0.0,
                initial_prompt="This is a podcast about a BigPay scam phone call, spoken in Malaysian English and Malay.",
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            ),

            # OPT5: Slightly higher temperature for diversity
            OptimalTestConfig(
                test_id="OPT5",
                description="Higher temperature (beam=7, temp=0.2, no prompt)",
                model_name="mesolitica/malaysian-whisper-base",
                language="auto",
                beam_size=7,
                temperature=0.2,
                initial_prompt=None,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            ),

            # OPT6: High beam + higher temperature
            OptimalTestConfig(
                test_id="OPT6",
                description="High beam + higher temp (beam=10, temp=0.2, no prompt)",
                model_name="mesolitica/malaysian-whisper-base",
                language="auto",
                beam_size=10,
                temperature=0.2,
                initial_prompt=None,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            ),

            # OPT7: Aggressive hallucination suppression
            OptimalTestConfig(
                test_id="OPT7",
                description="Aggressive suppression (beam=10, aggressive thresholds, with prompt)",
                model_name="mesolitica/malaysian-whisper-base",
                language="auto",
                beam_size=10,
                temperature=0.0,
                initial_prompt="This is a podcast about a BigPay scam phone call, spoken in Malaysian English and Malay.",
                compression_ratio_threshold=2.0,   # More aggressive
                logprob_threshold=-0.8,            # More aggressive
                no_speech_threshold=0.5,           # More permissive
            ),

            # OPT8: Kitchen sink (all optimizations)
            OptimalTestConfig(
                test_id="OPT8",
                description="Max quality (beam=10, temp=0.0, aggressive thresholds, with prompt)",
                model_name="mesolitica/malaysian-whisper-base",
                language="auto",
                beam_size=10,
                temperature=0.0,
                initial_prompt="This is a podcast about a BigPay scam phone call, spoken in Malaysian English and Malay.",
                compression_ratio_threshold=2.0,
                logprob_threshold=-0.8,
                no_speech_threshold=0.5,
            ),
        ]

        logger.info(f"Generated {len(configs)} optimal test configurations")
        return configs

    def is_test_completed(self, test_id: str) -> bool:
        """Check if a test has already been completed."""
        return any(r.test_id == test_id and r.error is None for r in self.test_results)

    def run_single_test(self, config: OptimalTestConfig) -> TestResult:
        """Run a single test configuration."""
        logger.info("=" * 80)
        logger.info(f"Running Test: {config.test_id}")
        logger.info(f"Description: {config.description}")
        logger.info("=" * 80)

        # Prepare output paths
        output_base = self.output_dir / config.test_id

        try:
            start_time = time.time()

            # Prepare temperature parameter (handle tuple for fallback)
            temperature_param = config.temperature if isinstance(config.temperature, float) else config.temperature[0]

            logger.info(f"Config: beam={config.beam_size}, temp={config.temperature}, "
                       f"compression_ratio={config.compression_ratio_threshold}, "
                       f"logprob={config.logprob_threshold}, "
                       f"no_speech={config.no_speech_threshold}")

            # Run transcription with all parameters
            json_path, txt_path, metrics = transcribe(
                audio_path=self.audio_path,
                out_base=output_base,
                model_name=config.model_name,
                language=config.language,
                device="auto",
                beam_size=config.beam_size,
                temperature=temperature_param,
                initial_prompt=config.initial_prompt,
                compression_ratio_threshold=config.compression_ratio_threshold,
                logprob_threshold=config.logprob_threshold,
                no_speech_threshold=config.no_speech_threshold,
            )

            elapsed_time = time.time() - start_time

            # Analyze quality
            quality_metrics = self.analyze_quality(json_path)

            # Create result
            result = TestResult(
                test_id=config.test_id,
                config=asdict(config),
                transcription_path=str(json_path),
                text_path=str(txt_path),
                metrics=metrics,
                quality_notes=quality_metrics["notes"],
                hallucination_count=quality_metrics["hallucination_count"],
                segment_count=quality_metrics["segment_count"],
                avg_segment_length=quality_metrics["avg_segment_length"],
                timestamp=datetime.now().isoformat()
            )

            logger.info(f"✅ Test {config.test_id} completed in {elapsed_time:.1f}s")
            logger.info(f"   Segments: {quality_metrics['segment_count']}")
            logger.info(f"   Avg segment length: {quality_metrics['avg_segment_length']:.1f}s")
            logger.info(f"   Hallucination score: {quality_metrics['hallucination_count']}")

            return result

        except Exception as e:
            logger.error(f"❌ Test {config.test_id} failed: {e}")
            return TestResult(
                test_id=config.test_id,
                config=asdict(config),
                transcription_path="",
                text_path="",
                metrics={},
                error=str(e),
                timestamp=datetime.now().isoformat()
            )

    def analyze_quality(self, json_path: Path) -> Dict:
        """Analyze transcription quality."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = data.get("segments", [])
            full_text = data.get("text", "")

            # Count segments
            segment_count = len(segments)

            # Calculate average segment length
            if segments:
                segment_lengths = [seg.get("end", 0) - seg.get("start", 0) for seg in segments]
                avg_segment_length = sum(segment_lengths) / len(segment_lengths)
            else:
                avg_segment_length = 0.0

            # Detect hallucinations (repetitive patterns)
            hallucination_count = self.count_hallucinations(full_text)

            # Generate quality notes
            notes = []
            if hallucination_count > 100:
                notes.append(f"HIGH hallucination count ({hallucination_count})")
            elif hallucination_count > 20:
                notes.append(f"MODERATE hallucination count ({hallucination_count})")
            else:
                notes.append(f"LOW hallucination count ({hallucination_count})")

            if avg_segment_length > 60:
                notes.append("Long segments (poor granularity)")
            elif avg_segment_length < 10:
                notes.append("Very short segments (over-segmented)")
            else:
                notes.append("Good segment length")

            # Check if audio starts near 0s
            first_start = segments[0].get("start", 0) if segments else 0
            if first_start > 10:
                notes.append(f"WARNING: Audio starts at {first_start:.1f}s (missing content?)")

            return {
                "segment_count": segment_count,
                "avg_segment_length": avg_segment_length,
                "hallucination_count": hallucination_count,
                "notes": " | ".join(notes)
            }

        except Exception as e:
            logger.warning(f"Quality analysis failed: {e}")
            return {
                "segment_count": 0,
                "avg_segment_length": 0.0,
                "hallucination_count": 0,
                "notes": f"Analysis failed: {e}"
            }

    def count_hallucinations(self, text: str) -> int:
        """
        Count potential hallucinations by detecting repetitive patterns.

        Strategy:
        - Split into words
        - Look for consecutive repeated words (> 5 repetitions)
        - Count total repetitions
        """
        words = text.lower().split()

        if not words:
            return 0

        hallucination_count = 0
        i = 0

        while i < len(words):
            current_word = words[i]
            repeat_count = 1

            # Count consecutive repetitions
            j = i + 1
            while j < len(words) and words[j] == current_word:
                repeat_count += 1
                j += 1

            # If repeated > 5 times, count as hallucination
            if repeat_count > 5:
                hallucination_count += repeat_count - 5  # Only count excess

            i = j if repeat_count > 1 else i + 1

        return hallucination_count

    def save_results(self):
        """Save results to JSON file."""
        results_file = self.results_dir / "optimal_test_results.json"

        results_data = {
            "audio_file": str(self.audio_path),
            "test_run_date": datetime.now().isoformat(),
            "total_tests": len(self.test_configs),
            "completed_tests": len([r for r in self.test_results if r.error is None]),
            "failed_tests": len([r for r in self.test_results if r.error is not None]),
            "results": [asdict(r) for r in self.test_results]
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {results_file}")

    def print_summary(self):
        """Print summary of all test results."""
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMAL TEST SUMMARY")
        logger.info("=" * 80)

        successful_tests = [r for r in self.test_results if r.error is None]
        failed_tests = [r for r in self.test_results if r.error is not None]

        logger.info(f"Total tests: {len(self.test_results)}")
        logger.info(f"Successful: {len(successful_tests)}")
        logger.info(f"Failed: {len(failed_tests)}")

        if successful_tests:
            logger.info("\n--- Quality Rankings (by hallucination count, lower is better) ---")
            ranked = sorted(successful_tests, key=lambda r: r.hallucination_count or float('inf'))

            for i, result in enumerate(ranked, 1):
                logger.info(f"\n{i}. {result.test_id} - {result.config.get('description', 'N/A')}")
                logger.info(f"   Hallucinations: {result.hallucination_count}")
                logger.info(f"   Segments: {result.segment_count}")
                logger.info(f"   Avg segment: {result.avg_segment_length:.1f}s")
                logger.info(f"   Notes: {result.quality_notes}")
                logger.info(f"   RTF: {result.metrics.get('rtf', 'N/A')}")

        if failed_tests:
            logger.info("\n--- Failed Tests ---")
            for result in failed_tests:
                logger.info(f"  {result.test_id}: {result.error}")

        logger.info("\n" + "=" * 80)

    def run_all_tests(self, resume: bool = False, dry_run: bool = False):
        """Run all tests in the matrix."""
        self.test_configs = self.generate_test_matrix()

        logger.info(f"\nOptimal test suite with {len(self.test_configs)} configurations")
        logger.info(f"Audio: {self.audio_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Resume mode: {resume}")
        logger.info(f"Dry run mode: {dry_run}\n")

        # Dry run mode: just show what would be tested
        if dry_run:
            logger.info("=" * 80)
            logger.info("DRY RUN - Test Configuration Preview")
            logger.info("=" * 80)
            for i, config in enumerate(self.test_configs, 1):
                status = "✓ COMPLETED" if self.is_test_completed(config.test_id) else "⏸ PENDING"
                logger.info(f"\n{i}. {config.test_id} [{status}]")
                logger.info(f"   Description: {config.description}")
                logger.info(f"   Model: {config.model_name}")
                logger.info(f"   Parameters: beam={config.beam_size}, temp={config.temperature}, "
                           f"lang={config.language}")
                logger.info(f"   Advanced: compression_ratio={config.compression_ratio_threshold}, "
                           f"logprob={config.logprob_threshold}, "
                           f"no_speech={config.no_speech_threshold}")
                if config.initial_prompt:
                    logger.info(f"   Prompt: {config.initial_prompt[:60]}...")

            completed = sum(1 for c in self.test_configs if self.is_test_completed(c.test_id))
            logger.info(f"\n" + "=" * 80)
            logger.info(f"Summary: {completed}/{len(self.test_configs)} tests already completed")
            logger.info(f"Remove --dry-run to execute tests")
            logger.info("=" * 80)
            return

        # Regular execution mode
        logger.info("Starting test execution...\n")
        for i, config in enumerate(self.test_configs, 1):
            logger.info(f"\n[Test {i}/{len(self.test_configs)}] {config.test_id}")

            # Skip if already completed and in resume mode
            if resume and self.is_test_completed(config.test_id):
                logger.info(f"⏭️  Skipping {config.test_id} (already completed)")
                continue

            # Run test
            result = self.run_single_test(config)

            # Store result (replace if exists)
            self.test_results = [r for r in self.test_results if r.test_id != config.test_id]
            self.test_results.append(result)

            # Save after each test
            self.save_results()

            logger.info(f"Progress: {i}/{len(self.test_configs)} tests completed\n")

        # Final summary
        self.print_summary()
        logger.info(f"\n✅ All tests completed! Results saved to: {self.results_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Run optimal Whisper configuration tests based on L3/L4 diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/optimal_whisper_test.py --audio "data/mamak session scam.mp3"

  # Resume interrupted test run
  python scripts/optimal_whisper_test.py --audio "data/mamak session scam.mp3" --resume

  # Preview tests without running
  python scripts/optimal_whisper_test.py --audio "data/mamak session scam.mp3" --dry-run
        """
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file to test"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/optimal_test",
        help="Directory for transcription outputs (default: outputs/optimal_test)"
    )
    parser.add_argument(
        "--results-dir",
        default="outputs/optimal_test",
        help="Directory for test results and analysis (default: outputs/optimal_test)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume testing from where it left off"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview test configurations without actually running them"
    )
    return parser


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()

    # Validate audio file
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        logger.error(f"Please provide a valid audio file path using --audio")
        sys.exit(1)

    if not audio_path.is_file():
        logger.error(f"Path is not a file: {audio_path}")
        sys.exit(1)

    # Log configuration
    logger.info(f"Starting Optimal Whisper Test Suite")
    logger.info(f"Audio file: {audio_path}")
    logger.info(f"Audio size: {audio_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Resume mode: {args.resume}")

    # Create and run test suite
    test_suite = OptimalWhisperTest(
        audio_path=audio_path,
        output_dir=output_dir,
        results_dir=results_dir
    )

    try:
        test_suite.run_all_tests(resume=args.resume, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.warning("\n\nTest suite interrupted by user")
        logger.info("Progress has been saved. Use --resume to continue from where you left off")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nTest suite failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
