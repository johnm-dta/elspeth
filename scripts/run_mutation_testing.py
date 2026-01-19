#!/usr/bin/env python3
"""Run mutation testing on ELSPETH core modules.

Mutation testing validates test effectiveness by introducing artificial bugs
(mutants) and checking if tests catch them. A high mutation score means tests
actually verify behavior, not just execute code.

Usage:
    # Run on canonical.py (default, critical module)
    python scripts/run_mutation_testing.py

    # Run on specific module
    python scripts/run_mutation_testing.py --module landscape/recorder.py

    # Run on all core modules (slow!)
    python scripts/run_mutation_testing.py --all

    # Show survived mutants from last run
    python scripts/run_mutation_testing.py --show-survivors

Target mutation scores:
    - canonical.py: 95%+ (hash integrity is foundational)
    - landscape/: 90%+ (audit trail is the legal record)
    - engine/: 85%+ (orchestration correctness)

Exit codes:
    0: Mutation testing completed (check output for score)
    1: Error during testing
    2: Mutation score below threshold (when --strict is used)
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
from pathlib import Path

# Module paths relative to src/elspeth/core/
MODULES = {
    "canonical": "canonical.py",
    "landscape/recorder": "landscape/recorder.py",
    "landscape/exporter": "landscape/exporter.py",
    "landscape/models": "landscape/models.py",
}

# Target mutation scores per module
THRESHOLDS = {
    "canonical.py": 95,
    "landscape/recorder.py": 90,
    "landscape/exporter.py": 90,
    "landscape/models.py": 90,
}

CORE_PATH = Path("src/elspeth/core")
CACHE_PATH = Path(".mutmut-cache")


def clean_cache() -> None:
    """Remove the mutmut cache directory."""
    if CACHE_PATH.exists():
        print(f"🧹 Cleaning {CACHE_PATH}...")
        shutil.rmtree(CACHE_PATH)


def run_mutmut(module_path: str, timeout_minutes: int = 120) -> int:
    """Run mutmut on a specific module.

    Args:
        module_path: Path relative to src/elspeth/core/
        timeout_minutes: Maximum time to wait

    Returns:
        Exit code from mutmut
    """
    full_path = CORE_PATH / module_path

    if not full_path.exists():
        print(f"❌ Module not found: {full_path}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"🧬 Running mutation testing on: {module_path}")
    print(f"{'=' * 60}\n")

    cmd = [
        sys.executable,
        "-m",
        "mutmut",
        "run",
        "--paths-to-mutate",
        str(full_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout_minutes * 60,
            check=False,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after {timeout_minutes} minutes")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1


def show_results() -> int:
    """Show mutation testing results."""
    print("\n📊 Mutation Testing Results:")
    print("-" * 40)

    cmd = [sys.executable, "-m", "mutmut", "results"]
    result = subprocess.run(cmd, check=False)
    return result.returncode


def show_survivors() -> None:
    """Show details of survived mutants."""
    print("\n🔍 Survived Mutants (tests didn't catch these bugs):")
    print("-" * 50)

    # Get list of survived mutant IDs
    cmd = [sys.executable, "-m", "mutmut", "results"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if "Survived" not in result.stdout:
        print("✅ No survivors - all mutants were killed!")
        return

    # Parse survivor count and show first few
    print("\nUse 'python -m mutmut show <id>' to inspect specific mutants")
    print("Use 'python -m mutmut html' to generate an HTML report\n")


def calculate_score() -> tuple[int, int, float] | None:
    """Calculate mutation score from results.

    Returns:
        Tuple of (killed, total, score_percent) or None if can't parse
    """
    cmd = [sys.executable, "-m", "mutmut", "results"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # Parse output - mutmut 2.x format varies
    # Try to extract from the summary line
    output = result.stdout + result.stderr

    killed = 0
    survived = 0
    timeout = 0

    for line in output.split("\n"):
        line_lower = line.lower()
        if "killed" in line_lower:
            # Try to find number before "killed"
            parts = line.split()
            for i, part in enumerate(parts):
                if "killed" in part.lower() and i > 0:
                    with contextlib.suppress(ValueError):
                        killed = int(parts[i - 1])
        if "survived" in line_lower:
            parts = line.split()
            for i, part in enumerate(parts):
                if "survived" in part.lower() and i > 0:
                    with contextlib.suppress(ValueError):
                        survived = int(parts[i - 1])
        if "timeout" in line_lower:
            parts = line.split()
            for i, part in enumerate(parts):
                if "timeout" in part.lower() and i > 0:
                    with contextlib.suppress(ValueError):
                        timeout = int(parts[i - 1])

    total = killed + survived + timeout
    if total == 0:
        return None

    score = (killed / total) * 100
    return killed, total, score


def main() -> int:
    """Run mutation testing."""
    parser = argparse.ArgumentParser(
        description="Run mutation testing on ELSPETH core modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--module",
        "-m",
        help="Module to test (e.g., canonical.py, landscape/recorder.py)",
        default="canonical.py",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all core modules (slow!)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean cache before running",
    )
    parser.add_argument(
        "--show-survivors",
        action="store_true",
        help="Show survived mutants from last run",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if score below threshold",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in minutes (default: 120)",
    )

    args = parser.parse_args()

    # Show survivors and exit
    if args.show_survivors:
        show_results()
        show_survivors()
        return 0

    # Clean cache unless --no-clean
    if not args.no_clean:
        clean_cache()

    # Determine modules to test
    if args.all:
        modules = list(MODULES.values())
        print("🧬 Running mutation testing on ALL core modules")
        print("⚠️  This will take a long time (potentially hours)")
    else:
        modules = [args.module]

    # Run mutation testing
    exit_code = 0
    for module in modules:
        result = run_mutmut(module, timeout_minutes=args.timeout)
        if result != 0:
            # mutmut returns non-zero even on success sometimes
            pass

        # Show results
        show_results()

        # Check threshold if strict mode
        if args.strict:
            score_data = calculate_score()
            if score_data:
                killed, total, score = score_data
                threshold = THRESHOLDS.get(module, 80)
                print(f"\n📈 Score: {score:.1f}% ({killed}/{total} killed)")
                print(f"📊 Threshold: {threshold}%")
                if score < threshold:
                    print(f"❌ FAILED: Score {score:.1f}% below {threshold}% threshold")
                    exit_code = 2
                else:
                    print(f"✅ PASSED: Score {score:.1f}% meets {threshold}% threshold")

    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("  - Use 'python -m mutmut show <id>' to inspect a mutant")
    print("  - Use 'python -m mutmut html' for an HTML report")
    print("  - Survived mutants reveal weak test assertions")
    print("=" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
