"""Run a scenario through the RGR harness.

Usage:
    python -m scripts.skill_rgr.run calibration --model gpt-5.5 --label red
    python -m scripts.skill_rgr.run batch1 --model gpt-5.5 --label red
    python -m scripts.skill_rgr.run batch1 --skill /tmp/edited_skill.md --label green
    python -m scripts.skill_rgr.run batch1 --model claude-opus-4-7 --label green-claude

Environment:
    OPENAI_API_KEY     — required for gpt-* models
    ANTHROPIC_API_KEY  — required for claude-* models
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

_HARNESS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_HARNESS_ROOT))
# Also make sibling scenario modules importable from each other
# (e.g. batch1_refactor_insist imports batch1_pressured).
sys.path.insert(0, str(_HARNESS_ROOT / "scenarios"))


def _load_scenario(name: str) -> Any:
    """Load the scenario object by name from ``scenarios/<name>.py``.

    The scenario module is expected to expose a single uppercase
    constant matching the module name (e.g. ``CALIBRATION`` in
    ``scenarios/calibration.py``).
    """

    module = importlib.import_module(f"scenarios.{name}")
    expected = name.upper()
    if not hasattr(module, expected):
        # Fall back to the first Scenario instance found.
        from harness import Scenario  # type: ignore[import-not-found]

        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, Scenario):
                return obj
        raise AttributeError(f"No Scenario found in scenarios.{name}")
    return getattr(module, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="Scenario module name under scenarios/")
    parser.add_argument(
        "--model",
        default="openrouter/openai/gpt-5",
        help=(
            "litellm model identifier. OpenRouter routes use the "
            "openrouter/<vendor>/<model> form (e.g. "
            "openrouter/openai/gpt-5, openrouter/anthropic/claude-opus-4)."
        ),
    )
    parser.add_argument(
        "--skill",
        default=None,
        help="Path to an alternate skill markdown file (for GREEN runs)",
    )
    parser.add_argument(
        "--label",
        default="run",
        help="Filename label for the transcript",
    )
    parser.add_argument(
        "--phase",
        choices=["red", "green", "none"],
        default="none",
        help="Predicate phase to evaluate after the run",
    )
    args = parser.parse_args()

    from harness import evaluate, load_skill, run_scenario

    scenario = _load_scenario(args.scenario)
    skill_text = load_skill(override_path=Path(args.skill)) if args.skill else load_skill()

    transcript = run_scenario(scenario, skill_text=skill_text, model=args.model, label=args.label)

    print(f"Scenario: {scenario.name}")
    print(f"Model:    {args.model}")
    print(f"Skill:    {args.skill or '<production>'}")
    print(f"Turns:    {sum(1 for e in transcript if e.get('role') == 'assistant')}")
    print(f"Tools:    {sum(1 for e in transcript if e.get('role') == 'tool')}")

    if args.phase != "none":
        results = evaluate(transcript, scenario, phase=args.phase)
        all_pass = all(results.values())
        print(f"\n{args.phase.upper()} predicate results:")
        print(json.dumps(results, indent=2))
        print(f"\n{args.phase.upper()} {'CONFIRMED' if all_pass else 'NOT confirmed'}")
        return 0 if all_pass else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
