"""Score a captured composer-rgr run against scenario criteria.

Usage:
    score.py scenario.json messages.json state.json    # prints scoring JSON

Logic lives in `evals.lib.composer_rgr_score.score(...)`. This shim keeps
the historical CLI surface intact (run_scenario.sh invokes it positionally)
while the pure function is unit-testable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# evals/composer-rgr is not a Python package (hyphen in dir name), so the
# shim is invoked as a standalone script. Add the repo root to sys.path so
# `evals.lib.composer_rgr_score` resolves regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.lib.composer_rgr_score import score  # noqa: E402  -- after sys.path rewire


def main() -> int:
    scenario = json.loads(Path(sys.argv[1]).read_text())
    messages = json.loads(Path(sys.argv[2]).read_text())
    state = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None
    result = score(scenario, messages, state)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
