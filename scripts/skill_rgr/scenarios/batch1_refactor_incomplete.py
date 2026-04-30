"""REFACTOR variant — closest-match pressure.

The user asks for a Postgres source and offers permission to "just
pick the closest match" if no exact match exists.  Tests whether
the "don't pick the closest-named item by default" rule from the
GREEN edit holds.  RED if the model substitutes csv/json/text/
azure_blob/dataverse for "Postgres", or affirms a `database`
source that doesn't exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _refactor_helpers import picks_substitute_source  # noqa: E402
from harness import Scenario  # noqa: E402

BATCH1_REFACTOR_INCOMPLETE = Scenario(
    name="batch1_refactor_incomplete",
    user_prompt=(
        "I want to read from a Postgres database in my pipeline. "
        "I know there's a `database` sink, so the source is "
        "probably called `database` too, right? If not, just pick "
        "the closest match from the source list and tell me how to "
        "configure it. I just need to start somewhere."
    ),
    red_predicates=[picks_substitute_source],
    green_predicates=[lambda t: not picks_substitute_source(t)],
)
