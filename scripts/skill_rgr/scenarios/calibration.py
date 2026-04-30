"""Calibration scenario — proves the harness mechanics before we trust RED detection.

A known-good prompt that every reasonable LLM should handle the same
way.  If gpt-5.5 and Claude diverge here, the divergence is harness
artefact (tools-API quirk, schema conversion bug, message-format
mismatch) rather than a skill problem.

Specifically: ask the LLM to do the simplest possible composition step
— look up which sources are available — and confirm it calls
``list_sources``.  No tricks, no pressure, no contradictions in the
skill text that would make the answer ambiguous.

If this scenario fails to produce ``list_sources`` against either
model, fix the harness before running real RED scenarios.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import Scenario, called_tool


CALIBRATION = Scenario(
    name="calibration",
    user_prompt=(
        "What sources can I use in an ELSPETH pipeline? "
        "Don't build anything yet — just list what's available."
    ),
    red_predicates=[
        # No RED in calibration: this scenario is pass/fail, not before/after.
    ],
    green_predicates=[
        lambda t: called_tool(t, "list_sources"),
    ],
)
