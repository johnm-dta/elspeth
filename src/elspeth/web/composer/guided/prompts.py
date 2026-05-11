"""Guided-mode skill loading + Step 3 context-block construction.

Module-cached via @lru_cache; per project memory, restart elspeth-web.service
after editing the skill markdown for live changes to take effect.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.web.composer.guided.recipe_match import RecipeMatch
    from elspeth.web.composer.guided.state_machine import (
        SinkResolved,
        SourceResolved,
    )

_SKILL_PATH = Path(__file__).parent / "skills" / "guided_pipeline.md"


@lru_cache(maxsize=1)
def load_guided_skill() -> str:
    """Load the guided-mode skill prompt. Cached per process; restart on edit."""
    return _SKILL_PATH.read_text(encoding="utf-8")


def build_step_3_context_block(
    *,
    source: SourceResolved,
    sink: SinkResolved,
    recipe_match: RecipeMatch | None,
) -> str:
    """Render the GUIDED CONTEXT block for the Step 3 LLM prompt."""
    src_payload = {
        "plugin": source.plugin,
        "columns": list(source.observed_columns),
        "sample": [dict(r) for r in source.sample_rows[:3]],
    }
    sink_payload = {
        "outputs": [
            {
                "plugin": o.plugin,
                "required_fields": list(o.required_fields),
                "schema_mode": o.schema_mode,
            }
            for o in sink.outputs
        ],
    }
    match_repr = "null" if recipe_match is None else json.dumps({"recipe_name": recipe_match.recipe_name})
    return (
        "GUIDED CONTEXT (server-resolved):\n"
        f"source: {json.dumps(src_payload)}\n"
        f"sink: {json.dumps(sink_payload)}\n"
        f"recipe_match: {match_repr}\n"
    )
