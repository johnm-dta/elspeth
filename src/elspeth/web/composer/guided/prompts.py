"""Guided-mode skill loading + Step 3 context-block construction.

Module-cached via @lru_cache; per project memory, restart elspeth-web.service
after editing the skill markdown for live changes to take effect.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SKILL_PATH = Path(__file__).parent / "skills" / "guided_pipeline.md"


@lru_cache(maxsize=1)
def load_guided_skill() -> str:
    """Load the guided-mode skill prompt. Cached per process; restart on edit."""
    return _SKILL_PATH.read_text(encoding="utf-8")
