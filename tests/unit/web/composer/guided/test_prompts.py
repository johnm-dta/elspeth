"""Tests for guided skill loading + the staged-skill cache hash."""

from __future__ import annotations

import hashlib

from elspeth.web.composer.guided.prompts import (
    _SKILLS_DIR,
    _STEP_FILE_NAMES,
    _STEP_PLAYBOOK_ORDER,
    guided_staged_skill_hash,
)


def test_guided_staged_skill_hash_covers_base_and_every_step_in_order() -> None:
    """The hash folds base.md + each step file in playbook order.

    Tracking every member of _STEP_PLAYBOOK_ORDER means step_4_wire.md is
    keyed the moment STEP_4_WIRE is appended — no second edit to the cache
    path is needed when a stage is added.
    """
    h = hashlib.sha256()
    h.update((_SKILLS_DIR / "base.md").read_bytes())
    for step in _STEP_PLAYBOOK_ORDER:
        h.update((_SKILLS_DIR / _STEP_FILE_NAMES[step]).read_bytes())
    assert guided_staged_skill_hash() == h.hexdigest()


def test_guided_staged_skill_hash_is_deterministic() -> None:
    assert guided_staged_skill_hash() == guided_staged_skill_hash()
