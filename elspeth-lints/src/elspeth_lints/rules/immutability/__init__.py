"""Immutability lint rules."""

from __future__ import annotations

from elspeth_lints.rules.immutability.freeze_guards import RULE as FREEZE_GUARDS_RULE
from elspeth_lints.rules.immutability.frozen_annotations import RULE as FROZEN_ANNOTATIONS_RULE

__all__ = ["FREEZE_GUARDS_RULE", "FROZEN_ANNOTATIONS_RULE"]
