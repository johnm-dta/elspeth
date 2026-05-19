"""Built-in elspeth-lints rules."""

from __future__ import annotations

from elspeth_lints.rules.audit_evidence import (
    AUDIT_EVIDENCE_NOMINAL_RULE,
    GUARD_SYMMETRY_RULE,
    GVE_ATTRIBUTION_RULE,
    TIER_1_DECORATION_RULE,
)
from elspeth_lints.rules.immutability import FREEZE_GUARDS_RULE, FROZEN_ANNOTATIONS_RULE
from elspeth_lints.rules.meta_no_new_bespoke_cicd_enforcer import RULE as META_NO_NEW_BESPOKE_CICD_ENFORCER_RULE
from elspeth_lints.rules.plugin_contract.options_metadata import RULE as OPTIONS_METADATA_RULE

BUILTIN_RULES = (
    META_NO_NEW_BESPOKE_CICD_ENFORCER_RULE,
    OPTIONS_METADATA_RULE,
    FREEZE_GUARDS_RULE,
    FROZEN_ANNOTATIONS_RULE,
    AUDIT_EVIDENCE_NOMINAL_RULE,
    TIER_1_DECORATION_RULE,
    GUARD_SYMMETRY_RULE,
    GVE_ATTRIBUTION_RULE,
)

__all__ = [
    "AUDIT_EVIDENCE_NOMINAL_RULE",
    "BUILTIN_RULES",
    "FREEZE_GUARDS_RULE",
    "FROZEN_ANNOTATIONS_RULE",
    "GUARD_SYMMETRY_RULE",
    "GVE_ATTRIBUTION_RULE",
    "META_NO_NEW_BESPOKE_CICD_ENFORCER_RULE",
    "OPTIONS_METADATA_RULE",
    "TIER_1_DECORATION_RULE",
]
