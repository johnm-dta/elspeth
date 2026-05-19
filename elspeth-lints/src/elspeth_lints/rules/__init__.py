"""Built-in elspeth-lints rules."""

from __future__ import annotations

from elspeth_lints.rules.meta_no_new_bespoke_cicd_enforcer import RULE as META_NO_NEW_BESPOKE_CICD_ENFORCER_RULE
from elspeth_lints.rules.plugin_contract.options_metadata import RULE as OPTIONS_METADATA_RULE

BUILTIN_RULES = (META_NO_NEW_BESPOKE_CICD_ENFORCER_RULE, OPTIONS_METADATA_RULE)

__all__ = ["BUILTIN_RULES", "META_NO_NEW_BESPOKE_CICD_ENFORCER_RULE", "OPTIONS_METADATA_RULE"]
