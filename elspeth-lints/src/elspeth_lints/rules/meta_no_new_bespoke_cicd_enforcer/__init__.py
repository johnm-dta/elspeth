"""Meta rule preventing new bespoke scripts/cicd/enforce_*.py gates."""

from __future__ import annotations

from elspeth_lints.rules.meta_no_new_bespoke_cicd_enforcer.rule import RULE, NoNewBespokeCicdEnforcerRule

__all__ = ["RULE", "NoNewBespokeCicdEnforcerRule"]
