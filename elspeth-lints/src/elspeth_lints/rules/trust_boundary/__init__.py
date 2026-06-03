"""Trust-boundary honesty-gate lint rules.

These rules audit the in-source claims made by ``@trust_boundary`` decorators
(defined in ``src/elspeth/contracts/trust_boundary.py``). They are honesty
gates: they verify that the decorator's metadata corresponds to facts about
the surrounding code, not vibes-justifications.

Three independent rules:

* :mod:`elspeth_lints.rules.trust_boundary.tests` — ``trust_boundary.tests``.
  Every decorator must carry a ``test_ref`` pointing to a real pytest node
  whose body asserts on malformed input (``pytest.raises`` /
  ``with pytest.raises`` / ``assertRaises``).
* :mod:`elspeth_lints.rules.trust_boundary.scope` — ``trust_boundary.scope``.
  ``source_param`` must name a parameter of the decorated function and the
  body must actually read from it.
* :mod:`elspeth_lints.rules.trust_boundary.tier` — ``trust_boundary.tier``.
  Only ``tier=3`` is accepted — Tier-1 and Tier-2 invariants must crash, not
  suppress.

Rule package layout mirrors the immutability/audit_evidence siblings:
``metadata.py`` declares the ``RuleMetadata`` literal; ``rule.py`` implements
the visitor and exports a ``RULE`` instance for the central registry.
"""

from __future__ import annotations

from elspeth_lints.rules.trust_boundary.scope import RULE as TRUST_BOUNDARY_SCOPE_RULE
from elspeth_lints.rules.trust_boundary.tests import RULE as TRUST_BOUNDARY_TESTS_RULE
from elspeth_lints.rules.trust_boundary.tier import RULE as TRUST_BOUNDARY_TIER_RULE

__all__ = [
    "TRUST_BOUNDARY_SCOPE_RULE",
    "TRUST_BOUNDARY_TESTS_RULE",
    "TRUST_BOUNDARY_TIER_RULE",
]
