"""Sanity bound on adequacy-guard runtime (closes plan-review M4).

The four-assertion adequacy guard must execute in well under 5 seconds for
the current 38-entry MANIFEST (37 registry tools plus
``request_advisor_hint``).  This is an order-of-magnitude bound, not a tight
budget — flake-source guidance per spec §1.4.

The sanity bound exists to detect O(n^2) regressions in any of the four
assertions as the manifest grows.  A guard that grows superlinearly with the
manifest size will eventually become a CI tax even if every individual
assertion still passes; an explicit budget pins the cost at order-of-magnitude
so the next reviewer notices.
"""

from __future__ import annotations

import time

from elspeth.web.composer.redaction import MANIFEST

from ._adequacy_helpers import compute_manifest_snapshot


def test_adequacy_guard_completes_in_under_5_seconds() -> None:
    """Closes plan-review M4: compute_manifest_snapshot finishes under 5s.

    ``compute_manifest_snapshot`` is the most expensive of the adequacy-guard
    helpers (it walks every type-driven argument model and hashes every node)
    and serves as a representative proxy for the guard's overall cost.  The
    5-second bound is an order-of-magnitude ceiling; a passing run typically
    completes in well under one second.
    """
    start = time.perf_counter()
    compute_manifest_snapshot(MANIFEST)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, (
        f"Adequacy guard took {elapsed:.2f}s for {len(MANIFEST)} entries; "
        "exceeded 5-second sanity bound (spec §1.4, plan-review M4). "
        "Investigate whether a recent change introduced superlinear cost "
        "before raising the bound."
    )
