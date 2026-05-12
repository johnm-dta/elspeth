"""Adequacy guard — assertion 1 (registry-manifest set equality, spec §4.4.1).

Closes plan-review B1 (the manifest enforces consistency with the dispatch
registry; a tool added to a dispatch dict without a manifest entry MUST
fail this test in CI).
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.redaction import MANIFEST

from ._adequacy_helpers import collect_registry_names


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Manifest is populated by Tasks 13-16; this assertion is "
        "intentionally red until then. Remove this marker when the manifest "
        "reaches parity with collect_registry_names() — Task 16's final "
        "completion gate. If pytest reports XPASS, the marker must come off "
        "(the suite is now correctly green)."
    ),
)
def test_manifest_keys_equal_registry_names() -> None:
    """Set-equality contract: MANIFEST.keys() == collect_registry_names()."""
    registry = collect_registry_names()
    manifest_keys = frozenset(MANIFEST.keys())
    assert manifest_keys == registry, (
        f"Manifest/registry parity broken. "
        f"Missing from manifest: {sorted(registry - manifest_keys)!r}. "
        f"Extra in manifest (not in registry): {sorted(manifest_keys - registry)!r}."
    )
