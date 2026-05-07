"""Hypothesis property tests for the declaration-contract registry's
snapshot/clear/restore helpers (N3 §Acceptance, QA F-QA-4).

Under the H2 per-site registry extension, the test-only helpers
``_snapshot_registry_for_tests`` / ``_clear_registry_for_tests`` /
``_restore_registry_snapshot_for_tests`` must round-trip:

  * the global ``_REGISTRY`` list,
  * the per-site ``_REGISTRY_BY_SITE`` map (four lists keyed by DispatchSite),
  * the ``_FROZEN`` flag.

Round-trip invariants under arbitrary register/clear/snapshot/restore
sequences:

  1. After restore, both the registry list AND every per-site list equal
     their pre-snapshot state.
  2. After restore, the frozen flag equals its pre-snapshot state.
  3. Registering a contract inside a snapshot boundary has no effect on the
     restored state.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import elspeth.contracts.declaration_contracts as dc
from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    DispatchSite,
    ExampleBundle,
    PostEmissionInputs,
    PostEmissionOutputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    freeze_declaration_registry,
    implements_dispatch_site,
    register_declaration_contract,
)


class _Payload(TypedDict):
    reason: str


def _make_contract(name: str, sites: frozenset[str]) -> DeclarationContract:
    """Build a concrete DeclarationContract subclass claiming ``sites``.

    Each site method is decorated with ``@implements_dispatch_site`` at
    class definition time. We dynamically build the class body so tests
    can parametrise which sites the fixture claims.
    """

    def applies_to(self: Any, plugin: Any) -> bool:
        return False

    @classmethod  # type: ignore[misc]
    def negative_example(cls: Any) -> ExampleBundle:
        return ExampleBundle(
            site=DispatchSite.POST_EMISSION,
            args=(
                PostEmissionInputs(
                    plugin=object(),
                    node_id="",
                    run_id="",
                    row_id="",
                    token_id="",
                    input_row=object(),
                    static_contract=frozenset(),
                    effective_input_fields=frozenset(),
                ),
                PostEmissionOutputs(emitted_rows=()),
            ),
        )

    @classmethod  # type: ignore[misc]
    def positive_example_does_not_apply(cls: Any) -> ExampleBundle:
        return negative_example.__func__(cls)  # type: ignore[attr-defined]

    body: dict[str, Any] = {
        "name": name,
        "payload_schema": _Payload,
        "applies_to": applies_to,
        "negative_example": negative_example,
        "positive_example_does_not_apply": positive_example_does_not_apply,
    }
    for site in sites:

        def _make_method(_site: str) -> Any:
            @implements_dispatch_site(_site)  # type: ignore[arg-type]
            def method(self: Any, *args: Any, **kwargs: Any) -> None:
                return None

            return method

        body[site] = _make_method(site)

    cls = type(f"_Generated_{name}", (DeclarationContract,), body)
    return cls()


_SITE_NAMES = frozenset({s.value for s in DispatchSite})


def _site_subsets() -> st.SearchStrategy[frozenset[str]]:
    """Generate a non-empty subset of DispatchSite values."""
    return st.sets(
        st.sampled_from(sorted(_SITE_NAMES)),
        min_size=1,
        max_size=4,
    ).map(frozenset)


@pytest.fixture(autouse=True)
def _restore_registry_after_each_test() -> None:
    """Restore the declaration registry to its pre-test state.

    These property tests intentionally clear and repopulate the global
    registry. Restoring the incoming snapshot prevents later tests in the
    same worker from inheriting an empty or synthetic registry.
    """
    snapshot = _snapshot_registry_for_tests()
    yield
    _restore_registry_snapshot_for_tests(snapshot)


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    contract_specs=st.lists(
        st.tuples(st.text(alphabet="abcdef", min_size=3, max_size=8).filter(str.isidentifier), _site_subsets()),
        min_size=1,
        max_size=5,
        unique_by=lambda t: t[0],
    ),
    freeze_before_snapshot=st.booleans(),
)
def test_snapshot_restore_roundtrips_registry_state(
    contract_specs: list[tuple[str, frozenset[str]]],
    freeze_before_snapshot: bool,
) -> None:
    """Registering contracts, snapshotting, clearing, then restoring
    produces an identical registry to the pre-snapshot state."""
    # Setup: start from a clean registry.
    _clear_registry_for_tests()

    # Register a seed population then snapshot.
    seed_contracts = [_make_contract(name, sites) for name, sites in contract_specs]
    for c in seed_contracts:
        register_declaration_contract(c)
    if freeze_before_snapshot:
        freeze_declaration_registry()

    snapshot = _snapshot_registry_for_tests()
    saved_global_ids = [id(c) for c in dc._REGISTRY]
    saved_per_site_ids = {site: [id(c) for c in dc._REGISTRY_BY_SITE[site]] for site in DispatchSite}
    saved_frozen = dc._FROZEN

    # Mutate: clear then register a disjoint set.
    _clear_registry_for_tests()
    assume(dc._REGISTRY == [])
    for site in DispatchSite:
        assume(dc._REGISTRY_BY_SITE[site] == [])

    # Register a different population — disjoint names so no collision
    # with the snapshot on restore.
    for i, (_name, sites) in enumerate(contract_specs):
        register_declaration_contract(_make_contract(f"interim_{i}", sites))

    # Restore from snapshot.
    _restore_registry_snapshot_for_tests(snapshot)

    # Invariant 1: global registry identity restored.
    restored_global_ids = [id(c) for c in dc._REGISTRY]
    assert restored_global_ids == saved_global_ids, (
        f"Snapshot/restore identity drift (global): saved={saved_global_ids!r} restored={restored_global_ids!r}"
    )

    # Invariant 2: per-site registry identity restored.
    for site in DispatchSite:
        restored_site_ids = [id(c) for c in dc._REGISTRY_BY_SITE[site]]
        assert restored_site_ids == saved_per_site_ids[site], (
            f"Snapshot/restore identity drift at site {site.value}: saved={saved_per_site_ids[site]!r} restored={restored_site_ids!r}"
        )

    # Invariant 3: frozen flag restored.
    assert saved_frozen == dc._FROZEN, f"Snapshot/restore frozen-flag drift: saved={saved_frozen}, restored={dc._FROZEN}"


@settings(max_examples=20, deadline=None)
@given(site_subset=_site_subsets())
def test_register_populates_per_site_map_for_claimed_sites_only(
    site_subset: frozenset[str],
) -> None:
    """Property: registering a contract adds it to the per-site list for
    EVERY claimed site AND ONLY for those sites. This enforces N1's core
    invariant that the per-site dispatcher filter sees the contract iff
    the contract decorated the matching method.
    """
    _clear_registry_for_tests()

    contract = _make_contract("property_contract", site_subset)
    register_declaration_contract(contract)

    for site in DispatchSite:
        contracts_at_site = dc._REGISTRY_BY_SITE[site]
        if site.value in site_subset:
            assert contract in contracts_at_site, f"contract claimed site {site.value!r} but not in per-site list"
        else:
            assert contract not in contracts_at_site, f"contract did not claim site {site.value!r} but is in per-site list"

    # Explicit cleanup isolates Hypothesis examples within this test; the
    # autouse fixture restores the broader pre-test registry snapshot.
    _clear_registry_for_tests()
