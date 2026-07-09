"""Framework bootstrap: registry assertion + freeze before any DAG execution.

Extracted from ``orchestrator/core.py`` (filigree elspeth-9e71ae82a4) so the
run-lifecycle coordinator can import it without a circular import through the
``core`` facade. ``core.py`` re-exports :func:`prepare_for_run` — the public
import path ``elspeth.engine.orchestrator.prepare_for_run`` (and the legacy
``…orchestrator.core.prepare_for_run``) is unchanged.

This module is a LEAF within the orchestrator package: it must not import
from other orchestrator submodules.
"""

from __future__ import annotations

# Module-level side-effect import: registers every production declaration
# contract BEFORE prepare_for_run() asserts the registry against the manifest.
# Lives here (next to prepare_for_run) so the bootstrap contract holds for
# every importer, not only for callers that happened to import core.py first.
import elspeth.engine.executors.declaration_contract_bootstrap  # noqa: F401
from elspeth.contracts.declaration_contracts import (
    EXPECTED_CONTRACT_SITES,
    contract_sites,
    declaration_registry_is_frozen,
    freeze_declaration_registry,
    registered_declaration_contracts,
)
from elspeth.contracts.tier_registry import freeze_tier_registry


def prepare_for_run() -> None:
    """Assert framework invariants and freeze both registries at bootstrap.

    This is the canonical bootstrap entry point (ADR-010 §Decision 3). It must
    be called AFTER all plugin modules have been imported (so module-level
    side-effects like ``register_declaration_contract`` have fired) and BEFORE
    any DAG node begins execution.

    The normal import chain guarantees this ordering in production:
    - this module (and ``orchestrator/core.py``) imports
      ``engine.executors.declaration_contract_bootstrap`` at module level.
    - ``declaration_contract_bootstrap.py`` imports every production contract
      module with a module-level ``register_declaration_contract(...)`` call.
    - Each imported contract module registers its contract as a module-level
      side-effect.

    If the declaration registry is empty at this point, the import chain is
    broken — this is an import-order bug, not a runtime configuration error.
    Crashing here prevents the framework from running silently without any
    runtime VAL checks active (the exact failure mode ADR-010 was designed to
    prevent).

    Raises:
        RuntimeError: every registered ``(contract_name, dispatch_site)``
            pair does not exactly equal the pairs in
            ``EXPECTED_CONTRACT_SITES`` (N1 per-site manifest extension,
            ADR-010 §Semantics amendment 2026-04-20). The message names
            every missing and every extra contract or site so the failure
            is self-diagnosing. Indicates one of: an import-order bug
            (contract module not imported), manifest drift (contract
            registered without a manifest entry), or site drift
            (contract's ``@implements_dispatch_site`` markers disagree
            with the manifest).
    """
    # Short-circuit if the registry is already frozen — bootstrap already ran.
    # Idempotency is required because Orchestrator.run() can be called multiple
    # times in a single process (e.g. test suites). The manifest-equality
    # assertion only needs to fire ONCE, on the first call; subsequent calls
    # trust that the previous freeze was performed after a successful
    # assertion.
    #
    # The ``_clear_registry_for_tests()`` helper resets ``_FROZEN = False``, so
    # test isolation that clears and repopulates the registry will still trigger
    # the assertion on the next call.
    if declaration_registry_is_frozen():
        return

    # ADR-010 §Decision 3 manifest gate, extended by §H2 landing scope N1:
    # Assert SET EQUALITY between every registered (contract_name,
    # dispatch_site) pair and every pair in ``EXPECTED_CONTRACT_SITES``
    # BEFORE freezing. The original C2 closure checked contract-name
    # equality; N1 tightens this to per-(name, site) equality so a
    # contract registering for the wrong site (or silently no-opping at a
    # site it claims to cover) is detected at bootstrap, not masked until
    # first row.
    #
    # Every plugin behaviour recorded as "compliant" (no violation raised)
    # must be evidence of every applicable contract's method having been
    # invoked — under audit-complete semantics (ADR-010 §Semantics) absence of violation
    # means "checked and passed," which is only true if the dispatcher
    # actually dispatched to the contract for its claimed sites. The N1
    # manifest closes the (name, site) drift vector the C2 set-of-names
    # manifest missed.
    contracts = registered_declaration_contracts()
    registered_sites: dict[str, frozenset[str]] = {c.name: frozenset(contract_sites(c)) for c in contracts}
    expected_sites: dict[str, frozenset[str]] = {name: frozenset(sites) for name, sites in EXPECTED_CONTRACT_SITES.items()}
    if registered_sites != expected_sites:
        # Compose a self-diagnosing message naming every drifted (name, site)
        # pair. Five mutually exclusive drift classes are surfaced:
        #   * name missing (manifest claims, nothing registered)
        #   * name extra (registered, manifest absent)
        #   * per-name: sites missing (contract registered with fewer sites
        #     than manifest declares)
        #   * per-name: sites extra (contract registered with more sites
        #     than manifest declares)
        #   * per-name: site-set mismatch (disjoint)
        expected_names = frozenset(expected_sites.keys())
        registered_names = frozenset(registered_sites.keys())
        missing_names = expected_names - registered_names
        extra_names = registered_names - expected_names
        site_drift_lines: list[str] = []
        for name in sorted(expected_names & registered_names):
            expected_for_name = expected_sites[name]
            registered_for_name = registered_sites[name]
            if expected_for_name == registered_for_name:
                continue
            missing_sites = expected_for_name - registered_for_name
            extra_sites = registered_for_name - expected_for_name
            site_drift_lines.append(
                f"  {name!r}: expected_sites={sorted(expected_for_name)!r}, "
                f"registered_sites={sorted(registered_for_name)!r}, "
                f"missing={sorted(missing_sites)!r}, extra={sorted(extra_sites)!r}"
            )
        raise RuntimeError(
            "Declaration contract registry mismatch at orchestrator bootstrap "
            "(ADR-010 §Decision 3 manifest gate + §H2 landing scope N1).\n"
            f"  Expected (manifest):  {sorted((n, sorted(s)) for n, s in expected_sites.items())!r}\n"
            f"  Registered:           {sorted((n, sorted(s)) for n, s in registered_sites.items())!r}\n"
            f"  Missing names (not registered but in manifest): {sorted(missing_names)!r}\n"
            f"  Extra names  (registered but not in manifest): {sorted(extra_names)!r}\n"
            "  Per-name site drift:\n" + ("\n".join(site_drift_lines) if site_drift_lines else "    (none)") + "\n"
            "\n"
            "If a name is missing: the contract's module-level "
            "register_declaration_contract(...) call did not fire. Check for "
            "a conditional import that skipped it, or an import-order bug "
            "where the module was not imported before prepare_for_run().\n"
            "If a name is extra: a contract was registered without being "
            "added to EXPECTED_CONTRACT_SITES. Update the manifest in the "
            "same commit as the registration.\n"
            "If per-name sites drift: the contract's "
            "@implements_dispatch_site(...) markers disagree with "
            "EXPECTED_CONTRACT_SITES. Either fix the markers or update the "
            "manifest (and run `elspeth-lints check --rules "
            "manifest.contract_manifest --root src/elspeth` to confirm "
            "MC3a/b/c are clean).\n"
            "\n"
            "A silent runtime VAL disable is exactly the failure mode ADR-010 "
            "was designed to prevent — extended to per-site coverage under "
            "the §Semantics amendment (2026-04-20)."
        )
    freeze_declaration_registry()
    freeze_tier_registry()
