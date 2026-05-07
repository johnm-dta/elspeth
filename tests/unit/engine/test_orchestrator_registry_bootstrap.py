"""Orchestrator bootstrap freezes the declaration + tier registries and
asserts at least one declaration contract is registered (ADR-010 §Decision 3).

Test isolation strategy:
- Each test saves and restores the full state of both registries
  (contents + freeze flag). This prevents test pollution cascading into
  subsequent tests that rely on contracts being registered.
- The isolation fixture is autouse=False because these tests intentionally
  manipulate registry state; they must not auto-apply to unrelated tests in
  the same module.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Module-level import so ``_isolate_both_registries`` snapshots a fully-
# populated registry even when this file is the *only* one pytest collects
# (``pytest test_orchestrator_registry_bootstrap.py::test_x`` in isolation).
# The authoritative bootstrap surface lives in declaration_contract_bootstrap.py;
# reloading that module replays every production contract registration in one
# place.
import elspeth.engine.executors.declaration_contract_bootstrap  # noqa: F401


@pytest.fixture()
def _isolate_both_registries():
    """Save + restore declaration_contracts and tier_registry module state.

    Saves:
    - declaration_contracts._REGISTRY contents
    - declaration_contracts._FROZEN flag
    - tier_registry._REGISTRY contents
    - tier_registry._REASONS contents
    - tier_registry._FROZEN flag

    Restores all on teardown. This covers both the content and freeze state
    so tests that call prepare_for_run() (which freezes both) do not leave
    subsequent tests with frozen or empty registries.
    """
    import elspeth.contracts.declaration_contracts as dc
    import elspeth.contracts.tier_registry as tr

    # Snapshot current state (H2: per-site map included).
    saved_dc_registry = list(dc._REGISTRY)
    saved_dc_per_site = {site: list(lst) for site, lst in dc._REGISTRY_BY_SITE.items()}
    saved_dc_frozen = dc._FROZEN
    saved_tr_registry = list(tr._REGISTRY)
    saved_tr_reasons = dict(tr._REASONS)
    saved_tr_frozen = tr._FROZEN

    yield

    # Restore declaration_contracts state
    dc._REGISTRY.clear()
    dc._REGISTRY.extend(saved_dc_registry)
    for site, lst in saved_dc_per_site.items():
        dc._REGISTRY_BY_SITE[site][:] = lst
    dc._FROZEN = saved_dc_frozen

    # Restore tier_registry state
    tr._REGISTRY.clear()
    tr._REGISTRY.extend(saved_tr_registry)
    tr._REASONS.clear()
    tr._REASONS.update(saved_tr_reasons)
    tr._FROZEN = saved_tr_frozen


def _replay_bootstrap_import_surface() -> None:
    """Reload every executor module imported by the bootstrap surface.

    ``importlib.reload(declaration_contract_bootstrap)`` alone is insufficient
    after ``_clear_registry_for_tests()`` because Python caches the modules
    imported *by* the bootstrap file. Parsing the bootstrap module's direct
    imports keeps the test aligned with the authoritative production import
    surface without hard-coding ``pass_through.py`` here.
    """
    import elspeth.engine.executors.declaration_contract_bootstrap as dc_bootstrap

    bootstrap_path = Path(dc_bootstrap.__file__)
    tree = ast.parse(bootstrap_path.read_text(), filename=str(bootstrap_path))
    prefix = "elspeth.engine.executors."

    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if not alias.name.startswith(prefix):
                continue
            importlib.reload(importlib.import_module(alias.name))


def test_bootstrap_asserts_registry_non_empty(_isolate_both_registries) -> None:
    """If no contracts are registered at bootstrap, prepare_for_run() must raise
    RuntimeError naming every contract missing from the manifest.

    This is the core safety guarantee of ADR-010 §Decision 3: a silently empty
    registry disables all runtime VAL checks. prepare_for_run() must make that
    failure visible immediately with an explicit missing-contract diff.
    """
    from elspeth.contracts.declaration_contracts import (
        EXPECTED_CONTRACT_SITES,
        _clear_registry_for_tests,
    )
    from elspeth.engine.orchestrator import prepare_for_run

    # Clear the registry WITHOUT re-importing the bootstrap module — simulates
    # an import-order bug where the authoritative import surface never fired.
    _clear_registry_for_tests()

    with pytest.raises(RuntimeError) as exc_info:
        prepare_for_run()

    message = str(exc_info.value)
    assert "declaration contract registry mismatch" in message.lower()
    # Every expected contract must be explicitly named in the diff.
    for expected_name in EXPECTED_CONTRACT_SITES:
        assert repr(expected_name) in message or expected_name in message, (
            f"Error message must name missing contract {expected_name!r}; got: {message}"
        )


def test_bootstrap_fails_when_specific_contract_conditionally_skipped(_isolate_both_registries) -> None:
    """Simulates the C2 failure mode: a *subset* of expected contracts is
    registered (e.g. a conditional import silently skipped PassThroughDeclarationContract).

    The registry is non-empty — the old ``if not contracts`` check would have
    passed. The new set-equality check must raise with ``missing:`` naming
    ``passes_through_input`` explicitly.

    This is the ADR-010 C2 acceptance criterion: wrapping a required contract
    registration in a skipped conditional must fail bootstrap loudly.
    """
    from typing import TypedDict

    from elspeth.contracts.declaration_contracts import (
        _clear_registry_for_tests,
        register_declaration_contract,
    )
    from elspeth.engine.orchestrator import prepare_for_run

    class _DummyPayload(TypedDict):
        reason: str

    from elspeth.contracts.declaration_contracts import (
        DeclarationContract,
        implements_dispatch_site,
    )

    class _UnexpectedContract(DeclarationContract):
        name = "unexpected_for_c2_test"
        payload_schema: type = _DummyPayload

        def applies_to(self, plugin: object) -> bool:
            return False

        @implements_dispatch_site("post_emission_check")
        def post_emission_check(self, inputs: object, outputs: object) -> None:
            pass

        @classmethod
        def negative_example(cls):
            raise NotImplementedError

        @classmethod
        def positive_example_does_not_apply(cls):
            raise NotImplementedError

    # Clear registry, then register ONLY an unexpected contract — the bootstrap
    # surface is missing. The registry is non-empty, so a bare truthiness check
    # would pass. The manifest-equality check must catch this.
    _clear_registry_for_tests()
    register_declaration_contract(_UnexpectedContract())

    with pytest.raises(RuntimeError) as exc_info:
        prepare_for_run()

    message = str(exc_info.value)
    # The canonical PassThroughDeclarationContract name must appear under "missing".
    assert "passes_through_input" in message, f"Expected error to name missing contract 'passes_through_input'; got: {message}"
    assert "missing" in message.lower()


def test_bootstrap_fails_when_extra_contract_registered(_isolate_both_registries) -> None:
    """If a contract outside the manifest is registered, bootstrap must raise.

    Set-equality — not subset — is the safety guarantee. An extra registration
    indicates the manifest is out of date; CI should have caught it. Bootstrap
    is the last line of defence.
    """
    import importlib
    from typing import TypedDict

    from elspeth.contracts.declaration_contracts import (
        _clear_registry_for_tests,
        register_declaration_contract,
    )
    from elspeth.engine.orchestrator import prepare_for_run

    class _DummyPayload(TypedDict):
        reason: str

    from elspeth.contracts.declaration_contracts import (
        DeclarationContract,
        implements_dispatch_site,
    )

    class _UnexpectedContract(DeclarationContract):
        name = "extra_not_in_manifest"
        payload_schema: type = _DummyPayload

        def applies_to(self, plugin: object) -> bool:
            return False

        @implements_dispatch_site("post_emission_check")
        def post_emission_check(self, inputs: object, outputs: object) -> None:
            pass

        @classmethod
        def negative_example(cls):
            raise NotImplementedError

        @classmethod
        def positive_example_does_not_apply(cls):
            raise NotImplementedError

    # Clear + re-register PassThrough so manifest is satisfied, then add an
    # unexpected extra contract.
    _clear_registry_for_tests()
    import elspeth.engine.executors.pass_through as pt_mod

    importlib.reload(pt_mod)
    register_declaration_contract(_UnexpectedContract())

    with pytest.raises(RuntimeError) as exc_info:
        prepare_for_run()

    message = str(exc_info.value)
    assert "extra_not_in_manifest" in message, f"Expected error to name extra contract 'extra_not_in_manifest'; got: {message}"
    assert "extra" in message.lower()


def test_bootstrap_passes_when_registry_exactly_matches_manifest(_isolate_both_registries) -> None:
    """The happy path: registry == EXPECTED_CONTRACT_SITES → no exception, registries freeze.

    Verifies the set-equality check permits the nominal case. Uses the
    importlib.reload pattern to re-trigger the authoritative bootstrap module
    after _clear_registry_for_tests().
    """
    from elspeth.contracts.declaration_contracts import (
        EXPECTED_CONTRACT_SITES,
        _clear_registry_for_tests,
        declaration_registry_is_frozen,
    )
    from elspeth.engine.orchestrator import prepare_for_run

    _clear_registry_for_tests()
    _replay_bootstrap_import_surface()

    # Must not raise.
    prepare_for_run()
    assert declaration_registry_is_frozen()
    assert len(EXPECTED_CONTRACT_SITES) == 7
    assert frozenset(EXPECTED_CONTRACT_SITES.keys()) == frozenset(
        {
            "passes_through_input",
            "declared_output_fields",
            "declared_required_fields",
            "schema_config_mode",
            "can_drop_rows",
            "source_guaranteed_fields",
            "sink_required_fields",
        }
    )


def test_bootstrap_freezes_declaration_registry(_isolate_both_registries) -> None:
    """After prepare_for_run(), the declaration-contract registry must be frozen.

    Subsequent registration attempts must raise FrameworkBugError, proving the
    registry cannot be extended after bootstrap completes.
    """
    from elspeth.contracts.declaration_contracts import (
        _clear_registry_for_tests,
        register_declaration_contract,
    )
    from elspeth.contracts.tier_registry import FrameworkBugError
    from elspeth.engine.orchestrator import prepare_for_run

    # Clear registry and re-import the authoritative bootstrap module to
    # repopulate every production contract via its module-level side-effects.
    _clear_registry_for_tests()
    _replay_bootstrap_import_surface()

    # Bootstrap: assert non-empty, freeze both registries.
    prepare_for_run()

    # Post-bootstrap: any registration attempt must fail with FrameworkBugError.
    from typing import TypedDict

    class _PostBootstrapPayload(TypedDict):
        reason: str

    from elspeth.contracts.declaration_contracts import (
        DeclarationContract,
        implements_dispatch_site,
    )

    class _PostBootstrapContract(DeclarationContract):
        name = "post_bootstrap_contract"
        payload_schema: type = _PostBootstrapPayload

        def applies_to(self, plugin: object) -> bool:
            return False

        @implements_dispatch_site("post_emission_check")
        def post_emission_check(self, inputs: object, outputs: object) -> None:
            pass

        @classmethod
        def negative_example(cls):
            raise NotImplementedError

        @classmethod
        def positive_example_does_not_apply(cls):
            raise NotImplementedError

    with pytest.raises(FrameworkBugError):
        register_declaration_contract(_PostBootstrapContract())


def test_bootstrap_freezes_tier_registry(_isolate_both_registries) -> None:
    """After prepare_for_run(), the tier-1 error registry must be frozen.

    Subsequent @tier_1_error registrations must raise FrameworkBugError.
    """
    from elspeth.contracts.declaration_contracts import _clear_registry_for_tests
    from elspeth.contracts.tier_registry import FrameworkBugError, tier_1_error
    from elspeth.engine.orchestrator import prepare_for_run

    # Clear declaration registry and reload the bootstrap module to repopulate.
    _clear_registry_for_tests()
    _replay_bootstrap_import_surface()

    prepare_for_run()

    # Post-bootstrap: tier-1 registration must fail.
    with pytest.raises(FrameworkBugError, match="frozen"):

        @tier_1_error(reason="post-bootstrap: must fail", caller_module=__name__)
        class _TooLate(Exception):
            pass


def test_prepare_for_run_succeeds_in_subprocess_via_bootstrap_import_surface() -> None:
    """Production-shape regression: prepare_for_run() must work in a fresh process.

    This is the real failure mode the explicit bootstrap module fixes: a fresh
    process importing ``elspeth.engine.orchestrator`` and calling
    ``prepare_for_run()`` must succeed without pytest importing
    ``pass_through.py`` as a side effect first.
    """
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_root / 'src'}:{existing_pythonpath}" if existing_pythonpath else str(repo_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from elspeth.engine.orchestrator import prepare_for_run; prepare_for_run()",
        ],
        check=False,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        "prepare_for_run() failed in a fresh subprocess despite the bootstrap "
        f"module. stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )


def test_resume_calls_prepare_for_run() -> None:
    """resume() must call prepare_for_run() before any recovery work.

    The resume path runs in a new process. Module-level imports ensure
    PassThroughDeclarationContract is registered, but without an explicit
    prepare_for_run() call the registries are never frozen — leaving a window
    where register_declaration_contract() could succeed after bootstrap.
    """
    import elspeth.engine.orchestrator.core as orchestrator_core
    from elspeth.contracts import Checkpoint, ResumePoint
    from elspeth.engine.orchestrator.core import Orchestrator
    from tests.fixtures.landscape import make_landscape_db
    from tests.fixtures.stores import MockPayloadStore

    calls: list[str] = []

    class ReconstructReached(RuntimeError):
        """Stops resume after the bootstrap ordering assertion fires."""

    def fake_prepare_for_run() -> None:
        calls.append("prepare_for_run")

    def fake_reconstruct_resume_state(self, resume_point, payload_store):  # type: ignore[no-untyped-def]
        assert calls == ["prepare_for_run"], (
            "Orchestrator.resume() must invoke prepare_for_run() before reconstructing resume state (ADR-010 §Decision 3)"
        )
        calls.append("reconstruct_resume_state")
        raise ReconstructReached

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(orchestrator_core, "prepare_for_run", fake_prepare_for_run)
    monkeypatch.setattr(Orchestrator, "_reconstruct_resume_state", fake_reconstruct_resume_state)
    try:
        checkpoint = Checkpoint(
            checkpoint_id="cp-resume-bootstrap",
            run_id="run-resume-bootstrap",
            token_id="tok-resume-bootstrap",
            node_id="node-resume-bootstrap",
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            checkpoint_node_config_hash="b" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(
            checkpoint=checkpoint,
            token_id=checkpoint.token_id,
            node_id=checkpoint.node_id,
            sequence_number=checkpoint.sequence_number,
        )

        orchestrator = Orchestrator(make_landscape_db())

        with pytest.raises(ReconstructReached):
            orchestrator.resume(
                resume_point=resume_point,
                config=object(),  # type: ignore[arg-type]
                graph=object(),  # type: ignore[arg-type]
                payload_store=MockPayloadStore(),
            )
    finally:
        monkeypatch.undo()

    assert calls == ["prepare_for_run", "reconstruct_resume_state"], (
        "resume() must freeze runtime-VAL registries before any recovery-state reconstruction."
    )
