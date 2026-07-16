# tests/conftest.py
"""Root conftest for test suite v2.

Responsibilities:
- Register ALL pytest markers
- Register Hypothesis profiles (ci, nightly, debug)
- Auto-mark tests by directory location
- Autouse secrets fixture for CI parity
"""

from __future__ import annotations

import os
import sys

import pytest
from hypothesis import Phase, Verbosity, settings

# Belt-and-suspenders fence: the Tier-1 guards in production code are
# explicit ``raise AuditIntegrityError`` (survives ``python -O``), but a
# handful of existing tests still use plain ``assert`` statements in
# arrange/act lines.  Running the suite under ``-O`` would silently erase
# those assertions, turning coverage-theatre into green-on-broken.  We
# refuse to import the suite under an optimised interpreter so the
# failure is loud and unmistakable.  Expressed as an ``if / raise`` (not
# ``assert``) because ``-O`` strips asserts at import time.
if sys.flags.optimize != 0:
    raise RuntimeError(
        "ELSPETH tests must not run under `python -O` — assert statements are stripped, "
        "which silently disables assertion-based test contracts.  Re-run without -O."
    )

# ---------------------------------------------------------------------------
# DeclarationContract registry population
# ---------------------------------------------------------------------------
#
# ADR-010 Phase 2A introduced a set-equality bootstrap check against
# EXPECTED_CONTRACTS (issue elspeth-b03c6112c0 / C2). Every contract in
# the manifest MUST be registered before ``prepare_for_run()`` is called,
# or bootstrap raises. Registration is a module-import side effect — see
# ``src/elspeth/contracts/declaration_contracts.py`` CLOSED-SET comment.
#
# Test files that invoke the orchestrator (directly or via ``elspeth
# run``) but do not transitively import the contract-defining executors
# hit an empty-registry bootstrap failure. In xdist-distributed runs the
# failure manifests non-deterministically depending on which worker
# receives which test file first. Importing the authoritative production
# bootstrap surface at root-conftest level populates the registry once
# per pytest process (xdist workers included) so every test starts from
# the same contract set production uses. Individual tests that need to
# clear or mutate the registry use the
# ``_snapshot_registry_for_tests`` / ``_restore_registry_snapshot_for_tests``
# helpers, which are pytest-gated (issue elspeth-cc511e7234 / C3).
import elspeth.engine.executors.declaration_contract_bootstrap  # noqa: F401

pytest_plugins = ["tests.fixtures.azurite"]

# ---------------------------------------------------------------------------
# Marker Registration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for test tiers."""
    config.addinivalue_line("markers", "integration: multi-component tests with real DB")
    config.addinivalue_line("markers", "e2e: full pipeline, real I/O, file-based DB")
    config.addinivalue_line("markers", "performance: benchmarks and regression detection")
    config.addinivalue_line("markers", "stress: load tests requiring ChaosLLM HTTP server")
    config.addinivalue_line("markers", "slow: long-running tests (>10s)")
    config.addinivalue_line(
        "markers",
        "composer_llm_eval: characterization/replay tests for the 2026-04-28 composer LLM evaluation scenarios",
    )
    config.addinivalue_line(
        "markers",
        "chaosllm(preset=None, **kwargs): Configure ChaosLLM server for the test. "
        "Use preset='name' to load a preset, and keyword args to override specific settings.",
    )


# ---------------------------------------------------------------------------
# Auto-Marking by Directory
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply markers based on test file location."""
    for item in items:
        path = str(item.fspath)
        if "/e2e/" in path:
            item.add_marker(pytest.mark.e2e)
        elif "/performance/" in path and "/stress/" in path:
            item.add_marker(pytest.mark.stress)
            item.add_marker(pytest.mark.performance)
        elif "/performance/" in path:
            item.add_marker(pytest.mark.performance)
        # integration/ tests get marker from their conftest


# ---------------------------------------------------------------------------
# Hypothesis Profiles
# ---------------------------------------------------------------------------

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)
settings.register_profile(
    "nightly",
    max_examples=1000,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)
settings.register_profile(
    "debug",
    max_examples=10,
    deadline=None,
    verbosity=Verbosity.verbose,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))


@pytest.fixture(autouse=True)
def _allow_raw_secrets_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow raw secrets in all tests — CI has no .env file.

    Locally, .env sets ELSPETH_ALLOW_RAW_SECRETS=true which bypasses
    the fingerprint key requirement for test API keys.  CI doesn't load
    .env, so tests that create AuditedHTTPClient with auth headers fail
    with FrameworkBugError.  This fixture ensures consistent behaviour.
    """
    monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")


@pytest.fixture(autouse=True)
def _freeze_runtime_val_registries_before_begin_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror the runtime-VAL manifest precondition for direct repository tests.

    Production reaches ``RunLifecycleRepository.begin_run()`` only after
    orchestrator bootstrap has sealed the declaration and Tier-1 registries.
    Many tests exercise the repository layer directly or register test-only
    declaration contracts that are intentionally outside
    ``EXPECTED_CONTRACT_SITES``. For those paths we freeze the current test
    registry state immediately before ``begin_run()`` rather than forcing the
    full production bootstrap equality check.

    Empty declaration registries are left unfrozen so ``begin_run()`` still
    fails closed when a test genuinely models the unsafe path.
    """
    import functools

    from elspeth.contracts.declaration_contracts import (
        freeze_declaration_registry,
        registered_declaration_contracts,
    )
    from elspeth.contracts.tier_registry import _TIER_1_ERRORS_VIEW, freeze_tier_registry
    from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository

    original_begin_run = RunLifecycleRepository.begin_run

    @functools.wraps(original_begin_run)
    def wrapped_begin_run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if registered_declaration_contracts():
            freeze_declaration_registry()
        if len(_TIER_1_ERRORS_VIEW) > 0:
            freeze_tier_registry()
        # OpenRouter catalog snapshot is mandatory at the runs row.  In
        # production, the L3 entry point (web lifespan / CLI bootstrap)
        # resolves these via ``read_openrouter_catalog_snapshot_id``;
        # the resulting tuple is non-None by construction (the bundled
        # fallback is always available).  Unit tests construct repos
        # directly and bypass that resolution — supply a deterministic
        # synthetic snapshot so tests don't need to thread the value
        # through every call site.  Tests that exercise the snapshot
        # contract supply their own values (or call the real reader).
        kwargs.setdefault("openrouter_catalog_sha256", "0" * 64)
        kwargs.setdefault("openrouter_catalog_source", "bundled")
        return original_begin_run(self, *args, **kwargs)

    monkeypatch.setattr(RunLifecycleRepository, "begin_run", wrapped_begin_run)

    # Mirror the snapshot defaulting at ``orchestrator.run`` so tests
    # invoking the orchestrator directly (without the L3 entry-point
    # wiring that resolves the snapshot in production) don't need to
    # thread the value through every call site. Production callers
    # (web/execution/service.py, cli.py, cli_helpers.py) always supply
    # both fields explicitly.  ``functools.wraps`` preserves
    # ``inspect.signature(Orchestrator.run)`` so signature-introspection
    # tests still see the real parameter names.
    from elspeth.engine.orchestrator.core import Orchestrator

    original_orch_run = Orchestrator.run

    def with_test_sink_effect_modes(config):  # type: ignore[no-untyped-def]
        """Mirror RuntimePluginFactory mode resolution for test-only sinks."""
        from dataclasses import replace
        from unittest.mock import Mock

        from elspeth.contracts import (
            ResolvedSinkEffectMode,
            SinkEffectExecutionPurpose,
            SinkEffectInputKind,
        )
        from elspeth.engine.orchestrator import PipelineConfig
        from elspeth.engine.orchestrator.preflight import validate_pipeline_sink_effect_capabilities

        if isinstance(config, Mock):
            # Resume unit tests use a spec-bound PipelineConfig mock for fields
            # unrelated to sink execution. Give the new admission boundary an
            # exact empty surface instead of letting auto-created Mock fields
            # masquerade as a forged admission receipt.
            config.sinks = {}
            config.sink_effect_modes = {}
            config.sink_effect_admission = None
            return config
        if type(config) is not PipelineConfig:
            return config
        modes: dict[str, str] = dict(config.sink_effect_modes)
        if not modes:
            for sink_name, sink in config.sinks.items():
                resolver = getattr(type(sink), "_resolve_sink_effect_mode", None)
                if resolver is None:
                    continue
                resolved = resolver(dict(sink.config), purpose=SinkEffectExecutionPurpose.FRESH)
                if resolved is None:
                    continue
                if not isinstance(resolved, ResolvedSinkEffectMode):
                    raise TypeError("test sink effect resolver must return ResolvedSinkEffectMode")
                modes[sink_name] = resolved.value
        admission = validate_pipeline_sink_effect_capabilities(
            config.sinks,
            configured_modes=modes,
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )
        return replace(config, sink_effect_modes=modes, sink_effect_admission=admission)

    @functools.wraps(original_orch_run)
    def wrapped_orch_run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Direct orchestrator tests bypass RuntimePluginFactory, which is the
        # production authority that resolves and binds sink-effect modes. Keep
        # test-only sink fixtures honest under the same effect-only runtime by
        # resolving their declared mode before the call. Production sink types
        # and tests that provide an explicit mode map are left untouched.
        if "config" in kwargs:
            kwargs["config"] = with_test_sink_effect_modes(kwargs["config"])
        elif args:
            args = (with_test_sink_effect_modes(args[0]), *args[1:])
        kwargs.setdefault("openrouter_catalog_sha256", "0" * 64)
        kwargs.setdefault("openrouter_catalog_source", "bundled")
        return original_orch_run(self, *args, **kwargs)

    monkeypatch.setattr(Orchestrator, "run", wrapped_orch_run)

    original_orch_resume = Orchestrator.resume

    @functools.wraps(original_orch_resume)
    def wrapped_orch_resume(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "config" in kwargs:
            kwargs["config"] = with_test_sink_effect_modes(kwargs["config"])
        elif len(args) >= 2:
            args = (args[0], with_test_sink_effect_modes(args[1]), *args[2:])
        return original_orch_resume(self, *args, **kwargs)

    monkeypatch.setattr(Orchestrator, "resume", wrapped_orch_resume)

    # ExecutionServiceImpl: in production the lifespan calls
    # ``set_openrouter_catalog_snapshot()`` after construction. Tests
    # construct the service directly and skip that step; auto-prime the
    # synthetic snapshot in ``__init__`` so ``_run_pipeline`` doesn't
    # raise on the wiring check. Production ``app.py`` still calls
    # ``set_openrouter_catalog_snapshot`` explicitly — and the setter
    # validates inputs — so this wrapper doesn't mask production bugs.
    from elspeth.web.execution.service import ExecutionServiceImpl

    original_exec_init = ExecutionServiceImpl.__init__

    @functools.wraps(original_exec_init)
    def wrapped_exec_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_exec_init(self, *args, **kwargs)
        self._openrouter_catalog_sha256 = "0" * 64
        self._openrouter_catalog_source = "bundled"

    monkeypatch.setattr(ExecutionServiceImpl, "__init__", wrapped_exec_init)


@pytest.fixture(autouse=True)
def _restore_runtime_val_registries_after_each_test() -> None:
    """Restore runtime-VAL registries and fail on leaked registry membership.

    Tests may freeze registries through production bootstrap paths; that flag
    is restored without failing. Membership and reason/site-map mutations must
    be restored by the test that made them, because leaking synthetic contracts
    or Tier-1 classes changes every later runtime-VAL manifest.
    """
    import elspeth.contracts.declaration_contracts as dc
    import elspeth.contracts.tier_registry as tr

    with dc._REGISTRY_LOCK:
        saved_dc_registry = list(dc._REGISTRY)
        saved_dc_per_site = {site: list(lst) for site, lst in dc._REGISTRY_BY_SITE.items()}
        saved_dc_frozen = dc._FROZEN

    with tr._REGISTRY_LOCK:
        saved_tr_registry = list(tr._REGISTRY)
        saved_tr_reasons = dict(tr._REASONS)
        saved_tr_frozen = tr._FROZEN

    yield

    leaked: list[str] = []
    with dc._REGISTRY_LOCK:
        if saved_dc_registry != dc._REGISTRY or any(saved_dc_per_site[site] != dc._REGISTRY_BY_SITE[site] for site in dc.DispatchSite):
            leaked.append("declaration-contract registry")
        dc._REGISTRY[:] = saved_dc_registry
        for site in dc.DispatchSite:
            dc._REGISTRY_BY_SITE[site][:] = saved_dc_per_site[site]
        dc._FROZEN = saved_dc_frozen

    with tr._REGISTRY_LOCK:
        if saved_tr_registry != tr._REGISTRY or saved_tr_reasons != tr._REASONS:
            leaked.append("Tier-1 error registry")
        tr._REGISTRY[:] = saved_tr_registry
        tr._REASONS.clear()
        tr._REASONS.update(saved_tr_reasons)
        tr._FROZEN = saved_tr_frozen

    if leaked:
        pytest.fail(f"Runtime-VAL registry state leaked from test: {', '.join(leaked)}")
