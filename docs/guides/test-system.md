# Test System Guide

Current as of 2026-05-20.

ELSPETH's test system is a pytest suite with strict marker/config validation,
Hypothesis profiles, shared fixture factories, and directory-based test tiers.
The active source of truth for runner behavior is
[`pyproject.toml`](../../pyproject.toml), not this guide.

On 2026-05-20, `python -m pytest --collect-only -q` collected 19,163 pytest
items in this checkout. The default marker expression deselected 115 slow,
stress, performance, or testcontainer items, leaving 19,048 default-selected
items. Treat these numbers as a freshness check, not a permanent contract.

## Runner Configuration

Pytest is configured in `pyproject.toml` under `[tool.pytest.ini_options]`.

Default behavior:

- Test discovery starts at `tests/`.
- `src` and `elspeth-lints/src` are on `pythonpath`.
- Strict marker and strict config checks are enabled.
- The default run excludes `slow`, `stress`, `performance`, and
  `testcontainer` tests.
- `.env` is loaded for integration-style tests that need local operator
  settings.

Useful commands:

```bash
python -m pytest
python -m pytest tests/unit
python -m pytest tests/integration
python -m pytest tests/property
python -m pytest tests/e2e -m e2e
python -m pytest -m "performance or stress"
python -m pytest -m testcontainer
python -m pytest --collect-only -q
```

Use the repository virtualenv for local checks:

```bash
/home/john/elspeth/.venv/bin/python -m pytest
```

## Suite Layout

Top-level test groups in the current tree:

- `tests/unit/` - focused behavior checks, including engine, web, plugin,
  lint, CLI, MCP, telemetry, and regression coverage.
- `tests/property/` - Hypothesis-driven invariants for audit, contracts,
  core, engine, plugins, sources, sinks, telemetry, and web behavior.
- `tests/integration/` - multi-component checks using real repositories,
  adapters, configuration, web services, and pipeline assembly paths.
- `tests/e2e/` - end-to-end pipeline, audit, external-system, example, and
  recovery paths.
- `tests/performance/` - benchmark, scalability, memory, and stress lanes.
- `tests/invariants/` - repository-level invariants that guard architecture
  and policy drift.
- `tests/fixtures/`, `tests/helpers/`, and `tests/strategies/` - shared test
  infrastructure, not product behavior suites.

The current tree also includes generated/golden coverage under `tests/golden/`.

## Markers

Registered markers include:

- `integration`
- `e2e`
- `performance`
- `stress`
- `slow`
- `composer_llm_eval`
- `chaosllm`
- `testcontainer`
- `fingerprint_baseline`

`tests/conftest.py` also auto-applies `e2e`, `performance`, and `stress`
markers from directory location. Integration-specific behavior lives in
`tests/integration/conftest.py`.

## Hypothesis Profiles

`tests/conftest.py` registers three profiles:

- `ci` - default profile, 100 examples.
- `nightly` - expanded profile, 1,000 examples.
- `debug` - small verbose profile, 10 examples.

Select a profile with:

```bash
HYPOTHESIS_PROFILE=debug python -m pytest tests/property
HYPOTHESIS_PROFILE=nightly python -m pytest tests/property
```

## Fixture And Factory Rules

Shared fixtures are intentionally centralized:

- `tests/conftest.py` owns root marker registration, Hypothesis profiles,
  runtime registry cleanup, and test-wide safety fixtures.
- Group-level `conftest.py` files own group-specific resources.
- `tests/fixtures/factories.py` re-exports production-type factories from
  `elspeth.testing` and defines test-only helpers.
- `tests/fixtures/plugins.py` owns canonical test plugins such as `ListSource`
  and `CollectSink`.
- `tests/strategies/` owns reusable Hypothesis strategies.

Use production assembly paths for integration, E2E, and performance tests. In
particular, do not hand-build execution graphs in higher-tier tests when the
behavior under test depends on configuration parsing, plugin discovery, or
runtime wiring.

## Safety Invariants

The root conftest enforces several project-specific safety properties:

- Tests refuse to run under `python -O` because optimized mode strips asserts.
- Runtime declaration-contract and Tier-1 registries are restored after each
  test and fail the test if membership leaks.
- Raw secrets are allowed in tests via environment override so local and CI
  behavior remain aligned.
- Production declaration-contract bootstrap is imported once per pytest process
  so xdist workers start from the same contract registry.

When adding tests around audit, tier-model, or config-contract behavior, prefer
fixtures and factories that exercise the production path. Tests that bypass the
path being protected can make broken production behavior look green.
