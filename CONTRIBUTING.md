# Contributing to ELSPETH

Thanks for your interest in contributing to ELSPETH! This document covers the basics for getting started.

## Development Setup

```bash
git clone https://github.com/johnm-dta/elspeth.git && cd elspeth
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,azure]"

# Azure Blob integration tests require the Azurite emulator
npm install
```

**Important:** Use `uv` for all package management. Do not use `pip` directly.

## Running Quality Checks

All of these must pass before submitting changes:

```bash
# Tests
.venv/bin/python -m pytest tests/ -v

# Type checking
.venv/bin/python -m mypy src/

# Linting
.venv/bin/python -m ruff check src/

# Config contracts verification
.venv/bin/python -m scripts.check_contracts
```

## Code Standards

- **No defensive programming** against our own code. Access typed fields directly (`obj.field`), not defensively (`getattr(obj, "field", None)`). See [Data Trust and Error Handling](docs/guides/data-trust-and-error-handling.md) for the trust-boundary rationale.
- **Three-tier trust model.** Handle errors differently based on data origin: crash on audit data corruption, quarantine bad user data, validate external API responses at the boundary.
- **No legacy shims or backwards compatibility.** When changing something, delete the old code completely. No deprecation wrappers, no commented-out code, no compatibility layers.
- **Test through production code paths.** Integration tests must use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()`, not manually constructed objects.

## Writing Tests

Use the centralized test factories — do not construct framework objects directly:

```python
# WRONG - couples test to constructor signature
from elspeth.contracts.plugin_context import PluginContext
ctx = PluginContext(recorder=recorder, run_id="run-1", node_id="node-1", ...)

# RIGHT - use factories
from tests.fixtures.landscape import make_recorder_with_run
from tests.fixtures.factories import make_context
recorder, run_id = make_recorder_with_run()
ctx = make_context(recorder=recorder, run_id=run_id)
```

**Key factories:**

| Factory | Location | Purpose |
|---------|----------|---------|
| `make_recorder_with_run()` | `tests.fixtures.landscape` | Create `LandscapeRecorder` with a registered run |
| `make_context()` | `tests.fixtures.factories` | Create `PluginContext` with correct wiring |
| `make_landscape_db()` | `tests.fixtures.landscape` | Create in-memory `LandscapeDB` |
| `register_test_node()` | `tests.fixtures.landscape` | Register a node in the audit trail |
| Shared plugins | `tests.fixtures.plugins` | `PassthroughTransform`, `FailingSink`, etc. |

**Additional quality checks:**

```bash
# Tier model enforcement (layer dependency detection)
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
```

## Commit Guidelines

- Keep commits focused on a single logical change.
- Write commit messages that explain *why*, not just *what*.
- Ensure all quality checks pass before committing.

## Reporting Issues

Open an issue on GitHub with:

- A clear description of the problem or suggestion.
- Steps to reproduce (for bugs).
- Expected vs. actual behavior.

For security-sensitive issues, follow [SECURITY.md](SECURITY.md) instead of
opening a public issue with details. Do not include exploit details, secrets, personal data, or sensitive audit material in a public issue.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
