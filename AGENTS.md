# Repository Guidelines

## Project Structure & Module Organization
The Python package lives in `src/elspeth/` with clear subsystem boundaries: `core/` (canonicalization, config, landscape, payload store), `engine/` (pipeline execution), and `plugins/` (`sources/`, `transforms/`, `sinks/`). Tests live in `tests/` with pytest, and design documentation is in `docs/` (see `docs/design/architecture.md`). Tooling, dependencies, and configs are centralized in `pyproject.toml`.

## Build, Test, and Development Commands
Use `uv` for all package management (no direct `pip` usage).

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
```

The CLI entry point is declared as `elspeth`, but the project is still in a design-to-implementation phase; verify availability in code and `README.md` before relying on runtime commands.

## Coding Style & Naming Conventions
Python 3.11+ with type hints is expected. Ruff enforces a line length of 88 and lint rules in `pyproject.toml`. Mypy runs in strict mode. Use `snake_case` for modules/functions and `CamelCase` for classes; tests should be named `test_*.py`. Avoid defensive programming patterns that hide bugs (for example, replacing direct attribute access with `.get()`), and do not add compatibility shims or legacy pathways—remove old code outright when behavior changes.

## Testing Guidelines
Tests use pytest and live under `tests/`. Markers include `slow` and `integration`; run fast unit tests with:

```bash
.venv/bin/python -m pytest -m "not slow"
```

Coverage configuration is in `pyproject.toml`; add tests alongside new core behavior, especially for canonicalization and config handling.

## Commit & Pull Request Guidelines
Commit messages follow a conventional style seen in history: `feat(scope): summary`, `docs: summary`, `test(scope): summary`. Use lowercase types and short, imperative summaries. PRs should include a concise description, rationale, tests run, and any doc or config updates (including `uv.lock` when dependencies change), and link relevant issues if applicable.

## Security & Configuration Tips
Configuration uses Dynaconf + Pydantic. Keep secrets in environment variables (for example, `OPENAI_API_KEY`) and avoid committing credentials or local configuration files.
