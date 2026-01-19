## Repository Shape (high level)
- Language: Python package under `src/elspeth/`
- Entry points:
  - CLI: `elspeth` → `src/elspeth/cli.py`
  - Script: `check-contracts` → `scripts/check_contracts.py`
- Key infra:
  - DB migrations: `alembic/` + `alembic.ini`
  - CI static analysis config: `config/cicd/no_bug_hiding.yaml`

## Major Directories
- `src/elspeth/contracts/`: Typed contracts (Pydantic models, protocols, enums, results/errors).
- `src/elspeth/core/`: Core utilities + framework foundations:
  - `canonical.py`, `config.py`, `dag.py`, `payload_store.py`, `logging.py`
  - `checkpoint/`, `rate_limit/`, `retention/`
  - `landscape/`: audit database subsystem (schema, recorder, lineage, export, etc.)
- `src/elspeth/engine/`: Execution runtime (orchestrator/processor, tokens, retries, triggers, executors, expression parsing).
- `src/elspeth/plugins/`: Plugin framework + built-in plugins:
  - `sources/`, `transforms/`, `sinks/`
- `src/elspeth/tui/`: Textual TUI for explain/inspection.
- `tests/`: Subsystem-oriented test suites + integration tests.

## Tech Stack (from `pyproject.toml`)
- CLI/TUI: Typer, Textual
- Config/contracts: Dynaconf, Pydantic v2
- Plugin system: pluggy
- Data: pandas
- DB: SQLAlchemy Core + Alembic
- Reliability: tenacity, pyrate-limiter
- Canonical hashing: rfc8785
- Graph validation: networkx
- Observability: OpenTelemetry
- Diffing: DeepDiff

## First-pass Subsystem Candidates
1. Contracts & schema layer (`contracts/`)
2. Configuration loader/resolution (`core/config.py`)
3. Canonicalization & hashing (`core/canonical.py`)
4. DAG compilation/validation (`core/dag.py`)
5. Landscape audit DB (all of `core/landscape/`)
6. Engine runtime (`engine/`)
7. Plugin framework (`plugins/`)
8. Built-in IO plugins (`plugins/sources|sinks|transforms`)
9. Checkpointing & recovery (`core/checkpoint/`)
10. Rate limiting (`core/rate_limit/`)
11. Retention/purge (`core/retention/` + `cli purge`)
12. TUI explain tooling (`tui/`)
