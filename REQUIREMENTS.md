# Requirements

System requirements and dependencies for ELSPETH.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Python Dependencies](#python-dependencies)
- [Optional Dependencies](#optional-dependencies)
- [Development Dependencies](#development-dependencies)
- [Database Requirements](#database-requirements)
- [Container Requirements](#container-requirements)
- [Verification](#verification)

---

## System Requirements

### Runtime

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.12+ | Required for modern type hints and performance |
| **Operating System** | Linux, macOS, Windows | Linux recommended for production |
| **Memory** | 512 MB minimum | More for large datasets or concurrent workers |
| **Disk** | Varies | Audit database grows with pipeline volume |

### Package Manager

ELSPETH uses **uv** for package management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

**Why uv?**
- 10-100x faster than pip
- Deterministic dependency resolution
- Better conflict detection
- Drop-in pip replacement

---

## Python Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `typer` | >=0.12 | CLI framework |
| `textual` | >=0.52 | Terminal UI for `explain` command |
| `dynaconf` | >=3.2 | Configuration management |
| `pydantic` | >=2.6 | Data validation and settings |
| `pluggy` | >=1.4 | Plugin system |
| `pandas` | >=2.2 | Tabular data handling |
| `sqlalchemy` | >=2.0 | Database abstraction |
| `alembic` | >=1.13 | Database migrations |
| `tenacity` | >=8.2 | Retry logic |

### Acceleration Stack

These dependencies are battle-tested libraries that replace hand-rolled implementations:

| Package | Version | Replaces |
|---------|---------|----------|
| `rfc8785` | >=0.1 | Hand-rolled canonical JSON (RFC 8785/JCS standard) |
| `networkx` | >=3.2 | Custom graph algorithms (DAG validation) |
| `opentelemetry-*` | >=1.23 | Custom tracing (immediate Jaeger visualization) |
| `structlog` | >=24.1 | Ad-hoc logging (structured events) |
| `pyrate-limiter` | >=3.1 | Custom rate limiting |
| `deepdiff` | >=7.0 | Custom comparison (for verify mode) |

---

## Optional Dependencies

### LLM Pack

For pipelines using LLM-based transforms:

```bash
uv pip install -e ".[llm]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `litellm` | >=1.30 | Unified LLM provider interface |
| `openai` | >=1.0 | OpenAI API client |
| `jinja2` | >=3.1 | Prompt templating |

### Azure Pack

For Azure cloud integration:

```bash
uv pip install -e ".[azure]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `azure-storage-blob` | >=12.19 | Azure Blob Storage sink |
| `azure-identity` | >=1.15 | Azure authentication |
| `azure-keyvault-secrets` | >=4.7 | Key Vault for fingerprint key |

### All Optional Dependencies

Install everything:

```bash
uv pip install -e ".[all]"
```

---

## Development Dependencies

For contributing to ELSPETH:

```bash
uv pip install -e ".[dev]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=8.0 | Test framework |
| `pytest-cov` | >=4.1 | Coverage reporting |
| `pytest-asyncio` | >=0.23 | Async test support |
| `hypothesis` | >=6.98 | Property-based testing |
| `mypy` | >=1.8 | Static type checking |
| `ruff` | >=0.3 | Linting and formatting |
| `pre-commit` | >=3.6 | Git hooks |
| `mutmut` | >=2.4 | Mutation testing |

---

## Database Requirements

### SQLite (Default)

No additional setup required. SQLite is included with Python.

- **Minimum version:** 3.35+ (for JSON functions)
- **Recommended:** Use write-ahead logging (WAL) for concurrent access

### PostgreSQL (Production)

For high-volume production deployments:

| Requirement | Version | Notes |
|-------------|---------|-------|
| PostgreSQL | 13+ | For JSON functions and performance |
| `psycopg2-binary` | >=2.9 | PostgreSQL driver (install separately) |

```bash
# Install PostgreSQL driver
uv pip install psycopg2-binary

# Or for production (requires libpq-dev)
uv pip install psycopg2
```

---

## Container Requirements

### Docker

For containerized deployments:

| Requirement | Version | Notes |
|-------------|---------|-------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Multi-container orchestration |

### Kubernetes

For Kubernetes deployments:

| Requirement | Notes |
|-------------|-------|
| Kubernetes | 1.24+ |
| Persistent volumes | For audit database and payload store |
| Secrets management | For API keys and fingerprint key |

---

## Verification

### Check Python Version

```bash
python --version
# Should output: Python 3.12.x or higher
```

### Check Installation

```bash
# After installation
elspeth --version
elspeth health --verbose
```

### Verify Dependencies

```bash
# List installed packages
uv pip list | grep -E "^(typer|pydantic|sqlalchemy|pandas)"
```

### Run Tests

```bash
# Quick smoke test
.venv/bin/python -m pytest tests/ -v --tb=short -x

# Full test suite
.venv/bin/python -m pytest tests/ -v
```

---

## Compatibility Notes

### Python Version

ELSPETH requires Python 3.12+ for:
- Modern type hints (`type` statement, improved generics)
- Performance improvements in the interpreter
- Better error messages

Python 3.11 is **not supported** due to missing language features used in the codebase.

### Platform Support

| Platform | Support Level | Notes |
|----------|--------------|-------|
| Linux (Ubuntu 22.04+) | Full | Recommended for production |
| Linux (other distros) | Full | Tested on Debian, RHEL |
| macOS (12+) | Full | Development supported |
| Windows (10+) | Partial | Works, but Linux recommended |

### Known Limitations

1. **TUI on Windows**: The Textual TUI may have rendering issues on older Windows Terminal versions
2. **SQLite on NFS**: Avoid SQLite on network filesystems; use PostgreSQL instead
3. **Apple Silicon**: Fully supported via native ARM Python

---

## See Also

- [README.md](README.md) - Quick start installation
- [Docker Guide](docs/guides/docker.md) - Container deployment
- [Configuration Reference](docs/reference/configuration.md) - Full configuration options
