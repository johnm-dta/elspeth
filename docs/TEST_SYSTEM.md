# ELSPETH Test System Documentation

**Last Updated:** 2026-01-20

This document describes ELSPETH's comprehensive test system, designed to ensure the audit trail is "beyond reproach" for a system where "I don't know what happened" is never acceptable.

---

## Overview

ELSPETH's test regime validates:
1. **Determinism** - Same input always produces same output/hash
2. **Trust Model Enforcement** - Three-Tier Trust Model is correctly applied
3. **Plugin Contracts** - All plugins honor their interfaces
4. **Audit Integrity** - Every decision is traceable to source data

## Test Pyramid Architecture

```
tests/
├── core/                     # Core subsystem unit tests
├── integration/              # Component boundary tests
├── property/                 # Hypothesis property-based tests
│   ├── canonical/            # Hash determinism tests
│   ├── landscape/            # Audit trail property tests
│   └── trust_model/          # Trust tier enforcement tests
├── contracts/                # Plugin interface contract tests
│   ├── sources/
│   ├── transforms/
│   └── sinks/
└── e2e/                      # End-to-end pipeline tests
```

### Test Distribution

| Level | Purpose | Speed | Count |
|-------|---------|-------|-------|
| **Unit Tests** | Isolated component logic | <1ms each | ~1,500 |
| **Integration Tests** | Component boundaries | 10-500ms | ~200 |
| **Property Tests** | Invariant verification | 1-10s per property | ~500 examples/property |
| **Contract Tests** | Plugin interface validation | 10-100ms | Per plugin |
| **E2E Tests** | Full pipeline scenarios | 1-30s | ~50 |

---

## Property-Based Testing

Property-based testing uses [Hypothesis](https://hypothesis.readthedocs.io/) to generate thousands of random inputs and verify that properties hold for ALL of them, not just hand-picked examples.

### Why Property-Based Testing is Critical

ELSPETH's audit integrity depends on properties that must hold universally:

| Property | Consequence if Violated |
|----------|------------------------|
| Hash determinism | Audit trail meaningless - same data produces different hashes |
| Canonical JSON stability | Historical hashes become invalid after upgrades |
| Enum coercion | Invalid values silently accepted or valid values rejected |
| Lineage completeness | "I don't know what happened" becomes possible |

### Running Property Tests

```bash
# Run all property tests
pytest tests/property/ -v

# Run with more examples (thorough mode)
HYPOTHESIS_PROFILE=nightly pytest tests/property/ -v

# Run specific property test file
pytest tests/property/canonical/test_hash_determinism.py -v --hypothesis-show-statistics
```

### Key Property Test Files

| File | Properties Tested |
|------|-------------------|
| `tests/property/canonical/test_hash_determinism.py` | JSON determinism, hash stability, key sorting, NaN rejection |
| `tests/property/landscape/test_enum_coercion.py` | All enum types coerce correctly |
| `tests/property/trust_model/test_tier_enforcement.py` | Trust tier boundaries |

---

## Contract Testing

Contract tests verify that all plugins honor their protocol interfaces. Unlike unit tests that test implementation, contract tests verify **behavioral contracts**.

### Plugin Contracts

| Plugin Type | Contract Requirements |
|-------------|----------------------|
| **Source** | `load()` yields `SourceRow`, quarantined rows have reasons |
| **Transform** | `process()` returns `TransformResult`, errors have details |
| **Sink** | `write()` returns `ArtifactDescriptor` with deterministic hash |

### Running Contract Tests

```bash
# Run all contract tests
pytest tests/contracts/ -v

# Run specific plugin contract
pytest tests/contracts/sources/test_csv_source_contract.py -v
```

---

## Mutation Testing

Mutation testing validates that tests actually catch bugs, not just execute code. It introduces artificial bugs (mutants) and checks if tests catch them.

### Why Mutation Testing Matters

100% code coverage doesn't mean tests are effective:

```python
# This has 100% coverage but catches NOTHING
def test_calculate_tax():
    calculate_tax(100)  # Executes code, asserts nothing
```

Mutation testing reveals these weak tests by modifying code and checking if tests fail.

### Configuration

Mutation testing is configured in `pyproject.toml`:

```toml
[tool.mutmut]
paths_to_mutate = "src/elspeth/core/"
runner = "python -m pytest tests/property/ tests/core/ -x --tb=no -q"
tests_dir = "tests/"
backup = false
```

### Target Mutation Scores

| Subsystem | Target | Rationale |
|-----------|--------|-----------|
| **canonical.py** | 95%+ | Hash integrity is foundational |
| **landscape/** | 90%+ | Audit trail is the legal record |
| **engine/** | 85%+ | Orchestration correctness |
| **plugins/** | 80%+ | Extension points |

### Running Mutation Tests

```bash
# Run mutation testing on canonical.py (critical module)
python -m mutmut run --paths-to-mutate src/elspeth/core/canonical.py

# View results
python -m mutmut results

# Run on landscape recorder
python -m mutmut run --paths-to-mutate src/elspeth/core/landscape/recorder.py

# Show specific survived mutant
python -m mutmut show 42
```

### Interpreting Results

| Status | Meaning | Action |
|--------|---------|--------|
| 🎉 **Killed** | Test caught the bug | Good - no action needed |
| 🙁 **Survived** | Tests missed the bug | Investigate and add assertion |
| ⏰ **Timeout** | Test took 10x baseline | May indicate infinite loop mutation |
| 🤔 **Suspicious** | Tests slow but passed | Investigate performance |

### CI Integration

Mutation testing runs weekly (Sunday 2 AM UTC) via `.github/workflows/mutation-testing.yaml`:
- Runs on `canonical.py` (mandatory) and `landscape/recorder.py` (scheduled only)
- Results uploaded as artifacts for analysis
- Can be triggered manually via `workflow_dispatch`

---

## Running Tests

### Quick Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/elspeth --cov-report=term-missing

# Run specific test category
pytest tests/core/ -v              # Unit tests
pytest tests/integration/ -v       # Integration tests
pytest tests/property/ -v          # Property tests
pytest tests/contracts/ -v         # Contract tests

# Run fast tests only (exclude slow)
pytest tests/ -v -m "not slow"
```

### Coverage Requirements

| Scope | Target | Threshold |
|-------|--------|-----------|
| Overall | 85% | 80% |
| New code | 95% | 90% |
| Core subsystems | 90% | 85% |

### Test Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.slow` | Tests >5s, skipped in fast runs |
| `@pytest.mark.integration` | Requires database or filesystem |
| `@pytest.mark.asyncio` | Async test (handled by pytest-asyncio) |

---

## Continuous Integration

### Pull Request Checks

Every PR runs:
1. **Unit & Integration Tests** - Full test suite
2. **Property Tests** - 100 examples per property
3. **Contract Tests** - All plugin contracts
4. **Type Checking** - mypy strict mode
5. **Linting** - ruff

### Weekly Scheduled Jobs

| Job | Schedule | Purpose |
|-----|----------|---------|
| **Mutation Testing** | Sunday 2 AM | Validate test effectiveness |
| **Property Tests (thorough)** | TBD | 1000+ examples per property |

---

## Adding New Tests

### Property Test Template

```python
from hypothesis import given, settings
from hypothesis import strategies as st

class TestMyModuleProperties:
    """Property tests for my_module determinism."""

    @given(data=st.dictionaries(st.text(), st.integers()))
    @settings(max_examples=500)
    def test_my_function_is_deterministic(self, data):
        """Property: my_function(x) == my_function(x) for all inputs."""
        result1 = my_function(data)
        result2 = my_function(data)
        assert result1 == result2
```

### Contract Test Template

```python
class TestMyPluginContract:
    """Contract tests for MyPlugin."""

    @pytest.fixture
    def plugin(self):
        return MyPlugin(config={})

    def test_process_returns_correct_type(self, plugin, ctx):
        """Contract: process() MUST return TransformResult."""
        result = plugin.process({"data": "test"}, ctx)
        assert isinstance(result, TransformResult)

    def test_error_has_details(self, plugin, ctx):
        """Contract: Error results MUST have details."""
        result = plugin.process(None, ctx)
        if result.is_error:
            assert result.error_details is not None
```

---

## Baseline Mutation Scores (2026-01-20)

| Module | Mutants | Killed | Survived | Score |
|--------|---------|--------|----------|-------|
| `canonical.py` | 27 | 27 | 0 | **100%** |
| `landscape/recorder.py` | 322 | TBD | TBD | Running... |

---

## Quality Philosophy

> The audit trail is a legal record. Silently coercing bad data is evidence tampering. If an auditor asks "why did row 42 get routed here?" and we give a confident wrong answer because we coerced garbage into a valid-looking value, we've committed fraud.
>
> — ELSPETH Design Manifesto

This test regime ensures that "I don't know what happened" remains an impossible answer.
