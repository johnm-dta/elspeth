# ELSPETH World-Class Test Regime Proposal

**Status:** Proposal
**Date:** 2026-01-20
**Objective:** Define a test regime that is "beyond reproach" for an auditable system where "I don't know what happened" is never acceptable.

---

## Executive Summary

ELSPETH's audit trail is a **legal record**. This test regime must validate that:
1. Every decision is traceable to source data, configuration, and code version
2. Hashes are deterministic and survive payload deletion
3. The Three-Tier Trust Model is correctly enforced
4. Plugin contracts are honored
5. Recovery from failures maintains audit integrity

**Current State:**
- 114 test files, 1,690 test functions, ~36K lines of test code
- Good unit test coverage of core subsystems
- Limited property-based testing (2 files use Hypothesis)
- No mutation testing
- No formal contract testing for plugin interfaces
- No chaos engineering

**Target State:**
- Comprehensive property-based testing for all determinism-critical paths
- Mutation testing validating test effectiveness (target: 80%+ mutation score)
- Contract testing for plugin interfaces
- Formal quality gates with actionable metrics
- Resilience testing for recovery scenarios

---

## Part 1: Test Pyramid Architecture

### Target Distribution

| Level | Current | Target | Rationale |
|-------|---------|--------|-----------|
| **Unit Tests** | ~1,500 | 1,800 | Focus on property-based additions |
| **Integration Tests** | ~190 | 300 | Expand Landscape/Engine boundary testing |
| **System Tests** | ~8 | 50 | Full pipeline scenarios with audit verification |
| **Property-Based Tests** | ~50 | 500+ | Critical for determinism verification |
| **Mutation Score** | Unknown | 80%+ | Validate test effectiveness |

### Test Organization

```
tests/
├── unit/                     # Isolated component tests (fast, <1ms each)
│   ├── core/                 # Canonical, DAG, Config
│   ├── landscape/            # Recorder, Models, Repositories
│   ├── engine/               # RowProcessor, Executors, Tokens
│   └── plugins/              # Base classes, Schemas, Results
├── integration/              # Component boundary tests (10-500ms each)
│   ├── landscape_db/         # Recorder + SQLite
│   ├── pipeline/             # Engine + Plugins + Landscape
│   └── checkpoint/           # Checkpoint + Recovery + Payloads
├── system/                   # Full pipeline tests (<30s each)
│   ├── audit_verification/   # Explain queries, lineage completeness
│   └── recovery/             # Crash recovery, resume scenarios
├── property/                 # Hypothesis property-based tests
│   ├── canonical/            # Hash determinism
│   ├── trust_model/          # Trust tier enforcement
│   └── state_machines/       # Stateful testing for Landscape
├── contracts/                # Plugin interface contract tests
│   ├── source_contracts/
│   ├── transform_contracts/
│   └── sink_contracts/
└── chaos/                    # Resilience tests (staging only)
    ├── database_failures/
    └── payload_store_failures/
```

---

## Part 2: Property-Based Testing Strategy

### Why Property-Based Testing is Critical for ELSPETH

ELSPETH's audit integrity depends on **properties that must hold for ALL inputs**, not just the examples we think of:

| Property | Why It Matters |
|----------|----------------|
| Hash determinism | Same data = same hash, always. Non-determinism = audit fraud. |
| Canonical JSON stability | Version upgrades must not change hashes |
| Enum coercion | Valid enum strings coerce correctly, invalid strings crash |
| Lineage completeness | Every terminal state has complete lineage |
| Recovery idempotence | Resume produces same result as uninterrupted run |

### Mandatory Property Tests

#### 1. Canonical JSON Determinism (P0 - HIGHEST)

```python
# tests/property/canonical/test_hash_determinism.py
from hypothesis import given, settings, assume
from hypothesis.strategies import (
    dictionaries, lists, integers, floats, text,
    booleans, none, recursive, sampled_from
)
import math

# Strategy for JSON-safe values (excluding NaN/Infinity)
json_primitives = (
    none() |
    booleans() |
    integers() |
    floats(allow_nan=False, allow_infinity=False) |
    text()
)

json_values = recursive(
    json_primitives,
    lambda children: (
        lists(children) |
        dictionaries(text(), children)
    ),
    max_leaves=50
)

@given(data=json_values)
@settings(max_examples=1000)
def test_canonical_json_deterministic(data):
    """Property: canonical_json(x) == canonical_json(x) for all valid inputs."""
    result1 = canonical_json(data)
    result2 = canonical_json(data)
    assert result1 == result2, f"Non-deterministic: {result1} != {result2}"

@given(data=json_values)
@settings(max_examples=1000)
def test_stable_hash_deterministic(data):
    """Property: stable_hash(x) == stable_hash(x) for all valid inputs."""
    hash1 = stable_hash(data)
    hash2 = stable_hash(data)
    assert hash1 == hash2

@given(data=json_values)
def test_canonical_json_sorts_keys(data):
    """Property: Keys are always sorted in output."""
    if isinstance(data, dict) and len(data) > 1:
        result = canonical_json(data)
        # Keys should appear in sorted order
        keys = list(data.keys())
        for i in range(len(keys) - 1):
            pos_a = result.find(f'"{keys[i]}"')
            pos_b = result.find(f'"{keys[i+1]}"')
            if pos_a != -1 and pos_b != -1:
                assert (pos_a < pos_b) == (keys[i] < keys[i+1])
```

#### 2. NaN/Infinity Rejection (P0 - HIGHEST)

```python
@given(floats(allow_nan=True, allow_infinity=True))
def test_nan_infinity_rejected(value):
    """Property: NaN and Infinity MUST raise ValueError."""
    if math.isnan(value) or math.isinf(value):
        with pytest.raises(ValueError, match="Cannot canonicalize non-finite"):
            canonical_json(value)
    else:
        # Valid floats should work
        result = canonical_json(value)
        assert isinstance(result, str)
```

#### 3. Enum Coercion Correctness (P0 - HIGHEST)

```python
from hypothesis import given
from hypothesis.strategies import sampled_from, text

@given(status=sampled_from(list(RunStatus)))
def test_run_status_enum_roundtrip(status):
    """Property: Enum → string → Enum is identity."""
    string_value = status.value
    recovered = RunStatus(string_value)
    assert recovered == status

@given(invalid=text().filter(lambda s: s not in [e.value for e in RunStatus]))
def test_invalid_enum_string_crashes(invalid):
    """Property: Invalid enum strings MUST raise ValueError."""
    assume(len(invalid) < 100)  # Reasonable string length
    with pytest.raises(ValueError):
        RunStatus(invalid)
```

#### 4. Lineage Completeness (P0 - HIGHEST)

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

class LandscapeStateMachine(RuleBasedStateMachine):
    """Stateful test for Landscape audit trail integrity."""

    def __init__(self):
        super().__init__()
        self.db = LandscapeDB.in_memory()
        self.recorder = LandscapeRecorder(self.db)
        self.active_run = None
        self.tokens = []
        self.node_states = []

    @rule()
    def begin_run(self):
        """Start a new run."""
        if self.active_run is None:
            self.active_run = self.recorder.begin_run(
                pipeline_name="test",
                config={"test": True},
                config_hash="abc123"
            )

    @rule(row_data=json_values)
    def create_token(self, row_data):
        """Create a token from source data."""
        if self.active_run:
            token = self.recorder.create_token(
                run_id=self.active_run.run_id,
                source_node_id="source_1",
                row_index=len(self.tokens),
                row_data=row_data
            )
            self.tokens.append(token)

    @invariant()
    def lineage_always_complete(self):
        """Invariant: Every token has complete lineage."""
        for token in self.tokens:
            lineage = explain(self.db, self.active_run.run_id, token.token_id)
            # Lineage must include source row
            assert lineage.source_row is not None
            # All node states must be present
            assert len(lineage.node_states) >= 0

TestLandscapeStateMachine = LandscapeStateMachine.TestCase
```

#### 5. Three-Tier Trust Model Enforcement

```python
# tests/property/trust_model/test_tier_enforcement.py

@given(
    null_field=sampled_from(['output_hash', 'completed_at', 'duration_ms']),
    status=sampled_from([NodeStateStatus.COMPLETED])
)
def test_tier1_crashes_on_null_required_fields(null_field, status):
    """Property: Tier 1 (audit data) crashes on NULL required fields."""
    # Create a mock row with NULL in required field
    row = MagicMock()
    row.status = status.value
    row.output_hash = "abc" if null_field != 'output_hash' else None
    row.completed_at = datetime.now() if null_field != 'completed_at' else None
    row.duration_ms = 100 if null_field != 'duration_ms' else None

    with pytest.raises(ValueError, match="audit integrity violation"):
        _row_to_node_state(row)
```

### Property Test Coverage Targets

| Subsystem | Properties to Test | Target Examples |
|-----------|-------------------|-----------------|
| **Canonical** | Determinism, NaN rejection, key sorting | 1,000+ |
| **Landscape** | Enum coercion, lineage completeness, state transitions | 500+ |
| **Engine** | Token lifecycle, fork/coalesce invariants | 500+ |
| **Config** | Reserved label detection, trigger validation | 200+ |
| **Plugins** | Schema validation, result type correctness | 200+ |

---

## Part 3: Contract Testing for Plugin System

### Plugin Interface Contracts

ELSPETH plugins are **system-owned code** with strict contracts. Contract tests verify:

1. **Source Contract:** `load()` yields `SourceRow` objects with correct lifecycle
2. **Transform Contract:** `process()` returns `TransformResult` with correct types
3. **Sink Contract:** `write()` returns `ArtifactDescriptor` with content hash

### Contract Test Structure

```python
# tests/contracts/source_contracts/test_source_protocol.py

class SourceContractTest:
    """Base class for source contract verification."""

    @pytest.fixture
    def source(self) -> SourceProtocol:
        """Subclasses provide concrete source."""
        raise NotImplementedError

    def test_load_yields_source_rows(self, source, ctx):
        """Contract: load() MUST yield SourceRow objects."""
        for row in source.load(ctx):
            assert isinstance(row, SourceRow)

    def test_valid_rows_have_data(self, source, ctx):
        """Contract: Valid rows MUST have non-None data."""
        for row in source.load(ctx):
            if row.is_valid:
                assert row.data is not None

    def test_quarantined_rows_have_reason(self, source, ctx):
        """Contract: Quarantined rows MUST have reason."""
        for row in source.load(ctx):
            if not row.is_valid:
                assert row.quarantine_reason is not None

    def test_close_is_idempotent(self, source, ctx):
        """Contract: close() can be called multiple times safely."""
        list(source.load(ctx))  # Exhaust iterator
        source.close()
        source.close()  # Should not raise


class TestCSVSourceContract(SourceContractTest):
    """Contract tests for CSVSource."""

    @pytest.fixture
    def source(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob")
        return CSVSource(path=str(csv_file))
```

### Transform Contract Tests

```python
# tests/contracts/transform_contracts/test_transform_protocol.py

class TransformContractTest:
    """Base class for transform contract verification."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        raise NotImplementedError

    @given(row=json_values)
    def test_process_returns_transform_result(self, transform, row, ctx):
        """Contract: process() MUST return TransformResult."""
        result = transform.process(row, ctx)
        assert isinstance(result, TransformResult)

    def test_success_result_has_data(self, transform, ctx):
        """Contract: Success results MUST have output data."""
        result = transform.process({"valid": "data"}, ctx)
        if result.is_success:
            assert result.data is not None

    def test_error_result_has_details(self, transform, ctx):
        """Contract: Error results MUST have error details."""
        # Trigger error condition
        result = transform.process({"invalid": None}, ctx)
        if result.is_error:
            assert result.error_details is not None
```

### Sink Contract Tests

```python
# tests/contracts/sink_contracts/test_sink_protocol.py

class SinkContractTest:
    """Base class for sink contract verification."""

    @pytest.fixture
    def sink(self) -> SinkProtocol:
        raise NotImplementedError

    def test_write_returns_artifact_descriptor(self, sink, ctx):
        """Contract: write() MUST return ArtifactDescriptor."""
        result = sink.write([{"id": 1}], ctx)
        assert isinstance(result, ArtifactDescriptor)

    def test_artifact_has_content_hash(self, sink, ctx):
        """Contract: ArtifactDescriptor MUST have content_hash."""
        result = sink.write([{"id": 1}], ctx)
        assert result.content_hash is not None
        assert len(result.content_hash) == 64  # SHA-256 hex

    def test_flush_is_idempotent(self, sink, ctx):
        """Contract: flush() can be called multiple times."""
        sink.write([{"id": 1}], ctx)
        sink.flush()
        sink.flush()  # Should not raise

    def test_content_hash_is_deterministic(self, sink, ctx):
        """Contract: Same data = same content_hash."""
        data = [{"id": 1, "name": "test"}]
        result1 = sink.write(data, ctx)
        sink.flush()
        result2 = sink.write(data, ctx)
        assert result1.content_hash == result2.content_hash
```

---

## Part 4: Mutation Testing

### Why Mutation Testing is Essential

100% code coverage doesn't mean tests are effective. Mutation testing validates that tests actually catch bugs:

```python
# Example: This test has 100% coverage but 0% effectiveness
def test_calculate_tax():
    calculate_tax(100)  # Executes code but asserts NOTHING
```

### Configuration

```toml
# setup.cfg
[mutmut]
paths_to_mutate = src/elspeth/
backup = False
runner = python -m pytest -x tests/unit tests/integration
tests_dir = tests/
```

### Target Mutation Scores by Subsystem

| Subsystem | Target Score | Priority | Rationale |
|-----------|--------------|----------|-----------|
| **Canonical** | 95%+ | P0 | Hash integrity is foundational |
| **Landscape** | 90%+ | P0 | Audit trail is the legal record |
| **Engine** | 85%+ | P0 | Orchestration must be correct |
| **Plugins** | 80%+ | P1 | Extension points |
| **Config** | 80%+ | P1 | Validation logic |
| **CLI/TUI** | 70%+ | P2 | User interface |

### CI Integration

```yaml
# .github/workflows/mutation-testing.yml
name: Mutation Testing

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly Sunday 2 AM
  workflow_dispatch:

jobs:
  mutmut:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[dev]"
          pip install mutmut

      - name: Run mutation testing (core)
        run: |
          mutmut run --paths-to-mutate src/elspeth/core/canonical.py
          mutmut results

      - name: Check mutation score
        run: |
          SCORE=$(mutmut results | grep "Mutation score" | awk '{print $NF}')
          if (( $(echo "$SCORE < 80" | bc -l) )); then
            echo "Mutation score $SCORE% below 80% threshold"
            exit 1
          fi
```

---

## Part 5: Quality Metrics & KPIs

### Actionable Metrics Dashboard

| Metric | Target | Alert Threshold | Action |
|--------|--------|-----------------|--------|
| **Test Pass Rate** | >99% | <98% | Fix failing tests immediately |
| **Flakiness Rate** | <0.5% | >2% | Run flaky test prevention |
| **Unit Test Coverage** | >85% | <80% | Add tests for uncovered code |
| **New Code Coverage** | >95% | <90% | Block PR until coverage met |
| **Mutation Score (Core)** | >90% | <85% | Improve test assertions |
| **Build Time (PR)** | <15min | >20min | Parallelize or optimize |
| **Integration Test Time** | <5min | >10min | Optimize fixtures |

### Quality Gates

```yaml
# .github/workflows/quality-gates.yml
name: Quality Gates

on: [pull_request]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run tests with coverage
        run: |
          pytest --cov=src/elspeth --cov-report=json --cov-fail-under=80

      - name: Check new code coverage
        run: |
          # Extract coverage for changed files only
          CHANGED=$(git diff --name-only origin/main...HEAD | grep '\.py$' | tr '\n' ' ')
          pytest --cov=src/elspeth --cov-report=term-missing $CHANGED
          # Fail if new code coverage < 95%

      - name: Verify property tests pass
        run: |
          pytest tests/property/ -v --hypothesis-show-statistics

      - name: Run contract tests
        run: |
          pytest tests/contracts/ -v
```

### Weekly Quality Report Template

```markdown
# ELSPETH Quality Report - Week of YYYY-MM-DD

## Summary
| Metric | This Week | Last Week | Trend |
|--------|-----------|-----------|-------|
| Test Pass Rate | X% | Y% | ↑/↓ |
| Flakiness Rate | X% | Y% | ↑/↓ |
| Coverage | X% | Y% | ↑/↓ |
| Mutation Score | X% | Y% | ↑/↓ |
| Build Time | Xm | Ym | ↑/↓ |

## Property Test Statistics
- Total properties: N
- Examples generated: M
- Shrunk failures: K

## Actions Taken
- [Description of fixes/improvements]

## Action Items
- [ ] [Next steps]
```

---

## Part 6: Integration Test Hardening

### Database Isolation Pattern

```python
# tests/integration/conftest.py

@pytest.fixture(scope="function")
def landscape_db():
    """Each test gets isolated in-memory database."""
    db = LandscapeDB.in_memory()
    yield db
    # Automatic cleanup - in-memory DB is discarded

@pytest.fixture(scope="function")
def recorder(landscape_db):
    """Recorder with isolated database."""
    return LandscapeRecorder(landscape_db)

@pytest.fixture(scope="function")
def payload_store(tmp_path):
    """Isolated payload store per test."""
    store = FilesystemPayloadStore(tmp_path / "payloads")
    yield store
    # tmp_path is automatically cleaned up
```

### Critical Integration Test Scenarios

| Scenario | Tests | Priority |
|----------|-------|----------|
| **Run Lifecycle** | begin → nodes → edges → tokens → complete | P0 |
| **Token Forking** | create → fork → child tokens → coalesce | P0 |
| **Aggregation** | buffer → trigger → flush → batch record | P0 |
| **Error Recording** | validation error, transform error, routing | P0 |
| **Checkpoint/Resume** | checkpoint → crash simulation → resume | P0 |
| **Lineage Query** | explain() returns complete lineage | P0 |
| **Payload Purge** | purge → graceful degradation | P1 |
| **Schema Compatibility** | old DB → new code | P1 |

---

## Part 7: Chaos Engineering (Future Phase)

### Prerequisites Before Chaos

- [ ] Comprehensive monitoring (metrics, logs, traces)
- [ ] Automated rollback capability
- [ ] Baseline metrics documented
- [ ] Staging environment ready

### Initial Chaos Experiments (Staging Only)

| Experiment | Hypothesis | Blast Radius |
|------------|------------|--------------|
| **SQLite WAL Failure** | Landscape gracefully degrades on write failure | 100% staging |
| **Payload Store Unavailable** | RowDataResult returns STORE_NOT_CONFIGURED | 100% staging |
| **Checkpoint Corruption** | Recovery manager detects and reports | Single run |
| **Enum Value Unknown** | Recorder crashes with clear error | Single row |

### Chaos Test Structure

```python
# tests/chaos/test_database_failures.py

class TestDatabaseFailures:
    """Chaos tests for database resilience."""

    @pytest.fixture
    def failing_db(self, monkeypatch):
        """Database that fails on write after N operations."""
        db = LandscapeDB.in_memory()
        original_execute = db.connection.execute
        call_count = [0]

        def failing_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 10:
                raise sqlite3.OperationalError("disk I/O error")
            return original_execute(*args, **kwargs)

        monkeypatch.setattr(db.connection, "execute", failing_execute)
        return db

    def test_recorder_reports_write_failure(self, failing_db):
        """Hypothesis: Write failures are reported, not silently swallowed."""
        recorder = LandscapeRecorder(failing_db)
        run = recorder.begin_run(...)

        # Should eventually fail with clear error
        with pytest.raises(sqlite3.OperationalError):
            for i in range(100):
                recorder.create_token(...)
```

---

## Part 8: Implementation Roadmap

### Phase 1: Property-Based Testing Foundation (Week 1-2)

- [ ] Add Hypothesis to dev dependencies
- [ ] Create `tests/property/` directory structure
- [ ] Implement canonical JSON property tests (1,000+ examples)
- [ ] Implement enum coercion property tests
- [ ] Add property tests to CI pipeline

### Phase 2: Contract Testing (Week 2-3)

- [ ] Create `tests/contracts/` directory structure
- [ ] Implement source contract base class and tests
- [ ] Implement transform contract base class and tests
- [ ] Implement sink contract base class and tests
- [ ] Run contract tests against all existing plugins

### Phase 3: Mutation Testing (Week 3-4)

- [ ] Install and configure mutmut
- [ ] Run baseline mutation testing on core subsystems
- [ ] Fix tests with low mutation scores
- [ ] Add mutation testing to weekly CI

### Phase 4: Quality Gates (Week 4-5)

- [ ] Configure coverage thresholds in CI
- [ ] Add new code coverage gate (95%)
- [ ] Create quality dashboard
- [ ] Establish weekly quality report process

### Phase 5: Integration Test Hardening (Week 5-6)

- [ ] Implement all critical integration scenarios
- [ ] Add Landscape state machine tests
- [ ] Verify checkpoint/resume with property tests
- [ ] Add schema compatibility tests

### Phase 6: Chaos Engineering Prep (Week 7-8)

- [ ] Set up staging environment
- [ ] Document baseline metrics
- [ ] Implement first chaos experiment (SQLite failure)
- [ ] Create chaos runbook

---

## Appendix A: Test File Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Unit Test | `test_<module>.py` | `test_canonical.py` |
| Integration | `test_<component>_integration.py` | `test_landscape_integration.py` |
| Property | `test_<module>_properties.py` | `test_canonical_properties.py` |
| Contract | `test_<plugin>_contract.py` | `test_csv_source_contract.py` |
| Chaos | `test_<failure>_chaos.py` | `test_sqlite_failure_chaos.py` |

## Appendix B: Hypothesis Settings by Test Type

```python
# conftest.py

from hypothesis import settings, Verbosity

# Fast tests for CI (default)
settings.register_profile("ci", max_examples=100)

# Thorough tests for nightly
settings.register_profile("nightly", max_examples=1000)

# Debug mode for investigating failures
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

# Load profile from environment
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))
```

## Appendix C: Quality Thresholds Reference

| Metric | Acceptable | Good | Excellent |
|--------|------------|------|-----------|
| Pass Rate | 95-98% | 98-99% | >99% |
| Flakiness | 2-5% | 0.5-2% | <0.5% |
| Coverage | 70-80% | 80-90% | >90% |
| Mutation Score | 60-70% | 70-85% | >85% |
| Build Time | 20-30min | 10-20min | <10min |

---

## Conclusion

This test regime addresses ELSPETH's unique requirements as an auditable system:

1. **Property-based testing** validates determinism properties that must hold for ALL inputs
2. **Contract testing** ensures plugins honor their interfaces
3. **Mutation testing** validates that tests actually catch bugs
4. **Quality gates** enforce standards before merge
5. **Chaos engineering** (future) validates resilience

The audit trail is a legal record. This test regime ensures that "I don't know what happened" remains an impossible answer.
