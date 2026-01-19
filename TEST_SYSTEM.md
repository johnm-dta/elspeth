# ELSPETH Test System

This document describes the testing architecture for ELSPETH, designed to ensure the audit trail is **beyond reproach**.

## Testing Philosophy

ELSPETH's test system is built on three principles aligned with the Three-Tier Trust Model:

1. **Audit Integrity is Non-Negotiable**: The Landscape audit trail must be 100% pristine. Tests verify that bad data crashes immediately rather than silently corrupting the record.

2. **Properties Over Examples**: Example-based tests prove specific cases work. Property-based tests prove invariants hold for ALL valid inputs. For audit systems, "same input = same hash" must hold universally.

3. **Defense in Depth**: Multiple test layers catch different failure modes - unit tests for logic, property tests for invariants, integration tests for subsystem interaction.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures, Hypothesis configuration
├── unit/                       # Traditional unit tests
├── integration/                # Subsystem integration tests
├── contracts/                  # Contract tests for plugin interfaces
│   ├── source_contracts/       # Source protocol contract tests
│   │   ├── test_source_protocol.py
│   │   └── test_csv_source_contract.py
│   ├── transform_contracts/    # Transform protocol contract tests
│   │   ├── test_transform_protocol.py
│   │   └── test_passthrough_contract.py
│   └── sink_contracts/         # Sink protocol contract tests
│       ├── test_sink_protocol.py
│       └── test_csv_sink_contract.py
└── property/                   # Property-based tests (Hypothesis)
    ├── canonical/              # Canonical JSON/hashing invariants
    │   ├── test_hash_determinism.py
    │   └── test_nan_rejection.py
    └── contracts/              # Contract and enum invariants
        └── test_enum_coercion.py
```

## Running Tests

### Quick Reference

```bash
# Run all tests
.venv/bin/python -m pytest tests/

# Run only property tests
.venv/bin/python -m pytest tests/property/

# Run with verbose output
.venv/bin/python -m pytest tests/property/ -v

# Run specific test file
.venv/bin/python -m pytest tests/property/canonical/test_hash_determinism.py
```

### Hypothesis Profiles

Property tests use [Hypothesis](https://hypothesis.readthedocs.io/) with configurable profiles:

| Profile | Examples | Use Case |
|---------|----------|----------|
| `ci` (default) | 100 | Fast CI feedback (~15 seconds) |
| `nightly` | 1000 | Thorough overnight/scheduled runs |
| `debug` | 10 | Quick iteration with verbose output |

```bash
# Use nightly profile for thorough testing
HYPOTHESIS_PROFILE=nightly .venv/bin/python -m pytest tests/property/

# Use debug profile when investigating failures
HYPOTHESIS_PROFILE=debug .venv/bin/python -m pytest tests/property/ -v
```

## Property-Based Tests

### Why Property Testing?

Traditional example tests verify specific cases:
```python
def test_hash_specific_value():
    assert stable_hash({"a": 1}) == "expected_hash"
```

Property tests verify invariants hold for ALL inputs:
```python
@given(data=json_values)
def test_hash_is_deterministic(data):
    assert stable_hash(data) == stable_hash(data)  # Must hold for ANY data
```

Hypothesis generates thousands of random inputs, including edge cases humans wouldn't think of, and when a test fails, it automatically shrinks to the minimal failing example.

### Canonical JSON Determinism (`test_hash_determinism.py`)

Verifies the foundational property of ELSPETH's audit trail: **same input MUST always produce the same hash**.

| Test Class | Properties Verified |
|------------|---------------------|
| `TestCanonicalJsonDeterminism` | `canonical_json(x) == canonical_json(x)` for all valid inputs; output is valid JSON; keys are sorted |
| `TestStableHashDeterminism` | `stable_hash(x) == stable_hash(x)` for all valid inputs; returns valid SHA-256 hex; different data → different hash |
| `TestPandasNumpyNormalization` | `numpy.int64`, `numpy.float64`, `numpy.bool_`, `pandas.Timestamp`, `pd.NaT`, `pd.NA` all hash deterministically and consistently with Python equivalents |
| `TestSpecialTypes` | `bytes`, `Decimal`, `datetime` hash deterministically; naive datetime treated as UTC |
| `TestStructuralProperties` | No unnecessary whitespace; dict hash independent of key insertion order; list hash depends on element order |

**Key Constraint**: Integer strategies are bounded to ±(2^53-1) for RFC 8785/JCS compatibility (JavaScript-safe integers).

### NaN/Infinity Rejection (`test_nan_rejection.py`)

Verifies that non-finite floats are **strictly rejected**, not silently converted. This is defense-in-depth for audit integrity.

| Test Class | Properties Verified |
|------------|---------------------|
| `TestNaNRejection` | Python `float('nan')`, NumPy `np.nan` raise `ValueError` in `canonical_json()` and `stable_hash()`; nested NaN in dicts/lists rejected |
| `TestInfinityRejection` | Both `+inf` and `-inf` (Python and NumPy) raise `ValueError`; nested infinity rejected |
| `TestNonFiniteEdgeCases` | NaN's self-inequality doesn't bypass check; mixed valid/invalid data rejected; non-finite in numpy arrays rejected |
| `TestValidFloatsAccepted` | All finite floats accepted; zero (including -0.0) accepted; edge floats (max, min positive) accepted |

**Why This Matters**: NaN in audit data means undefined comparison behavior. Silent conversion to `null` or `0` would be data corruption in the legal record.

### Enum Coercion (`test_enum_coercion.py`)

Verifies that database-stored enums follow Tier 1 (Full Trust) rules: **invalid values MUST crash, not silently coerce**.

| Enum | Storage | Test Coverage |
|------|---------|---------------|
| `RunStatus` | Database | Roundtrip, valid string coercion, invalid string rejection |
| `NodeStateStatus` | Database | Roundtrip, invalid string rejection |
| `BatchStatus` | Database | Roundtrip, invalid string rejection |
| `Determinism` | Database | Roundtrip, invalid string rejection (critical - undeclared = crash) |
| `NodeType` | Database | Roundtrip, invalid string rejection |
| `RoutingKind` | Database | Roundtrip, invalid string rejection |
| `RoutingMode` | Database | Roundtrip, invalid string rejection |
| `TriggerType` | Database | Roundtrip, invalid string rejection |
| `CallType` | Database | Roundtrip, invalid string rejection |
| `CallStatus` | Database | Roundtrip, invalid string rejection |
| `ExportStatus` | Database | Roundtrip, invalid string rejection |
| `RowOutcome` | Derived | Has string values, `is_terminal` property consistent |

**Parametrized Consistency Tests** verify ALL database-stored enums:
- All members have non-empty string values
- All values are lowercase (database convention)
- No duplicate values within a type
- String conversion roundtrips correctly
- Empty string rejected
- Uppercase variants rejected (case-sensitive)
- Whitespace variants rejected

## Test Strategies Reference

Common Hypothesis strategies used across property tests:

```python
# JSON-safe primitives (RFC 8785 compatible)
_MAX_SAFE_INT = 2**53 - 1
_MIN_SAFE_INT = -(2**53 - 1)

json_primitives = (
    st.none()
    | st.booleans()
    | st.integers(min_value=_MIN_SAFE_INT, max_value=_MAX_SAFE_INT)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=100)
)

# Recursive JSON structures
json_values = st.recursive(
    json_primitives,
    lambda children: (
        st.lists(children, max_size=10)
        | st.dictionaries(st.text(max_size=20), children, max_size=10)
    ),
    max_leaves=50,
)

# Row-like data (what transforms process)
row_data = st.dictionaries(
    keys=st.text(min_size=1, max_size=50),
    values=json_primitives,
    min_size=1,
    max_size=20,
)
```

## Adding New Property Tests

### Template for New Property Test File

```python
# tests/property/<category>/test_<feature>.py
"""Property-based tests for <feature>.

These tests verify <invariant description>.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.<module> import <function_under_test>


class Test<Feature>Properties:
    """Property tests for <feature>."""

    @given(data=<strategy>)
    @settings(max_examples=200)
    def test_<property_name>(self, data) -> None:
        """Property: <describe what must hold for all inputs>."""
        result1 = <function_under_test>(data)
        result2 = <function_under_test>(data)
        assert result1 == result2, f"Property violated for: {data!r}"
```

### Guidelines

1. **Name tests after properties, not implementations**: `test_hash_is_deterministic` not `test_hash_function_works`

2. **Document why the property matters**: Connect to audit integrity, trust model, or system invariants

3. **Use appropriate `max_examples`**:
   - Simple properties: 100-200
   - Complex properties with many edge cases: 300-500
   - Properties that are slow to test: 50-100

4. **Constrain strategies appropriately**: Match real-world constraints (e.g., RFC 8785 integer bounds)

5. **Test both positive and negative cases**: Verify valid inputs succeed AND invalid inputs fail correctly

## Contract Testing

Contract tests verify that plugin implementations honor their protocol guarantees. They test **interface contracts**, not implementation details.

### Running Contract Tests

```bash
# Run all contract tests
.venv/bin/python -m pytest tests/contracts/source_contracts/ tests/contracts/transform_contracts/ tests/contracts/sink_contracts/

# Run specific plugin contract
.venv/bin/python -m pytest tests/contracts/source_contracts/test_csv_source_contract.py -v
```

### Contract Test Base Classes

The contract test framework provides abstract base classes that define the protocol requirements:

| Base Class | Protocol | Key Contracts |
|------------|----------|---------------|
| `SourceContractTestBase` | `SourceProtocol` | `load()` yields `SourceRow`; valid rows have data; quarantined rows have error; `close()` is idempotent |
| `TransformContractTestBase` | `TransformProtocol` | `process()` returns `TransformResult`; success has output data; error has reason; `close()` is idempotent |
| `SinkContractTestBase` | `SinkProtocol` | `write()` returns `ArtifactDescriptor`; `content_hash` is SHA-256; same data = same hash; `flush()`/`close()` are idempotent |

### Creating Contract Tests for New Plugins

```python
# tests/contracts/source_contracts/test_my_source_contract.py
from .test_source_protocol import SourceContractPropertyTestBase

class TestMySourceContract(SourceContractPropertyTestBase):
    """Contract tests for MySource."""

    @pytest.fixture
    def source_data(self, tmp_path: Path) -> Path:
        """Create test data for the source."""
        data_file = tmp_path / "test_data.txt"
        data_file.write_text("test data")
        return data_file

    @pytest.fixture
    def source(self, source_data: Path) -> SourceProtocol:
        """Create a configured source instance."""
        return MySource({
            "path": str(source_data),
            "schema": {"fields": "dynamic"},
            "on_validation_failure": "discard",
        })

    # Inherits all protocol contract tests automatically!
    # Add plugin-specific tests below:

    def test_my_source_specific_behavior(self, source, ctx):
        """Test behavior specific to MySource."""
        ...
```

### Key Contract Properties

**Source Contracts** (Critical for data ingestion):
- `load()` MUST yield `SourceRow` objects, not raw dicts
- Valid rows MUST have `.row` as a dict
- Quarantined rows MUST have `.quarantine_error` and `.quarantine_destination`
- `close()` MUST be idempotent (safe to call multiple times)

**Transform Contracts** (Critical for processing):
- `process()` MUST return `TransformResult`
- Success results MUST have output data (`.row` or `.rows`)
- Error results MUST have `.reason` dict
- DETERMINISTIC transforms MUST produce same output for same input

**Sink Contracts** (Critical for audit integrity):
- `write()` MUST return `ArtifactDescriptor`
- `ArtifactDescriptor.content_hash` MUST be valid SHA-256 (64 hex chars)
- `ArtifactDescriptor.size_bytes` MUST be >= 0
- Same data MUST produce same `content_hash` (determinism!)
- `flush()` and `close()` MUST be idempotent

## Future Test Phases

The property-based testing foundation is Phase 1 of a comprehensive test regime:

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Property-Based Testing Foundation | ✅ Complete |
| 2 | Contract Testing (Source/Transform/Sink protocols) | ✅ Complete |
| 3 | Mutation Testing (mutmut) | Planned |
| 4 | Quality Gates (CI thresholds) | Planned |
| 5 | Integration Test Hardening | Planned |
| 6 | Chaos Engineering Prep | Planned |

See `docs/plans/2026-01-20-world-class-test-regime.md` for the complete test regime proposal.

## Troubleshooting

### Test Fails with "RFC 8785 integer overflow"

The integer value exceeds JavaScript-safe bounds (±2^53-1). Ensure your strategy uses:
```python
st.integers(min_value=-(2**53 - 1), max_value=2**53 - 1)
```

### Hypothesis is Slow

- Use `HYPOTHESIS_PROFILE=debug` for quick iteration
- Reduce `max_examples` in specific tests
- Check if strategies are generating overly complex data

### Flaky Property Test

Hypothesis uses a deterministic random seed, so true flakiness is rare. If a test fails intermittently:
1. Check for global state or time-dependent behavior
2. Look for non-determinism in the code under test
3. Use `@seed(...)` decorator to reproduce specific failures

### Finding the Minimal Failing Example

When a property test fails, Hypothesis automatically shrinks to the minimal example. The failure output shows:
```
Falsifying example: test_property(data={'key': 'a'})
```

This is the simplest input that triggers the failure.
