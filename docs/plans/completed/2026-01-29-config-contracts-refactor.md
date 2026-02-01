# Config Contracts Refactor: Protocol-Based Settings→Runtime Enforcement

> **Priority:** P2 (Architectural debt, creates ongoing bug risk)
> **Effort:** Medium-High (3-4 sessions)
> **Risk:** Low (additive changes, comprehensive test coverage, existing AST infrastructure)
> **Revision:** v2 - Incorporates 4-perspective review panel feedback

## Executive Summary

The original proposal was reviewed by a 4-perspective panel:
- **Architecture Critic**: Approved with suggestions (drop unnecessary protocol, preserve `from_policy()`)
- **Python Engineering**: Request changes (slots, documentation, TypeVar fix)
- **Quality Assurance**: Request changes (test gaps for reverse orphans, field mappings, regression)
- **Systems Thinking**: Request changes (wrong leverage point; use structural typing)

This revision adopts the **protocol-based approach** recommended by Systems Thinking, making orphaned fields a **compile-time error** rather than relying solely on runtime/AST enforcement.

---

## Problem Statement (Unchanged)

The contracts model was designed to own all cross-boundary types, but **Settings→Runtime config mappings were never properly contracted**. This caused P2-2026-01-21 (`exponential_base` silently ignored) and audit revealed 10+ orphaned config fields across the codebase.

### The Drifting Goals Pattern

The original contracts plan (2026-01-16) specified contracts would own cross-boundary types. What shipped was re-exports: "These are re-exports from core/config.py for import consistency."

This is the **Drifting Goals** archetype - the standard drifted from "contracts own types" to "contracts re-export for convenience," creating the architectural gap that allowed orphaned fields to accumulate.

### Feedback Loop Analysis

```
Settings field added
    ↓ (no enforcement)
Runtime doesn't receive it
    ↓ (silent failure - no error)
Tests pass (don't check the mapping)
    ↓ (false confidence)
More fields added carelessly
    ↓
[Reinforcing loop amplifies]
```

The proposed fix must **break this loop** at a high leverage point.

---

## Target Architecture: Protocol-Based Contracts

### The Key Insight

The original proposal put contracts at Level 5 (Rules - AST checker enforcement). The systems thinking review identified this creates friction and false positives.

**Better leverage point: Level 10 (Structure)** - Use Python's structural typing so orphaned fields become **compile-time errors** caught by mypy.

### The Pattern

```python
# contracts/config/runtime.py - Define what engine EXPECTS
class RuntimeRetryProtocol(Protocol):
    """Contract: what RetryManager requires."""
    max_attempts: int
    base_delay: float
    max_delay: float
    exponential_base: float

# contracts/config/runtime.py - Concrete implementation
@dataclass(frozen=True, slots=True)
class RuntimeRetryConfig:
    """Runtime retry config implementing the protocol."""
    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float  # Internal only - documented in INTERNAL_DEFAULTS
    exponential_base: float

    @classmethod
    def from_settings(cls, settings: "RetrySettings") -> "RuntimeRetryConfig":
        """Trust boundary: Tier 3 (Settings) → Tier 2 (Runtime)."""
        return cls(
            max_attempts=settings.max_attempts,
            base_delay=settings.initial_delay_seconds,
            max_delay=settings.max_delay_seconds,
            jitter=INTERNAL_DEFAULTS["jitter"],
            exponential_base=settings.exponential_base,
        )

    @classmethod
    def from_policy(cls, policy: RetryPolicy | None) -> "RuntimeRetryConfig":
        """Plugin policy → Runtime. Preserves existing behavior."""
        if policy is None:
            return cls.default()
        return cls(
            max_attempts=policy.get("max_attempts", POLICY_DEFAULTS["max_attempts"]),
            base_delay=policy.get("base_delay", POLICY_DEFAULTS["base_delay"]),
            max_delay=policy.get("max_delay", POLICY_DEFAULTS["max_delay"]),
            jitter=policy.get("jitter", POLICY_DEFAULTS["jitter"]),
            exponential_base=policy.get("exponential_base", POLICY_DEFAULTS["exponential_base"]),
        )

# engine/retry.py - Accepts protocol, not concrete type
class RetryManager:
    def __init__(self, config: RuntimeRetryProtocol): ...
```

### Why This Works

1. **mypy enforces completeness** - If `RuntimeRetryConfig` is missing a protocol field, mypy fails
2. **No AST checker needed for field alignment** - Python's type system handles it
3. **Decoupled** - Protocols don't import Settings or Engine
4. **Explicit trust boundary** - `from_settings()` is the only path from user config to runtime

---

## Detailed Architecture

### Package Structure

```
src/elspeth/contracts/
├── __init__.py           # Re-exports everything
├── config/
│   ├── __init__.py       # Re-exports
│   ├── protocols.py      # Runtime protocols (what engine expects)
│   ├── runtime.py        # Runtime dataclasses implementing protocols
│   ├── defaults.py       # INTERNAL_DEFAULTS, POLICY_DEFAULTS registries
│   └── alignment.py      # Field mapping documentation (machine-readable)
├── ...existing files...
```

### protocols.py - What Engine Components Expect

```python
"""Runtime configuration protocols.

These protocols define what engine components REQUIRE. Runtime config
dataclasses MUST implement these protocols. mypy enforces this.

Engine components accept protocols, not concrete types, enabling:
1. Structural typing enforcement (mypy catches missing fields)
2. Testability (can mock with any compatible object)
3. Decoupling (engine doesn't depend on Settings)
"""
from typing import Protocol, runtime_checkable

@runtime_checkable
class RuntimeRetryProtocol(Protocol):
    """What RetryManager requires for retry behavior."""
    max_attempts: int
    base_delay: float
    max_delay: float
    exponential_base: float
    # Note: jitter is internal implementation detail, not in protocol

@runtime_checkable
class RuntimeRateLimitProtocol(Protocol):
    """What RateLimitRegistry requires for rate limiting."""
    enabled: bool
    default_requests_per_second: float | None
    default_requests_per_minute: float | None

@runtime_checkable
class RuntimeConcurrencyProtocol(Protocol):
    """What Orchestrator/plugins require for parallelism."""
    max_workers: int

@runtime_checkable
class RuntimeCheckpointProtocol(Protocol):
    """What CheckpointManager requires."""
    enabled: bool
    frequency: int
    aggregation_boundaries: bool
```

### runtime.py - Concrete Implementations

```python
"""Runtime configuration dataclasses.

These implement the protocols defined in protocols.py. Each has:
1. from_settings() - Trust boundary conversion (Tier 3 → Tier 2)
2. from_policy() - Plugin policy conversion (where applicable)
3. default() - Sensible defaults for testing

All use frozen=True (immutable) and slots=True (memory efficient).
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .defaults import INTERNAL_DEFAULTS, POLICY_DEFAULTS
from .protocols import (
    RuntimeRetryProtocol,
    RuntimeRateLimitProtocol,
    RuntimeConcurrencyProtocol,
    RuntimeCheckpointProtocol,
)

if TYPE_CHECKING:
    from elspeth.core.config import (
        RetrySettings,
        RateLimitSettings,
        ConcurrencySettings,
        CheckpointSettings,
    )
    from elspeth.contracts.engine import RetryPolicy


@dataclass(frozen=True, slots=True)
class RuntimeRetryConfig:
    """Runtime retry configuration.

    Implements RuntimeRetryProtocol for use by RetryManager.

    Internal Defaults (not exposed in Settings):
        jitter: 1.0 - Full jitter applied to exponential backoff

    Field Mappings (Settings → Runtime):
        max_attempts ← max_attempts
        base_delay ← initial_delay_seconds (RENAMED)
        max_delay ← max_delay_seconds (RENAMED)
        exponential_base ← exponential_base
        jitter ← INTERNAL_DEFAULTS["jitter"] (HARDCODED)
    """
    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float
    exponential_base: float

    @classmethod
    def from_settings(cls, settings: "RetrySettings") -> "RuntimeRetryConfig":
        """Convert user-facing Settings to runtime config.

        This is the Tier 3 → Tier 2 trust boundary conversion.
        """
        return cls(
            max_attempts=settings.max_attempts,
            base_delay=settings.initial_delay_seconds,
            max_delay=settings.max_delay_seconds,
            jitter=INTERNAL_DEFAULTS["RuntimeRetryConfig"]["jitter"],
            exponential_base=settings.exponential_base,
        )

    @classmethod
    def from_policy(cls, policy: "RetryPolicy | None") -> "RuntimeRetryConfig":
        """Convert plugin retry policy to runtime config.

        Merges with POLICY_DEFAULTS for missing fields.
        """
        if policy is None:
            return cls.default()
        defaults = POLICY_DEFAULTS["RuntimeRetryConfig"]
        return cls(
            max_attempts=policy.get("max_attempts", defaults["max_attempts"]),
            base_delay=policy.get("base_delay", defaults["base_delay"]),
            max_delay=policy.get("max_delay", defaults["max_delay"]),
            jitter=policy.get("jitter", defaults["jitter"]),
            exponential_base=policy.get("exponential_base", defaults["exponential_base"]),
        )

    @classmethod
    def default(cls) -> "RuntimeRetryConfig":
        """Default configuration for testing."""
        defaults = POLICY_DEFAULTS["RuntimeRetryConfig"]
        return cls(**defaults)

    @classmethod
    def no_retry(cls) -> "RuntimeRetryConfig":
        """Configuration that disables retries."""
        return cls(
            max_attempts=1,
            base_delay=0.0,
            max_delay=0.0,
            jitter=0.0,
            exponential_base=1.0,
        )


# Similar pattern for RuntimeRateLimitConfig, RuntimeConcurrencyConfig, RuntimeCheckpointConfig
```

### defaults.py - Documented Hardcoded Values

```python
"""Default value registries for runtime configs.

INTERNAL_DEFAULTS: Values that are hardcoded and NOT exposed in Settings.
    These must be explicitly documented here. The alignment checker
    verifies every hardcoded value in from_settings() appears here.

POLICY_DEFAULTS: Defaults used when merging plugin policies (from_policy).
    These provide sensible defaults when plugins don't specify values.
"""

INTERNAL_DEFAULTS: dict[str, dict[str, float | int | bool | str]] = {
    "RuntimeRetryConfig": {
        "jitter": 1.0,  # Full jitter for exponential backoff
    },
    # Add others as needed
}

POLICY_DEFAULTS: dict[str, dict[str, float | int | bool | str]] = {
    "RuntimeRetryConfig": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 60.0,
        "jitter": 1.0,
        "exponential_base": 2.0,
    },
    "RuntimeRateLimitConfig": {
        "enabled": False,
        "default_requests_per_second": None,
        "default_requests_per_minute": None,
    },
    "RuntimeConcurrencyConfig": {
        "max_workers": 4,
    },
    "RuntimeCheckpointConfig": {
        "enabled": True,
        "frequency": 100,
        "aggregation_boundaries": True,
    },
}
```

### alignment.py - Machine-Readable Field Mappings

```python
"""Field alignment documentation for AST checker.

This module provides machine-readable documentation of Settings → Runtime
field mappings. The AST checker uses this to verify from_settings()
implementations are complete and correct.

Format:
    FIELD_MAPPINGS[RuntimeClass][runtime_field] = {
        "source": "SettingsClass.field_name" | "INTERNAL" | "COMPUTED",
        "transform": None | "description of transformation",
    }
"""

FIELD_MAPPINGS: dict[str, dict[str, dict[str, str | None]]] = {
    "RuntimeRetryConfig": {
        "max_attempts": {
            "source": "RetrySettings.max_attempts",
            "transform": None,
        },
        "base_delay": {
            "source": "RetrySettings.initial_delay_seconds",
            "transform": "renamed: initial_delay_seconds → base_delay",
        },
        "max_delay": {
            "source": "RetrySettings.max_delay_seconds",
            "transform": "renamed: max_delay_seconds → max_delay",
        },
        "jitter": {
            "source": "INTERNAL",
            "transform": "hardcoded from INTERNAL_DEFAULTS",
        },
        "exponential_base": {
            "source": "RetrySettings.exponential_base",
            "transform": None,
        },
    },
    # Add others as implemented
}

# Reverse mapping for orphan detection
SETTINGS_TO_RUNTIME: dict[str, str] = {
    "RetrySettings": "RuntimeRetryConfig",
    "RateLimitSettings": "RuntimeRateLimitConfig",
    "ConcurrencySettings": "RuntimeConcurrencyConfig",
    "CheckpointSettings": "RuntimeCheckpointConfig",
}

# Settings classes that intentionally have no Runtime counterpart
EXEMPT_SETTINGS: set[str] = {
    "SourceSettings",      # Plugin options passed directly
    "TransformSettings",   # Plugin options passed directly
    "SinkSettings",        # Plugin options passed directly
    "AggregationSettings", # Used directly by executors
    "GateSettings",        # Used directly by expression parser
    "CoalesceSettings",    # Used directly by CoalesceExecutor
    "TriggerConfig",       # Nested in AggregationSettings
    "DatabaseSettings",    # Passed to SQLAlchemy directly
    "LandscapeSettings",   # Passed to LandscapeDB directly
    "LandscapeExportSettings",  # Nested, accessed directly
    "PayloadStoreSettings",     # Passed to PayloadStore directly
    "ServiceRateLimit",    # Nested in RateLimitSettings
    "ElspethSettings",     # Top-level container, not a runtime type
}
```

---

## Implementation Tasks

### Phase 1: Create Protocol Infrastructure

**Task 1.1: Create contracts/config/ subpackage**
- Create `contracts/config/__init__.py`
- Create `contracts/config/protocols.py` with runtime protocols
- Create `contracts/config/defaults.py` with INTERNAL_DEFAULTS and POLICY_DEFAULTS
- Create `contracts/config/alignment.py` with FIELD_MAPPINGS and EXEMPT_SETTINGS
- Update `contracts/__init__.py` to re-export from new location
- **Tests:** Verify all existing imports still work

**Task 1.2: Create RuntimeRetryConfig**
- Define in `contracts/config/runtime.py`
- Include ALL fields from current `engine/retry.py::RetryConfig`
- Use `@dataclass(frozen=True, slots=True)`
- Add `from_settings()` with explicit field mapping
- Add `from_policy()` preserving current behavior
- Add `default()` and `no_retry()` factories
- Document jitter hardcode in docstring AND in INTERNAL_DEFAULTS
- **Tests:**
  - `test_runtime_retry_implements_protocol()` - mypy structural check
  - `test_runtime_has_no_orphan_fields()` - reverse orphan detection
  - `test_retry_field_name_mapping()` - explicit mapping assertions

**Task 1.3: Create RuntimeRateLimitConfig**
- Define in `contracts/config/runtime.py`
- Map ALL 5 fields from `RateLimitSettings`
- Add to FIELD_MAPPINGS
- **Tests:** Protocol implementation + alignment tests

**Task 1.4: Create RuntimeConcurrencyConfig**
- Define in `contracts/config/runtime.py`
- Map `max_workers` from `ConcurrencySettings`
- Add to FIELD_MAPPINGS
- **Tests:** Protocol implementation + alignment tests

**Task 1.5: Create RuntimeCheckpointConfig**
- Define in `contracts/config/runtime.py`
- Map ALL fields including `aggregation_boundaries`
- Add to FIELD_MAPPINGS
- **Tests:** Protocol implementation + alignment tests

### Phase 2: Wire Runtime Configs to Engine

**Task 2.1: Migrate RetryManager to protocol**
- Update `engine/retry.py` to import `RuntimeRetryConfig` from contracts
- Change `RetryManager.__init__` to accept `RuntimeRetryProtocol`
- Delete local `RetryConfig` dataclass (no alias needed - clean break)
- Update `Orchestrator` to use `RuntimeRetryConfig.from_settings()`
- **Tests:**
  - All existing retry tests must pass
  - `test_exponential_base_bug_regression()` - P2-2026-01-21 prevention

**Task 2.2: Wire RuntimeRateLimitConfig to Orchestrator**
- Create `RuntimeRateLimitConfig.from_settings()` in CLI
- Pass to `RateLimitRegistry` constructor
- Inject registry into Orchestrator
- Pass to plugins that need rate limiting
- Remove xfail marker from alignment test
- **Tests:** Integration test with rate limiting active

**Task 2.3: Wire RuntimeConcurrencyConfig to ThreadPoolExecutor**
- Create `RuntimeConcurrencyConfig.from_settings()` in CLI
- Pass `max_workers` to ThreadPoolExecutor in Orchestrator
- Update plugin executor to respect global concurrency
- Remove xfail marker from alignment test
- **Tests:** Concurrency integration test

**Task 2.4: Wire RuntimeCheckpointConfig to normal runs**
- Pass checkpoint config to Orchestrator in ALL runs (not just resume)
- Implement `aggregation_boundaries` enforcement
- Remove xfail markers from alignment tests
- **Tests:** Checkpoint integration tests for normal runs

### Phase 3: Extend AST Enforcement

**Task 3.1: Add Settings→Runtime alignment check to check_contracts.py**
- New rule: Every Settings class must have Runtime counterpart OR be in EXEMPT_SETTINGS
- Uses SETTINGS_TO_RUNTIME mapping from alignment.py
- **Tests:** AST checker unit tests

**Task 3.2: Add from_settings() field coverage check**
- Parse from_settings() body to extract accessed Settings fields
- Compare against Settings class fields
- Flag any Settings field NOT accessed (potential orphan)
- Allow exemptions via FIELD_MAPPINGS with source="INTERNAL"
- **Tests:** AST checker tests with intentional orphan

**Task 3.3: Add field name mapping validation**
- Verify `runtime_field=settings.settings_field` matches FIELD_MAPPINGS
- Catch misrouted fields (e.g., `base_delay=settings.max_delay_seconds`)
- **Tests:** AST checker tests with intentional misroute

**Task 3.4: Add hardcode documentation enforcement**
- Scan from_settings() for literal assignments (not settings.X)
- Require literal to exist in INTERNAL_DEFAULTS for that class
- **Tests:** AST checker tests with undocumented hardcode

### Phase 4: Test Suite Hardening

**Task 4.1: Add reverse orphan detection tests**
```python
def test_runtime_has_no_orphan_fields():
    """Every Runtime field must have a documented Settings origin."""
    for runtime_cls in [RuntimeRetryConfig, RuntimeRateLimitConfig, ...]:
        runtime_fields = set(runtime_cls.__annotations__.keys())
        mapping = FIELD_MAPPINGS[runtime_cls.__name__]
        documented_fields = set(mapping.keys())
        orphans = runtime_fields - documented_fields
        assert not orphans, f"Undocumented Runtime fields: {orphans}"
```

**Task 4.2: Add explicit field mapping tests**
```python
def test_retry_field_mapping_explicit():
    """Verify Settings→Runtime field name mappings."""
    settings = RetrySettings(
        max_attempts=5,
        initial_delay_seconds=2.0,
        max_delay_seconds=120.0,
        exponential_base=3.0,
    )
    runtime = RuntimeRetryConfig.from_settings(settings)

    # Explicit assertions - sentinel values would miss renamed fields
    assert runtime.max_attempts == 5
    assert runtime.base_delay == 2.0  # initial_delay_seconds → base_delay
    assert runtime.max_delay == 120.0  # max_delay_seconds → max_delay
    assert runtime.exponential_base == 3.0
    assert runtime.jitter == INTERNAL_DEFAULTS["RuntimeRetryConfig"]["jitter"]
```

**Task 4.3: Add P2-2026-01-21 regression test**
```python
def test_exponential_base_bug_regression():
    """Regression: exponential_base was silently ignored (P2-2026-01-21)."""
    settings = RetrySettings(
        max_attempts=3,
        exponential_base=3.0,
        initial_delay_seconds=1.0,
    )
    runtime = RuntimeRetryConfig.from_settings(settings)

    # Compute expected delays with exponential_base=3.0
    expected_delays = [1.0, 3.0, 9.0]  # 1 * 3^0, 1 * 3^1, 1 * 3^2

    # Verify runtime config would produce correct backoff
    for attempt, expected in enumerate(expected_delays):
        actual = runtime.base_delay * (runtime.exponential_base ** attempt)
        assert actual == pytest.approx(expected), \
            f"Attempt {attempt}: expected {expected}, got {actual}"
```

**Task 4.4: Add property-based roundtrip tests (optional but recommended)**
```python
from hypothesis import given, strategies as st

@given(
    max_attempts=st.integers(min_value=1, max_value=10),
    exponential_base=st.floats(min_value=1.1, max_value=10.0),
    initial_delay=st.floats(min_value=0.1, max_value=5.0),
)
def test_retry_config_roundtrip_hypothesis(max_attempts, exponential_base, initial_delay):
    """All valid RetrySettings values survive from_settings() conversion."""
    settings = RetrySettings(
        max_attempts=max_attempts,
        exponential_base=exponential_base,
        initial_delay_seconds=initial_delay,
    )
    runtime = RuntimeRetryConfig.from_settings(settings)

    assert runtime.max_attempts == max_attempts
    assert runtime.exponential_base == exponential_base
    assert runtime.base_delay == initial_delay
```

### Phase 5: Cleanup and Documentation

**Task 5.1: Remove band-aid alignment tests**
- `tests/core/test_config_alignment.py` becomes redundant
- Replace with new protocol-based tests
- Keep field categorization docs if useful

**Task 5.2: Update CLAUDE.md**
- Document Settings→Runtime pattern
- Add to "Core Architecture" section
- Reference trust boundary model
- Add to "Critical Implementation Patterns"

**Task 5.3: Add mypy strict checks for contracts/**
- Ensure `--strict` in mypy config for contracts/
- Protocol structural typing requires strict mode

---

## Success Criteria

### Immediate (Phase 1-2 Complete)

- [ ] All runtime config dataclasses live in `contracts/config/runtime.py`
- [ ] All implement corresponding protocols from `contracts/config/protocols.py`
- [ ] mypy enforces protocol compliance (compile-time orphan detection)
- [ ] All `from_settings()` methods have explicit field mappings documented
- [ ] All hardcoded defaults are in INTERNAL_DEFAULTS
- [ ] Zero orphaned fields (all alignment tests pass)
- [ ] `RateLimitSettings` actually controls rate limiting
- [ ] `ConcurrencySettings.max_workers` actually controls thread pool
- [ ] `CheckpointSettings` works for normal runs

### Long-term (Phase 3-5 Complete)

- [ ] AST checker prevents future field orphaning at CI time
- [ ] Adding a Settings field without Runtime mapping fails CI
- [ ] Field misrouting detected by AST checker
- [ ] Undocumented hardcodes fail CI
- [ ] CLAUDE.md documents the pattern
- [ ] Property-based tests catch edge cases

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| mypy strict mode breaks existing code | Medium | Medium | Run mypy on contracts/ only first, fix incrementally |
| Protocol pattern unfamiliar to team | Low | Low | Docstrings explain pattern, CLAUDE.md documents |
| AST checker false positives | Low | Medium | EXEMPT_SETTINGS whitelist, FIELD_MAPPINGS for special cases |
| RateLimitRegistry integration complex | Medium | Low | Already implemented, just not wired |
| Performance from dataclass overhead | Very Low | Very Low | slots=True, created once per run |

---

## Comparison: Original vs Revised

| Aspect | Original Proposal | Revised (Protocol-Based) |
|--------|-------------------|--------------------------|
| Orphan detection | AST checker at CI | mypy at compile-time + AST checker |
| False positive risk | Medium (AST heuristics) | Low (structural typing is exact) |
| Developer friction | High (custom AST rules) | Low (standard mypy) |
| Coupling | contracts/ imports config + engine | contracts/ standalone (protocols) |
| Implementation effort | Medium | Medium-High (worth it pre-release) |
| Long-term maintenance | Custom AST checker to maintain | Standard Python typing |

---

## Dependencies

- **None** - Foundational infrastructure cleanup
- **Requires:** mypy strict mode in CI (already enabled)
- **Blocked by:** Nothing
- **Blocks:** No features, but reduces ongoing bug risk

---

## Verification Commands

After each phase:

```bash
# Run alignment tests
.venv/bin/python -m pytest tests/core/test_config_alignment.py -v

# Run all retry tests including regression
.venv/bin/python -m pytest tests/engine/test_retry*.py tests/integration/test_retry*.py -v

# Type check contracts (strict mode)
.venv/bin/python -m mypy src/elspeth/contracts/ --strict

# Run AST checker
python scripts/check_contracts.py

# Lint
.venv/bin/python -m ruff check src/elspeth/contracts/
```

---

## References

- **Original contracts plan:** `docs/plans/completed/2026-01-16-contracts-subsystem.md`
- **Bug that exposed this:** `docs/bugs/closed/engine-retry/P2-2026-01-21-retry-exponential-base-ignored.md`
- **Quality audit findings:** `docs/quality-audit/INTEGRATION_SEAM_ANALYSIS_REPORT.md` (Finding #8)
- **Existing AST checkers:** `scripts/check_contracts.py`, `scripts/cicd/no_bug_hiding.py`
- **Alignment tests (current):** `tests/core/test_config_alignment.py`
- **Data Manifesto:** `CLAUDE.md` (Three-Tier Trust Model section)
- **Systems Thinking Review:** Identified Drifting Goals archetype and leverage point analysis
