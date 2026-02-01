# Test Defect Report

## Summary

- `test_can_import_settings_from_contracts` only asserts non-None, so it doesn’t validate that `elspeth.contracts` actually re-exports the correct core config classes.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/test_config.py:7` shows the test only checks non-None, which would pass even if the re-exported objects are the wrong classes.
```python
def test_can_import_settings_from_contracts(self) -> None:
    from elspeth.contracts import (
        DatasourceSettings,
        ElspethSettings,
    )

    # Just verify import works
    assert ElspethSettings is not None
    assert DatasourceSettings is not None
```
- `src/elspeth/contracts/config.py:6` documents that these are re-exports from `core/config.py`, but the test never asserts identity or equality with those core classes.
```python
# These are re-exports from core/config.py for import consistency.
from elspeth.core.config import (
    CheckpointSettings,
    ...
    SinkSettings,
)
```

## Impact

- Contract regressions (e.g., re-export pointing to the wrong class or a stub) can slip through while tests still pass, weakening guarantees about the public API surface.

## Root Cause Hypothesis

- The test was written as a smoke check for importability and never tightened to assert correct object identity as the contract list grew.

## Recommended Fix

- Add identity checks that `elspeth.contracts.<name>` is the same object as `elspeth.core.config.<name>` for each config class; this makes the test fail if re-exports drift.
- Example pattern:
```python
from elspeth.core import config as core_config
from elspeth import contracts

assert contracts.DatasourceSettings is core_config.DatasourceSettings
```
- Priority justification: These are public contract types; weak checks can hide compatibility-breaking regressions.
---
# Test Defect Report

## Summary

- The contract tests only cover 2 of 12 config settings re-exported from `elspeth.contracts.config`, leaving most config contracts unverified.

## Severity

- Severity: minor
- Priority: P2

## Category

- Incomplete Contract Coverage

## Evidence

- `src/elspeth/contracts/config.py:10` shows 12 config settings re-exported.
```python
from elspeth.core.config import (
    CheckpointSettings,
    ConcurrencySettings,
    DatabaseSettings,
    DatasourceSettings,
    ElspethSettings,
    LandscapeExportSettings,
    LandscapeSettings,
    PayloadStoreSettings,
    RateLimitSettings,
    RetrySettings,
    RowPluginSettings,
    SinkSettings,
)
```
- `tests/contracts/test_config.py:7` only imports and exercises `DatasourceSettings` and `ElspethSettings`; no tests touch the other 10 settings.
```python
from elspeth.contracts import (
    DatasourceSettings,
    ElspethSettings,
)
```

## Impact

- Missing or incorrect re-exports (e.g., `RateLimitSettings` dropped or miswired) would not be detected, reducing confidence in contract stability.

## Root Cause Hypothesis

- Tests were added early for the minimal config surface and not expanded as more config contracts were added.

## Recommended Fix

- Parameterize the tests over all config settings (e.g., `elspeth.contracts.config.__all__`) to assert importability, BaseModel subclassing, and frozen behavior for each.
- Example pattern:
```python
import pytest
from elspeth.contracts import config as contract_config
from elspeth.core import config as core_config

@pytest.mark.parametrize("name", contract_config.__all__)
def test_contract_reexports(name: str) -> None:
    assert getattr(contract_config, name) is getattr(core_config, name)
```
- Priority justification: The contract layer is the public API; gaps here risk silent breakages across subsystems.
