---
name: config-contracts-guide
description: >
  Settings-to-Runtime configuration contract system — protocol-based verification,
  from_settings() mapping, alignment tests, adding new Settings fields, and the tier
  model enforcement allowlist. Use when modifying config.py, contracts/config/, runtime
  config dataclasses, or the tier model allowlist.
---

# Configuration Contracts Guide

## Settings-to-Runtime Configuration Pattern

Configuration uses a two-layer pattern to prevent field orphaning:

```text
USER YAML -> Settings (Pydantic) -> Runtime*Config (dataclass) -> Engine Components
             validation            conversion                    runtime behavior
```

**Why two layers?**

1. **Settings classes** (e.g., `RetrySettings`): Pydantic models for YAML validation
2. **Runtime*Config classes** (e.g., `RuntimeRetryConfig`): Frozen dataclasses for engine use

The P2-2026-01-21 bug showed the problem: `exponential_base` was added to `RetrySettings` but never mapped to the engine. Users configured it, Pydantic validated it, but it was silently ignored at runtime.

## Protocol-Based Verification

```python
# contracts/config/protocols.py
@runtime_checkable
class RuntimeRetryProtocol(Protocol):
    """What RetryManager EXPECTS from retry config."""
    @property
    def max_attempts(self) -> int: ...
    @property
    def exponential_base(self) -> float: ...  # mypy catches if missing!

# contracts/config/runtime.py
@dataclass(frozen=True, slots=True)
class RuntimeRetryConfig:
    """Implements RuntimeRetryProtocol."""
    max_attempts: int
    exponential_base: float
    # ... other fields

    @classmethod
    def from_settings(cls, settings: "RetrySettings") -> "RuntimeRetryConfig":
        return cls(
            max_attempts=settings.max_attempts,
            exponential_base=settings.exponential_base,  # Explicit mapping!
        )

# engine/retry.py
class RetryManager:
    def __init__(self, config: RuntimeRetryProtocol):  # Accepts protocol
        self._config = config
```

## Enforcement Layers

1. **mypy (structural typing)**: Verifies `RuntimeRetryConfig` satisfies `RuntimeRetryProtocol`
2. **AST checker**: Verifies `from_settings()` uses all Settings fields (run: `.venv/bin/python -m scripts.check_contracts`)
3. **Alignment tests**: Verifies field mappings are correct and complete

## Key Files

| File | Purpose |
| ---- | ------- |
| `contracts/config/protocols.py` | Protocol definitions (what engine expects) |
| `contracts/config/runtime.py` | Runtime*Config dataclasses with `from_settings()` |
| `contracts/config/alignment.py` | Field mapping documentation (`FIELD_MAPPINGS`) |
| `contracts/config/defaults.py` | Default values (`POLICY_DEFAULTS`, `INTERNAL_DEFAULTS`) |
| `tests/unit/core/test_config_alignment.py` | Comprehensive alignment verification |

## Adding a New Settings Field (Checklist)

1. Add to Settings class in `core/config.py` (Pydantic model)
2. Add to Runtime*Config in `contracts/config/runtime.py` (dataclass field)
3. Map in `from_settings()` method (explicit assignment)
4. If renamed: document in `FIELD_MAPPINGS` in `alignment.py`
5. If internal-only: document in `INTERNAL_DEFAULTS` in `defaults.py`
6. Run `.venv/bin/python -m scripts.check_contracts` and `pytest tests/unit/core/test_config_alignment.py`

## Tier Model Enforcement Allowlist

The allowlist for the `trust_tier.tier_model` elspeth-lints rule lives in `config/cicd/enforce_tier_model/` as a directory of per-module YAML files:

```text
config/cicd/enforce_tier_model/
+-- _defaults.yaml   # version + defaults (fail_on_stale, fail_on_expired)
+-- cli.yaml         # per-file rules for cli.py, cli_helpers.py
+-- contracts.yaml   # contracts/* entries
+-- core.yaml        # core/* entries
+-- engine.yaml      # engine/* entries
+-- mcp.yaml         # mcp/* entries
+-- plugins.yaml     # plugins/* entries
+-- telemetry.yaml   # telemetry/* entries
+-- testing.yaml     # testing/* entries
+-- tui.yaml         # tui/* entries
```

**Adding a new allowlist entry:** Determine the top-level module from the finding's file path (e.g., `core/canonical.py` -> `core.yaml`) and add the entry to that module's YAML file under `allow_hits:`.
New signed entries and signed-entry repairs go through the cicd-judge workflow;
do not hand-write judge metadata. For stale signature repair, `NO_MATCHING_FINDING`,
`SCOPE_BINDING_DRIFT`, `AST_PATH_BINDING_DRIFT`, and operator-only signing steps,
load `cicd-allowlist-audit`.

Run the rule with:

```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check \
  --rules trust_tier.tier_model \
  --root src/elspeth
```
