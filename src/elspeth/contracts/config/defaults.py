# src/elspeth/contracts/config/defaults.py
"""Default value registries for runtime configuration.

Two categories of defaults:

1. INTERNAL_DEFAULTS: Values hardcoded in runtime code, NOT exposed in Settings.
   These are implementation details that users shouldn't need to configure.
   Documented here for:
   - Transparency (auditors can see what values are used)
   - AST checker validation (ensure these stay out of Settings)
   - Maintenance (single source of truth for internal defaults)

2. POLICY_DEFAULTS: Defaults for plugin RetryPolicy dicts.
   Plugins return partial RetryPolicy, we fill in missing fields.
   These MUST match RetryConfig dataclass defaults.

Why document INTERNAL_DEFAULTS?
    The P2-2026-01-21 bug showed that undocumented hardcoded values are
    invisible. When jitter=1.0 is buried in RetryConfig, no one knows
    whether it should be configurable. By documenting it here:
    - We explicitly declare "this is intentionally internal"
    - AST checkers can verify it stays internal
    - Future maintainers know the design intent
"""

from typing import Final

# =============================================================================
# INTERNAL DEFAULTS - Values hardcoded in runtime, NOT in Settings
# =============================================================================

INTERNAL_DEFAULTS: Final[dict[str, dict[str, int | float | bool | str]]] = {
    # RetryConfig internal defaults (not exposed in RetrySettings)
    "retry": {
        # jitter adds randomness to backoff to prevent thundering herd
        # Fixed at 1.0 second - not user-configurable by design
        "jitter": 1.0,
    },
    # Telemetry internal defaults
    "telemetry": {
        # Queue size for async export buffer
        # 1000 events absorbs bursts without excessive memory
        # Not user-configurable - internal implementation detail
        "queue_size": 1000,
    },
}


# =============================================================================
# POLICY_DEFAULTS - Defaults for plugin RetryPolicy dicts
# =============================================================================
#
# When plugins return partial RetryPolicy (TypedDict with total=False),
# missing fields get these defaults. MUST stay in sync with RetryConfig
# dataclass defaults.
#
# NOTE: This is duplicated from engine/retry.py to avoid circular imports.
# The authoritative source is here; engine/retry.py should import from here.

POLICY_DEFAULTS: Final[dict[str, int | float]] = {
    "max_attempts": 3,
    "base_delay": 1.0,
    "max_delay": 60.0,
    "jitter": 1.0,  # Internal default, included for policy completeness
    "exponential_base": 2.0,
}


def get_internal_default(subsystem: str, field: str) -> int | float | bool | str:
    """Get an internal default value.

    Args:
        subsystem: Subsystem name (e.g., "retry", "checkpoint")
        field: Field name within subsystem

    Returns:
        The default value

    Raises:
        KeyError: If subsystem or field not found (bug - internal defaults
                  should always be present for documented fields)
    """
    return INTERNAL_DEFAULTS[subsystem][field]


def get_policy_default(field: str) -> int | float:
    """Get a policy default value.

    Args:
        field: Field name (e.g., "max_attempts", "jitter")

    Returns:
        The default value

    Raises:
        KeyError: If field not found (bug - policy should have all fields)
    """
    return POLICY_DEFAULTS[field]
