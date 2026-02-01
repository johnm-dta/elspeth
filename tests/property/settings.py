# tests/property/settings.py
"""Standardized Hypothesis settings profiles for property tests.

Provides consistent test intensity across all property test modules.
Import these instead of using inline @settings(max_examples=...).

Usage:
    from tests.property.settings import STANDARD_SETTINGS

    @given(data=row_data)
    @STANDARD_SETTINGS
    def test_something(data):
        ...

Tiers:
- DETERMINISM_SETTINGS: 500 examples - P0 hash/canonical tests (audit integrity)
- STATE_MACHINE_SETTINGS: 200 examples - Stateful tests (sufficient state exploration)
- STANDARD_SETTINGS: 100 examples - Regular property tests
- SLOW_SETTINGS: 50 examples - I/O bound tests (database, file system)
- QUICK_SETTINGS: 20 examples - Fast validation tests (enums, simple rejection)
"""

from hypothesis import settings

# P0: Audit integrity - canonical hashing MUST be deterministic
# These tests protect the foundational guarantee of the audit trail
DETERMINISM_SETTINGS = settings(max_examples=500)

# Stateful tests need enough steps to explore state space
# RuleBasedStateMachine tests benefit from more examples
STATE_MACHINE_SETTINGS = settings(max_examples=200)

# Standard property tests - good balance of coverage and speed
STANDARD_SETTINGS = settings(max_examples=100)

# I/O-bound tests - real database, file system, network
# Fewer examples due to inherent slowness
SLOW_SETTINGS = settings(max_examples=50)

# Quick validation tests - simple input rejection, enum validation
# Fast tests where more examples add little value
QUICK_SETTINGS = settings(max_examples=20)
