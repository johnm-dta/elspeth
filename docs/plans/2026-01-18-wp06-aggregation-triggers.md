# WP-06: Aggregation Triggers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace plugin-driven aggregation trigger decisions with config-driven engine evaluation

**Architecture:** Move trigger evaluation from `BaseAggregation.should_trigger()` to the engine. The engine reads trigger configuration (count, timeout, condition, end_of_source) and decides when to flush batches. Plugins only accept/reject rows - they don't decide when to trigger. This separation allows the same aggregation plugin to be used with different trigger rules without code changes.

**Tech Stack:** Python 3.12, Pydantic, SQLAlchemy, existing TriggerType enum from WP-05

**Dependencies:** WP-05 (TriggerType enum must exist in contracts/enums.py)

---

## Pre-Flight Check

Before starting, verify WP-05 is complete:

```bash
python -c "from elspeth.contracts.enums import TriggerType; print([t.value for t in TriggerType])"
```

Expected output: `['count', 'timeout', 'condition', 'end_of_source', 'manual']`

If this fails, WP-05 must be completed first.

---

## Breaking Change Impact Assessment

**This WP makes breaking API changes.** The following test files contain code that will break and must be updated in Tasks 5-6:

| Test File | `AcceptResult.trigger` refs | `should_trigger()` impls |
|-----------|----------------------------|--------------------------|
| `tests/engine/test_executors.py` | 8 | 4 |
| `tests/engine/test_processor.py` | 2 | 2 |
| `tests/engine/test_plugin_detection.py` | 2 | 1 |
| `tests/plugins/test_integration.py` | 2 | 1 |
| `tests/plugins/test_results.py` | 5 | 0 |
| `tests/plugins/test_base.py` | 1 | 1 |
| `tests/plugins/test_node_id_protocol.py` | 1 | 1 |
| `tests/plugins/test_protocols.py` | 3 | 1 |
| `tests/contracts/test_results.py` | 5 | 0 |
| `tests/contracts/test_audit.py` | 1 | 0 |

**Total:** ~30 `trigger` references, 11 mock `should_trigger()` implementations.

Tasks 5 and 6 include grep commands to find all these. Update them ALL - do not leave broken tests.

---

## ExpressionParser Interface Note

The existing `ExpressionParser.evaluate(row)` accepts any dict, not just row data. For trigger conditions, pass a context dict:

```python
context = {"batch_count": 50, "batch_age_seconds": 30.5}
result = parser.evaluate(context)  # Works - parameter name is just "row"
```

Condition expressions use these variables: `batch_count`, `batch_age_seconds`.

---

## Scope Discipline

**DO NOT:**
- Add features not in this plan
- "Improve" existing code while you're in there
- Refactor unrelated code
- Add extra tests beyond what's specified
- Add logging, metrics, or observability not mentioned
- Create abstractions "for future use"

**DO:**
- Follow TDD exactly as written
- Run only the commands specified
- Commit only what's specified
- Move to the next task when done

If you see something that "should" be fixed but isn't in the plan, **note it and move on**. Do not fix it.

---

## Task 1: Add TriggerConfig model to config.py

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**IMPORTANT:** Per plugin-protocol.md lines 893-908, triggers are **combinable** with OR logic
(first one to fire wins). The YAML structure allows specifying multiple triggers:

```yaml
trigger:
  count: 1000           # Fire after 1000 rows
  timeout: 1h           # Or after 1 hour
  condition: "row['type'] == 'flush_signal'"  # Or on special row
```

Note: `end_of_source` is **implicit** - always checked at source exhaustion.

**Step 1: Write the failing test**

Create `tests/core/test_config_aggregation.py`:

```python
"""Tests for AggregationSettings configuration."""

import pytest
from pydantic import ValidationError


class TestTriggerConfig:
    """Tests for TriggerConfig model.

    Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
    """

    def test_count_trigger_only(self) -> None:
        """Count-only trigger configuration."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(count=100)
        assert trigger.count == 100
        assert trigger.timeout_seconds is None
        assert trigger.condition is None

    def test_timeout_trigger_only(self) -> None:
        """Timeout-only trigger configuration."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(timeout_seconds=30.0)
        assert trigger.timeout_seconds == 30.0
        assert trigger.count is None
        assert trigger.condition is None

    def test_condition_trigger_only(self) -> None:
        """Condition-only trigger configuration."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(condition="batch_count >= 50")
        assert trigger.condition == "batch_count >= 50"
        assert trigger.count is None
        assert trigger.timeout_seconds is None

    def test_combined_count_and_timeout(self) -> None:
        """Combined count + timeout triggers (first to fire wins)."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(count=100, timeout_seconds=60.0)
        assert trigger.count == 100
        assert trigger.timeout_seconds == 60.0
        assert trigger.condition is None

    def test_combined_all_triggers(self) -> None:
        """All three trigger types combined."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(
            count=1000,
            timeout_seconds=3600.0,  # 1 hour
            condition="row['type'] == 'flush_signal'",
        )
        assert trigger.count == 1000
        assert trigger.timeout_seconds == 3600.0
        assert trigger.condition == "row['type'] == 'flush_signal'"

    def test_at_least_one_trigger_required(self) -> None:
        """At least one trigger must be specified."""
        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError, match="at least one trigger"):
            TriggerConfig()

    def test_condition_validates_expression(self) -> None:
        """Condition trigger validates expression syntax."""
        from elspeth.core.config import TriggerConfig

        # Invalid Python syntax
        with pytest.raises(ValidationError, match="Invalid.*syntax"):
            TriggerConfig(condition="batch_count >=")

    def test_count_must_be_positive(self) -> None:
        """Count threshold must be positive."""
        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError):
            TriggerConfig(count=0)

        with pytest.raises(ValidationError):
            TriggerConfig(count=-1)

    def test_timeout_must_be_positive(self) -> None:
        """Timeout must be positive."""
        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError):
            TriggerConfig(timeout_seconds=0)

        with pytest.raises(ValidationError):
            TriggerConfig(timeout_seconds=-1.0)

    def test_has_count_property(self) -> None:
        """has_count property indicates count trigger configured."""
        from elspeth.core.config import TriggerConfig

        with_count = TriggerConfig(count=100)
        without_count = TriggerConfig(timeout_seconds=30.0)

        assert with_count.has_count is True
        assert without_count.has_count is False

    def test_has_timeout_property(self) -> None:
        """has_timeout property indicates timeout trigger configured."""
        from elspeth.core.config import TriggerConfig

        with_timeout = TriggerConfig(timeout_seconds=30.0)
        without_timeout = TriggerConfig(count=100)

        assert with_timeout.has_timeout is True
        assert without_timeout.has_timeout is False

    def test_has_condition_property(self) -> None:
        """has_condition property indicates condition trigger configured."""
        from elspeth.core.config import TriggerConfig

        with_condition = TriggerConfig(condition="batch_count > 0")
        without_condition = TriggerConfig(count=100)

        assert with_condition.has_condition is True
        assert without_condition.has_condition is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_config_aggregation.py::TestTriggerConfig::test_count_trigger_only -v`

Expected: FAIL with `ImportError: cannot import name 'TriggerConfig'`

**Step 3: Implement TriggerConfig**

In `src/elspeth/core/config.py`, add after the `_IDENTIFIER_PATTERN` line (around line 16):

```python
class TriggerConfig(BaseModel):
    """Trigger configuration for aggregation batches.

    Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
    The engine evaluates all configured triggers after each accept and fires when
    ANY condition is met.

    Trigger types:
    - count: Fire after N rows accumulated
    - timeout: Fire after N seconds since first accept
    - condition: Fire when expression evaluates to true

    Note: end_of_source is IMPLICIT - always checked at source exhaustion.
    It is not configured here because it always applies.

    Example YAML (combined triggers):
        trigger:
          count: 1000           # Fire after 1000 rows
          timeout: 3600         # Or after 1 hour
          condition: "row['type'] == 'flush_signal'"  # Or on special row
    """

    model_config = {"frozen": True}

    count: int | None = Field(
        default=None,
        gt=0,
        description="Fire after N rows accumulated",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Fire after N seconds since first accept",
    )
    condition: str | None = Field(
        default=None,
        description="Fire when expression evaluates to true",
    )

    @field_validator("condition")
    @classmethod
    def validate_condition_expression(cls, v: str | None) -> str | None:
        """Validate condition is a valid expression at config time."""
        if v is None:
            return v

        from elspeth.engine.expression_parser import (
            ExpressionParser,
            ExpressionSecurityError,
            ExpressionSyntaxError,
        )

        try:
            ExpressionParser(v)
        except ExpressionSyntaxError as e:
            raise ValueError(f"Invalid condition syntax: {e}") from e
        except ExpressionSecurityError as e:
            raise ValueError(f"Forbidden construct in condition: {e}") from e
        return v

    @model_validator(mode="after")
    def validate_at_least_one_trigger(self) -> "TriggerConfig":
        """At least one trigger must be configured."""
        if self.count is None and self.timeout_seconds is None and self.condition is None:
            raise ValueError(
                "At least one trigger must be configured (count, timeout_seconds, or condition)"
            )
        return self

    @property
    def has_count(self) -> bool:
        """Whether count trigger is configured."""
        return self.count is not None

    @property
    def has_timeout(self) -> bool:
        """Whether timeout trigger is configured."""
        return self.timeout_seconds is not None

    @property
    def has_condition(self) -> bool:
        """Whether condition trigger is configured."""
        return self.condition is not None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_config_aggregation.py::TestTriggerConfig -v`

Expected: All 12 tests pass

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(config): add TriggerConfig model for aggregation triggers

Per plugin-protocol.md, triggers are combinable (first to fire wins):
- count: Fire after N rows
- timeout: Fire after N seconds
- condition: Fire when expression true

end_of_source is implicit (always checked at source exhaustion).

Expressions validated using ExpressionParser at config load time.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Add AggregationSettings model to config.py

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config_aggregation.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config_aggregation.py`:

```python
class TestAggregationSettings:
    """Tests for AggregationSettings model."""

    def test_aggregation_settings_valid(self) -> None:
        """Valid aggregation settings with all fields."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=100),
            output_mode="single",
        )
        assert settings.name == "batch_stats"
        assert settings.plugin == "stats_aggregation"
        assert settings.trigger.count == 100
        assert settings.output_mode == "single"

    def test_aggregation_settings_combined_triggers(self) -> None:
        """Aggregation with combined triggers (per plugin-protocol.md)."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=1000, timeout_seconds=3600.0),
            output_mode="single",
        )
        assert settings.trigger.count == 1000
        assert settings.trigger.timeout_seconds == 3600.0

    def test_aggregation_settings_default_output_mode(self) -> None:
        """Output mode defaults to 'single'."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=100),
        )
        assert settings.output_mode == "single"

    def test_aggregation_settings_passthrough_mode(self) -> None:
        """Passthrough output mode is valid."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(timeout_seconds=60.0),
            output_mode="passthrough",
        )
        assert settings.output_mode == "passthrough"

    def test_aggregation_settings_transform_mode(self) -> None:
        """Transform output mode is valid."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=50),
            output_mode="transform",
        )
        assert settings.output_mode == "transform"

    def test_aggregation_settings_invalid_output_mode(self) -> None:
        """Invalid output mode is rejected."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        with pytest.raises(ValidationError, match="output_mode"):
            AggregationSettings(
                name="batch_stats",
                plugin="stats_aggregation",
                trigger=TriggerConfig(count=100),
                output_mode="invalid",  # type: ignore[arg-type]
            )

    def test_aggregation_settings_requires_name(self) -> None:
        """Name is required."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        with pytest.raises(ValidationError, match="name"):
            AggregationSettings(
                plugin="stats_aggregation",
                trigger=TriggerConfig(count=100),
            )

    def test_aggregation_settings_options_default_empty(self) -> None:
        """Options defaults to empty dict."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=100),
        )
        assert settings.options == {}

    def test_aggregation_settings_with_options(self) -> None:
        """Options can be provided."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=100),
            options={"fields": ["value"], "compute_mean": True},
        )
        assert settings.options == {"fields": ["value"], "compute_mean": True}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_config_aggregation.py::TestAggregationSettings::test_aggregation_settings_valid -v`

Expected: FAIL with `ImportError: cannot import name 'AggregationSettings'`

**Step 3: Implement AggregationSettings**

In `src/elspeth/core/config.py`, add after `TriggerConfig`:

```python
class AggregationSettings(BaseModel):
    """Aggregation configuration for batching rows.

    Aggregations collect rows until a trigger fires, then process the batch.
    The engine evaluates trigger conditions - plugins only accept/reject rows.

    Output modes:
    - single: Batch produces one aggregated result row
    - passthrough: Batch releases all accepted rows unchanged
    - transform: Batch applies a transform function to produce results

    Example YAML:
        aggregations:
          - name: batch_stats
            plugin: stats_aggregation
            trigger:
              type: count
              threshold: 100
            output_mode: single
            options:
              fields: ["value"]
              compute_mean: true
    """

    model_config = {"frozen": True}

    name: str = Field(description="Aggregation identifier (unique within pipeline)")
    plugin: str = Field(description="Plugin name to instantiate")
    trigger: TriggerConfig = Field(description="When to flush the batch")
    output_mode: Literal["single", "passthrough", "transform"] = Field(
        default="single",
        description="How batch produces output rows",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_config_aggregation.py::TestAggregationSettings -v`

Expected: All 8 tests pass

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(config): add AggregationSettings model

Defines aggregation configuration with:
- name: unique identifier
- plugin: aggregation plugin to use
- trigger: TriggerConfig (count/timeout/condition/end_of_source)
- output_mode: single/passthrough/transform
- options: plugin-specific config

Engine evaluates triggers, plugins only accept/reject rows.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Add aggregations field to ElspethSettings

> **SCOPE CHECK:** Add ONE field and ONE validator. Do not reorganize ElspethSettings, do not add other fields, do not "clean up" the class while you're there.

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config_aggregation.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config_aggregation.py`:

```python
class TestElspethSettingsAggregations:
    """Tests for aggregations in ElspethSettings."""

    def test_elspeth_settings_aggregations_default_empty(self) -> None:
        """Aggregations defaults to empty list."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
        )
        assert settings.aggregations == []

    def test_elspeth_settings_with_aggregations(self) -> None:
        """Aggregations can be configured."""
        from elspeth.core.config import (
            AggregationSettings,
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
            TriggerConfig,
        )

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
            aggregations=[
                AggregationSettings(
                    name="batch_stats",
                    plugin="stats",
                    trigger=TriggerConfig(count=100),
                ),
            ],
        )
        assert len(settings.aggregations) == 1
        assert settings.aggregations[0].name == "batch_stats"

    def test_elspeth_settings_rejects_duplicate_aggregation_names(self) -> None:
        """Aggregation names must be unique."""
        from elspeth.core.config import (
            AggregationSettings,
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
            TriggerConfig,
        )

        with pytest.raises(ValidationError, match="duplicate.*name"):
            ElspethSettings(
                datasource=DatasourceSettings(plugin="csv"),
                sinks={"output": SinkSettings(plugin="csv")},
                output_sink="output",
                aggregations=[
                    AggregationSettings(
                        name="batch_stats",
                        plugin="stats",
                        trigger=TriggerConfig(count=100),
                    ),
                    AggregationSettings(
                        name="batch_stats",  # Duplicate!
                        plugin="other_stats",
                        trigger=TriggerConfig(timeout_seconds=30),
                    ),
                ],
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_config_aggregation.py::TestElspethSettingsAggregations::test_elspeth_settings_aggregations_default_empty -v`

Expected: FAIL with `TypeError: ElspethSettings.__init__() got an unexpected keyword argument 'aggregations'`

**Step 3: Add aggregations to ElspethSettings**

In `src/elspeth/core/config.py`, find `ElspethSettings` class and add the field after `gates`:

```python
# Optional - aggregations (config-driven batching)
aggregations: list[AggregationSettings] = Field(
    default_factory=list,
    description="Aggregation configurations for batching rows",
)
```

**Step 4: Add validator for unique aggregation names**

Add to `ElspethSettings` class:

```python
@model_validator(mode="after")
def validate_unique_aggregation_names(self) -> "ElspethSettings":
    """Ensure aggregation names are unique."""
    names = [agg.name for agg in self.aggregations]
    duplicates = [name for name in names if names.count(name) > 1]
    if duplicates:
        raise ValueError(f"Duplicate aggregation name(s): {set(duplicates)}")
    return self
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/core/test_config_aggregation.py::TestElspethSettingsAggregations -v`

Expected: All 3 tests pass

**Step 6: Commit**

```bash
git add -A && git commit -m "feat(config): add aggregations field to ElspethSettings

Allows config-driven aggregation definitions:
- aggregations: list[AggregationSettings]
- Validates uniqueness of aggregation names

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Create TriggerEvaluator in engine

**Files:**
- Create: `src/elspeth/engine/triggers.py`
- Test: `tests/engine/test_triggers.py`

**IMPORTANT:** Per plugin-protocol.md, triggers are combinable with OR logic.
The TriggerEvaluator must check ALL configured triggers and return True if ANY fires.

**Step 1: Write the failing test**

Create `tests/engine/test_triggers.py`:

```python
"""Tests for TriggerEvaluator."""

import time

import pytest


class TestTriggerEvaluator:
    """Tests for TriggerEvaluator class.

    Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
    The evaluator checks ALL configured triggers with OR logic.
    """

    def test_count_trigger_not_reached(self) -> None:
        """Count trigger returns False when threshold not reached."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        # Accept 50 rows - should not trigger
        for _ in range(50):
            evaluator.record_accept()

        assert evaluator.should_trigger() is False

    def test_count_trigger_reached(self) -> None:
        """Count trigger returns True when threshold reached."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        # Accept 100 rows - should trigger
        for _ in range(100):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True

    def test_count_trigger_exceeded(self) -> None:
        """Count trigger returns True when threshold exceeded."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        for _ in range(150):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True

    def test_timeout_trigger_not_reached(self) -> None:
        """Timeout trigger returns False when time not exceeded."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(timeout_seconds=10.0)
        evaluator = TriggerEvaluator(config)

        evaluator.record_accept()  # Start the timer
        # Immediately check - should not trigger
        assert evaluator.should_trigger() is False

    def test_timeout_trigger_reached(self) -> None:
        """Timeout trigger returns True when time exceeded."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(timeout_seconds=0.01)  # 10ms
        evaluator = TriggerEvaluator(config)

        evaluator.record_accept()  # Start the timer
        time.sleep(0.02)  # Wait 20ms
        assert evaluator.should_trigger() is True

    def test_condition_trigger_not_met(self) -> None:
        """Condition trigger returns False when condition not met."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(condition="batch_count >= 50")
        evaluator = TriggerEvaluator(config)

        for _ in range(30):
            evaluator.record_accept()

        assert evaluator.should_trigger() is False

    def test_condition_trigger_met(self) -> None:
        """Condition trigger returns True when condition met."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(condition="batch_count >= 50")
        evaluator = TriggerEvaluator(config)

        for _ in range(50):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True

    def test_condition_trigger_with_age(self) -> None:
        """Condition trigger can use batch_age_seconds."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(
            condition="batch_count >= 10 and batch_age_seconds > 0.01",
        )
        evaluator = TriggerEvaluator(config)

        for _ in range(15):
            evaluator.record_accept()

        # Condition uses AND, need both count and age
        time.sleep(0.02)
        assert evaluator.should_trigger() is True

    def test_combined_count_and_timeout_count_wins(self) -> None:
        """Combined triggers: count fires first (OR logic)."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        # Count = 10, Timeout = 1 hour
        config = TriggerConfig(count=10, timeout_seconds=3600.0)
        evaluator = TriggerEvaluator(config)

        # Accept 10 rows - count trigger fires
        for _ in range(10):
            evaluator.record_accept()

        result = evaluator.should_trigger()
        assert result is True
        assert evaluator.which_triggered() == "count"

    def test_combined_count_and_timeout_timeout_wins(self) -> None:
        """Combined triggers: timeout fires first (OR logic)."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        # Count = 1000, Timeout = 10ms
        config = TriggerConfig(count=1000, timeout_seconds=0.01)
        evaluator = TriggerEvaluator(config)

        # Accept 5 rows (way under count threshold)
        for _ in range(5):
            evaluator.record_accept()

        # Wait for timeout
        time.sleep(0.02)

        result = evaluator.should_trigger()
        assert result is True
        assert evaluator.which_triggered() == "timeout"

    def test_combined_all_triggers_count_wins(self) -> None:
        """Combined count + timeout + condition: count fires first."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(
            count=5,
            timeout_seconds=3600.0,
            condition="batch_count >= 1000",  # Never fires
        )
        evaluator = TriggerEvaluator(config)

        for _ in range(5):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True
        assert evaluator.which_triggered() == "count"

    def test_combined_none_fire_yet(self) -> None:
        """Combined triggers: none fire until at least one condition met."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(
            count=100,
            timeout_seconds=3600.0,
        )
        evaluator = TriggerEvaluator(config)

        # Accept 10 rows (under threshold), immediately check (under timeout)
        for _ in range(10):
            evaluator.record_accept()

        assert evaluator.should_trigger() is False
        assert evaluator.which_triggered() is None

    def test_reset_clears_state(self) -> None:
        """Reset clears batch count and timer."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        for _ in range(100):
            evaluator.record_accept()

        assert evaluator.should_trigger() is True

        evaluator.reset()

        assert evaluator.should_trigger() is False
        assert evaluator.batch_count == 0

    def test_batch_count_property(self) -> None:
        """batch_count property returns current count."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=100)
        evaluator = TriggerEvaluator(config)

        assert evaluator.batch_count == 0

        for _ in range(42):
            evaluator.record_accept()

        assert evaluator.batch_count == 42

    def test_batch_age_seconds_property(self) -> None:
        """batch_age_seconds returns time since first accept."""
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(timeout_seconds=10.0)
        evaluator = TriggerEvaluator(config)

        # Before any accepts, age is 0
        assert evaluator.batch_age_seconds == 0.0

        evaluator.record_accept()
        time.sleep(0.01)

        # After accept and sleep, age should be > 0
        assert evaluator.batch_age_seconds > 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_triggers.py::TestTriggerEvaluator::test_count_trigger_not_reached -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.engine.triggers'`

**Step 3: Implement TriggerEvaluator**

Create `src/elspeth/engine/triggers.py`:

```python
"""Trigger evaluation for aggregation batches.

Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
The TriggerEvaluator evaluates all configured triggers with OR logic.

The engine creates one evaluator per aggregation and calls should_trigger()
after each accept. When should_trigger() returns True, which_triggered()
indicates which trigger fired (for audit trail).

Trigger types:
- count: Fires when batch_count >= threshold
- timeout: Fires when batch_age_seconds >= timeout_seconds
- condition: Fires when custom expression evaluates to True
- end_of_source: Implicit - engine handles at source exhaustion (not in TriggerConfig)
"""

import time
from typing import Literal

from elspeth.contracts.enums import TriggerType
from elspeth.core.config import TriggerConfig
from elspeth.engine.expression_parser import ExpressionParser


class TriggerEvaluator:
    """Evaluates trigger conditions for an aggregation batch.

    Per plugin-protocol.md: Triggers are combinable (first to fire wins).
    All configured triggers are evaluated with OR logic.

    Created by engine for each aggregation. Tracks batch state (count, age)
    and evaluates whether ANY configured trigger condition is met.

    Example:
        evaluator = TriggerEvaluator(TriggerConfig(count=100, timeout_seconds=60))

        for row in rows:
            if aggregation.accept(row).accepted:
                evaluator.record_accept()
                if evaluator.should_trigger():
                    print(f"Triggered by: {evaluator.which_triggered()}")
                    aggregation.flush()
                    evaluator.reset()
    """

    def __init__(self, config: TriggerConfig) -> None:
        """Initialize evaluator with trigger configuration.

        Args:
            config: Trigger configuration from AggregationSettings
        """
        self._config = config
        self._batch_count = 0
        self._first_accept_time: float | None = None
        self._last_triggered: Literal["count", "timeout", "condition"] | None = None

        # Pre-parse condition expression if applicable
        self._condition_parser: ExpressionParser | None = None
        if config.condition is not None:
            self._condition_parser = ExpressionParser(config.condition)

    @property
    def batch_count(self) -> int:
        """Current number of accepted rows in batch."""
        return self._batch_count

    @property
    def batch_age_seconds(self) -> float:
        """Seconds since first accept in this batch."""
        if self._first_accept_time is None:
            return 0.0
        return time.monotonic() - self._first_accept_time

    def record_accept(self) -> None:
        """Record that a row was accepted into the batch.

        Call this after each successful accept. Updates batch_count and
        starts the timer on first accept.
        """
        self._batch_count += 1
        if self._first_accept_time is None:
            self._first_accept_time = time.monotonic()

    def should_trigger(self) -> bool:
        """Evaluate whether ANY trigger condition is met (OR logic).

        Returns:
            True if any configured trigger should fire, False otherwise.

        Side effect:
            Sets _last_triggered to the trigger type that fired.
        """
        self._last_triggered = None

        # Check count trigger
        if self._config.count is not None:
            if self._batch_count >= self._config.count:
                self._last_triggered = "count"
                return True

        # Check timeout trigger
        if self._config.timeout_seconds is not None:
            if self.batch_age_seconds >= self._config.timeout_seconds:
                self._last_triggered = "timeout"
                return True

        # Check condition trigger
        if self._condition_parser is not None:
            context = {
                "batch_count": self._batch_count,
                "batch_age_seconds": self.batch_age_seconds,
            }
            result = self._condition_parser.evaluate(context)
            if bool(result):
                self._last_triggered = "condition"
                return True

        return False

    def which_triggered(self) -> Literal["count", "timeout", "condition"] | None:
        """Return which trigger fired on the last should_trigger() call.

        Returns:
            "count", "timeout", or "condition" if a trigger fired.
            None if no trigger fired.

        Note:
            This is used for the audit trail (TriggerType.COUNT, etc.)
        """
        return self._last_triggered

    def get_trigger_type(self) -> TriggerType | None:
        """Get TriggerType enum for the trigger that fired.

        Returns:
            TriggerType enum if a trigger fired, None otherwise.
        """
        if self._last_triggered == "count":
            return TriggerType.COUNT
        elif self._last_triggered == "timeout":
            return TriggerType.TIMEOUT
        elif self._last_triggered == "condition":
            return TriggerType.CONDITION
        return None

    def reset(self) -> None:
        """Reset state for a new batch.

        Call this after flush completes to prepare for the next batch.
        """
        self._batch_count = 0
        self._first_accept_time = None
        self._last_triggered = None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_triggers.py::TestTriggerEvaluator -v`

Expected: All 16 tests pass

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(engine): add TriggerEvaluator for config-driven aggregation triggers

Per plugin-protocol.md, triggers are combinable (first to fire wins):
- Evaluates ALL configured triggers with OR logic
- which_triggered() reports which trigger fired for audit trail
- get_trigger_type() returns TriggerType enum for Landscape recording

end_of_source is implicit and handled by engine at source exhaustion.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Delete AcceptResult.trigger field

> **SCOPE CHECK:** Delete the `trigger` field and fix ALL references. Do NOT refactor AcceptResult, do NOT add new fields, do NOT "improve" tests while fixing them - just remove the trigger argument/assertion.

**Files:**
- Modify: `src/elspeth/contracts/results.py`
- Modify: Tests that reference `AcceptResult.trigger`

**Step 1: Find all references to AcceptResult.trigger**

Run:
```bash
grep -r "\.trigger" src/elspeth --include="*.py" | grep -v "__pycache__"
grep -r "\.trigger" tests --include="*.py" | grep -v "__pycache__"
grep -r "trigger=" src/elspeth --include="*.py" | grep -v "__pycache__" | grep -i accept
```

**Step 2: Write a test that AcceptResult has no trigger field**

Add to `tests/contracts/test_results.py` (or create if it doesn't exist):

```python
def test_accept_result_has_no_trigger_field() -> None:
    """AcceptResult should NOT have a trigger field (moved to engine)."""
    from dataclasses import fields

    from elspeth.contracts.results import AcceptResult

    field_names = [f.name for f in fields(AcceptResult)]
    assert "trigger" not in field_names, "trigger field should be removed (WP-06)"
```

**Step 3: Delete trigger field from AcceptResult**

In `src/elspeth/contracts/results.py`, change:

```python
@dataclass
class AcceptResult:
    """Result of aggregation accept check.

    Indicates whether the row was accepted into a batch.
    """

    accepted: bool
    trigger: bool  # DELETE THIS LINE
    batch_id: str | None = field(default=None, repr=False)
```

To:

```python
@dataclass
class AcceptResult:
    """Result of aggregation accept check.

    Indicates whether the row was accepted into a batch.
    The engine evaluates trigger conditions separately (WP-06).
    """

    accepted: bool
    batch_id: str | None = field(default=None, repr=False)
```

**Step 4: Fix all call sites**

Search and fix each location where `AcceptResult` is created with `trigger=`:

```bash
grep -rn "AcceptResult(" src/elspeth tests --include="*.py"
```

For each location:
- Remove `trigger=True` or `trigger=False` from the constructor call
- If code checks `result.trigger`, remove that check (engine handles this now)

Example fix in processor.py (around line 242):

BEFORE:
```python
if accept_result.trigger:
    self._aggregation_executor.flush(...)
```

AFTER (REMOVE the if block - engine controls triggering):
```python
# Trigger evaluation moved to engine (WP-06)
# The processor no longer decides when to flush
```

**Step 5: Run tests**

Run: `pytest tests/ -v -k "accept" --tb=short`

Fix any remaining references to `trigger` field.

**Step 6: Run full test suite**

Run: `pytest tests/ -v`

**Step 7: Commit**

```bash
git add -A && git commit -m "refactor(contracts): remove AcceptResult.trigger field

Trigger evaluation moved from plugin to engine (WP-06).
Plugins only return accepted/rejected - engine uses TriggerEvaluator
to decide when to flush batches.

BREAKING: AcceptResult no longer has 'trigger' field.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Delete BaseAggregation.should_trigger() and reset()

> **SCOPE CHECK:** Delete TWO methods and fix ALL test mocks. Do NOT refactor BaseAggregation, do NOT change accept() or flush() signatures, do NOT "clean up" the class. Mock aggregations in tests: just delete their should_trigger() method.

**Files:**
- Modify: `src/elspeth/plugins/base.py`
- Modify: Any tests or implementations referencing these methods

**Step 1: Find all references**

Run:
```bash
grep -rn "should_trigger" src/elspeth tests --include="*.py"
grep -rn "\.reset\(\)" src/elspeth tests --include="*.py" | grep -i aggreg
```

**Step 2: Write test that methods don't exist**

Add to `tests/plugins/test_base.py`:

```python
def test_base_aggregation_no_should_trigger() -> None:
    """BaseAggregation should NOT have should_trigger() (moved to engine)."""
    from elspeth.plugins.base import BaseAggregation

    assert not hasattr(BaseAggregation, "should_trigger"), (
        "should_trigger() should be removed (WP-06)"
    )


def test_base_aggregation_no_reset() -> None:
    """BaseAggregation should NOT have reset() (engine manages batch lifecycle)."""
    from elspeth.plugins.base import BaseAggregation

    assert not hasattr(BaseAggregation, "reset"), (
        "reset() should be removed (WP-06)"
    )
```

**Step 3: Delete should_trigger() and reset() from BaseAggregation**

In `src/elspeth/plugins/base.py`, find `BaseAggregation` class and DELETE these methods:

```python
# DELETE THIS METHOD
@abstractmethod
def should_trigger(self) -> bool:
    """Check if batch should flush."""
    ...

# DELETE THIS METHOD
def reset(self) -> None:  # noqa: B027
    """Reset internal state.

    Override if you have state beyond what __init__ sets up.
    """
```

**Step 4: Update the class docstring**

Update the `BaseAggregation` docstring to remove references to `should_trigger()`:

```python
class BaseAggregation(ABC):
    """Base class for aggregation transforms (stateful batching).

    Subclass and implement accept() and flush().

    Phase 3 Integration:
    - Engine creates Landscape batch on first accept()
    - Engine persists batch membership on every accept()
    - Engine manages batch state transitions
    - Engine evaluates trigger conditions (WP-06)

    Example:
        class StatsAggregation(BaseAggregation):
            name = "stats"
            input_schema = InputSchema
            output_schema = StatsSchema

            def __init__(self, config):
                super().__init__(config)
                self._values = []

            def accept(self, row, ctx) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True)

            def flush(self, ctx) -> list[dict]:
                result = {"mean": statistics.mean(self._values)}
                self._values = []
                return [result]
    """
```

**Step 5: Fix any aggregation implementations**

Search for classes that extend `BaseAggregation` and implement `should_trigger()`:

```bash
grep -rn "def should_trigger" src/elspeth tests --include="*.py"
```

For each implementation:
- Delete the `should_trigger()` method
- Delete the `reset()` method if present
- Update `accept()` to not set trigger (already handled in Task 5)

**Step 6: Run tests**

Run: `pytest tests/plugins/ -v`

**Step 7: Commit**

```bash
git add -A && git commit -m "refactor(plugins): remove should_trigger() and reset() from BaseAggregation

Trigger evaluation and batch lifecycle moved to engine (WP-06).
Aggregation plugins only need accept() and flush() methods now.

BREAKING: BaseAggregation no longer has should_trigger() or reset().

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Integrate TriggerEvaluator into AggregationExecutor

> **SCOPE CHECK:** Add `aggregation_settings` parameter, create evaluators, add `should_flush()` method, update `accept()` and `flush()`. Do NOT refactor existing executor code, do NOT add new methods beyond what's specified, do NOT "improve" the batch lifecycle.

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Test: `tests/engine/test_executors.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_executors.py`:

```python
class TestAggregationExecutorTriggers:
    """Tests for config-driven trigger evaluation in AggregationExecutor."""

    def test_accept_does_not_return_trigger(self) -> None:
        """accept() returns AcceptResult without trigger field."""
        from dataclasses import fields

        from elspeth.contracts.results import AcceptResult

        field_names = [f.name for f in fields(AcceptResult)]
        assert "trigger" not in field_names

    def test_executor_evaluates_count_trigger(
        self,
        recorder: "LandscapeRecorder",
        span_factory: "SpanFactory",
        mock_aggregation: "AggregationProtocol",
        ctx: "PluginContext",
    ) -> None:
        """Executor evaluates count trigger and returns should_flush."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=3),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id="run-1",
            aggregation_settings={mock_aggregation.node_id: settings},
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            run_id="run-1",
            row_data={"value": 1},
        )

        # First two accepts - should not trigger
        result1 = executor.accept(mock_aggregation, token, ctx, step_in_pipeline=1)
        assert result1.accepted is True
        assert executor.should_flush(mock_aggregation.node_id) is False

        result2 = executor.accept(mock_aggregation, token, ctx, step_in_pipeline=1)
        assert executor.should_flush(mock_aggregation.node_id) is False

        # Third accept - should trigger
        result3 = executor.accept(mock_aggregation, token, ctx, step_in_pipeline=1)
        assert executor.should_flush(mock_aggregation.node_id) is True

    def test_executor_reset_trigger_after_flush(
        self,
        recorder: "LandscapeRecorder",
        span_factory: "SpanFactory",
        mock_aggregation: "AggregationProtocol",
        ctx: "PluginContext",
    ) -> None:
        """Executor resets trigger state after flush."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id="run-1",
            aggregation_settings={mock_aggregation.node_id: settings},
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            run_id="run-1",
            row_data={"value": 1},
        )

        # Accept until trigger
        executor.accept(mock_aggregation, token, ctx, step_in_pipeline=1)
        executor.accept(mock_aggregation, token, ctx, step_in_pipeline=1)
        assert executor.should_flush(mock_aggregation.node_id) is True

        # Flush
        executor.flush(mock_aggregation, ctx, "count", step_in_pipeline=1)

        # After flush, trigger should be reset
        assert executor.should_flush(mock_aggregation.node_id) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_executors.py::TestAggregationExecutorTriggers -v`

Expected: FAIL (either test fixture issues or missing aggregation_settings parameter)

**Step 3: Update AggregationExecutor**

In `src/elspeth/engine/executors.py`, modify `AggregationExecutor`:

```python
from elspeth.core.config import AggregationSettings
from elspeth.engine.triggers import TriggerEvaluator


class AggregationExecutor:
    """Executes aggregations with batch tracking and audit recording.

    Manages the lifecycle of batches:
    1. Create batch on first accept (if _batch_id is None)
    2. Track batch members as rows are accepted
    3. Evaluate trigger conditions after each accept (WP-06)
    4. Transition batch through states: draft -> executing -> completed/failed
    5. Reset trigger evaluator and batch_id after flush for next batch

    CRITICAL: Terminal state CONSUMED_IN_BATCH is DERIVED from batch_members table,
    NOT stored in node_states.status (which is always "completed" for successful accepts).
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
        *,
        aggregation_settings: dict[str, AggregationSettings] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            run_id: Run identifier for batch creation
            aggregation_settings: Map of node_id -> AggregationSettings (WP-06)
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._member_counts: dict[str, int] = {}  # batch_id -> count for ordinals
        self._batch_ids: dict[str, str | None] = {}  # node_id -> current batch_id
        self._aggregation_settings = aggregation_settings or {}
        self._trigger_evaluators: dict[str, TriggerEvaluator] = {}

        # Create trigger evaluators for each configured aggregation
        for node_id, settings in self._aggregation_settings.items():
            self._trigger_evaluators[node_id] = TriggerEvaluator(settings.trigger)

    def should_flush(self, node_id: str) -> bool:
        """Check if the aggregation should flush based on trigger config.

        Args:
            node_id: Aggregation node ID

        Returns:
            True if trigger condition is met, False otherwise
        """
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is None:
            # No config - never trigger (legacy behavior or end_of_source only)
            return False
        return evaluator.should_trigger()

    def accept(
        self,
        aggregation: AggregationProtocol,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
    ) -> AcceptResult:
        """Accept a row into an aggregation batch.

        Creates batch on first accept (if no batch_id for this node).
        Records batch membership for accepted rows.
        Updates trigger evaluator if row is accepted.

        Args:
            aggregation: Aggregation plugin to execute
            token: Current token with row data
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)

        Returns:
            AcceptResult with accepted flag and batch_id
        """
        node_id = aggregation.node_id
        assert node_id is not None, "node_id must be set by orchestrator"

        # ... existing batch creation and accept logic ...

        # After successful accept, update trigger evaluator
        if result.accepted:
            evaluator = self._trigger_evaluators.get(node_id)
            if evaluator is not None:
                evaluator.record_accept()

        return result

    def flush(
        self,
        aggregation: AggregationProtocol,
        ctx: PluginContext,
        trigger_reason: str,
        step_in_pipeline: int,
    ) -> list[dict[str, Any]]:
        """Flush an aggregation batch and return output rows.

        Transitions batch through: draft -> executing -> completed/failed.
        Resets batch_id and trigger evaluator for next batch.
        """
        # ... existing flush logic ...

        # After successful flush, reset trigger evaluator
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is not None:
            evaluator.reset()

        return outputs
```

**Step 4: Run tests**

Run: `pytest tests/engine/test_executors.py -v`

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(engine): integrate TriggerEvaluator into AggregationExecutor

AggregationExecutor now:
- Accepts aggregation_settings dict mapping node_id -> AggregationSettings
- Creates TriggerEvaluator for each configured aggregation
- Updates evaluator on accept, resets on flush
- Exposes should_flush() for engine to query trigger state

Engine calls should_flush() after each accept to decide when to flush.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Update RowProcessor to use engine-controlled triggers

> **SCOPE CHECK:** Add `aggregation_settings` parameter, pass to executor, call `should_flush()` after accept, call `flush()` when triggered. Do NOT refactor RowProcessor, do NOT change non-aggregation code paths, do NOT add new abstractions.

**Files:**
- Modify: `src/elspeth/engine/processor.py`
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_processor.py`:

```python
class TestProcessorAggregationTriggers:
    """Tests for config-driven aggregation triggers in RowProcessor."""

    def test_processor_flushes_on_count_trigger(
        self,
        # ... fixtures ...
    ) -> None:
        """Processor flushes aggregation when count trigger reached."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        aggregation_settings = {
            "agg-node-1": AggregationSettings(
                name="test_agg",
                plugin="test",
                trigger=TriggerConfig(count=3),
            ),
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=span_factory,
            run_id="run-1",
            source_node_id="source-1",
            aggregation_settings=aggregation_settings,
        )

        # Process 3 rows - should trigger flush after 3rd
        for i in range(3):
            results = processor.process_row(
                row_index=i,
                row_data={"value": i},
                transforms=[mock_aggregation],
                ctx=ctx,
            )

        # Verify flush was called
        assert mock_aggregation.flush.call_count == 1
```

**Step 2: Update RowProcessor**

In `src/elspeth/engine/processor.py`, update the `__init__` and aggregation handling:

```python
def __init__(
    self,
    recorder: LandscapeRecorder,
    span_factory: SpanFactory,
    run_id: str,
    source_node_id: str,
    *,
    edge_map: dict[tuple[str, str], str] | None = None,
    route_resolution_map: dict[tuple[str, str], str] | None = None,
    config_gates: list[GateSettings] | None = None,
    config_gate_id_map: dict[str, str] | None = None,
    aggregation_settings: dict[str, AggregationSettings] | None = None,  # ADD THIS
) -> None:
    # ... existing init ...
    self._aggregation_executor = AggregationExecutor(
        recorder,
        span_factory,
        run_id,
        aggregation_settings=aggregation_settings,  # Pass settings
    )
```

Update aggregation handling in `_process_single_token`:

```python
elif isinstance(transform, BaseAggregation):
    # Aggregation transform
    accept_result = self._aggregation_executor.accept(
        aggregation=transform,
        token=current_token,
        ctx=ctx,
        step_in_pipeline=step,
    )

    # Engine evaluates trigger (WP-06)
    if self._aggregation_executor.should_flush(transform.node_id):
        self._aggregation_executor.flush(
            aggregation=transform,
            ctx=ctx,
            trigger_reason=self._get_trigger_reason(transform.node_id),
            step_in_pipeline=step,
        )

    return (
        RowResult(
            token=current_token,
            final_data=current_token.row_data,
            outcome=RowOutcome.CONSUMED_IN_BATCH,
        ),
        child_items,
    )
```

Add helper method:

```python
def _get_trigger_reason(self, node_id: str) -> str:
    """Get the trigger reason string for audit trail."""
    evaluator = self._aggregation_executor._trigger_evaluators.get(node_id)
    if evaluator is None:
        return "manual"
    return evaluator.trigger_type.value
```

**Step 3: Run tests**

Run: `pytest tests/engine/test_processor.py -v`

**Step 4: Commit**

```bash
git add -A && git commit -m "feat(engine): RowProcessor uses engine-controlled aggregation triggers

RowProcessor now:
- Accepts aggregation_settings parameter
- Passes settings to AggregationExecutor
- Queries should_flush() after each accept
- Flushes when trigger condition is met
- Records trigger_type in audit trail

Completes WP-06: config-driven aggregation triggers.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Run full verification

**Step 1: Run mypy**

```bash
mypy src/elspeth/core/config.py src/elspeth/engine/triggers.py src/elspeth/engine/executors.py src/elspeth/engine/processor.py src/elspeth/contracts/results.py src/elspeth/plugins/base.py --strict
```

**Step 2: Run all affected tests**

```bash
pytest tests/core/test_config_aggregation.py tests/engine/test_triggers.py tests/engine/test_executors.py tests/engine/test_processor.py tests/contracts/ tests/plugins/ -v
```

**Step 3: Verify no stale references**

```bash
# Should return nothing
grep -rn "AcceptResult.*trigger" src/elspeth tests --include="*.py" | grep -v "__pycache__"
grep -rn "should_trigger" src/elspeth tests --include="*.py" | grep -v "__pycache__" | grep -v test_base_aggregation
grep -rn "\.reset\(\)" src/elspeth tests --include="*.py" | grep -v "__pycache__" | grep -i aggreg
```

**Step 4: Run full test suite**

```bash
pytest tests/ -v
```

**Step 5: Final commit**

```bash
git add -A && git commit -m "chore: verify WP-06 aggregation triggers complete

Verification:
- TriggerConfig model with count/timeout/condition/end_of_source
- AggregationSettings model with trigger and output_mode
- TriggerEvaluator evaluates trigger conditions
- AggregationExecutor integrates trigger evaluation
- RowProcessor uses engine-controlled triggers
- AcceptResult.trigger field removed
- BaseAggregation.should_trigger() and reset() removed
- All tests pass
- No stale references to removed APIs

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

**Per plugin-protocol.md: Triggers are combinable (first to fire wins)**

- [ ] `TriggerConfig` model has optional fields: count, timeout_seconds, condition
- [ ] `TriggerConfig` requires at least one trigger configured
- [ ] `TriggerConfig` validates count > 0, timeout_seconds > 0
- [ ] `TriggerConfig` validates condition expression syntax
- [ ] `TriggerConfig.has_count`, `has_timeout`, `has_condition` properties work
- [ ] `AggregationSettings` model has name, plugin, trigger, output_mode, options
- [ ] `ElspethSettings.aggregations` field exists and validates unique names
- [ ] `TriggerEvaluator` class exists in `engine/triggers.py`
- [ ] `TriggerEvaluator.should_trigger()` evaluates ALL triggers with OR logic
- [ ] `TriggerEvaluator.which_triggered()` returns which trigger fired
- [ ] `TriggerEvaluator.get_trigger_type()` returns TriggerType enum
- [ ] `TriggerEvaluator.reset()` clears state
- [ ] `AggregationExecutor` accepts `aggregation_settings` parameter
- [ ] `AggregationExecutor.should_flush()` method exists
- [ ] `RowProcessor` queries `should_flush()` after each accept
- [ ] `AcceptResult.trigger` field is DELETED
- [ ] `BaseAggregation.should_trigger()` is DELETED
- [ ] `BaseAggregation.reset()` is DELETED
- [ ] No references to removed APIs remain
- [ ] `mypy --strict` passes on all modified files
- [ ] All tests pass
- [ ] end_of_source handled implicitly by engine at source exhaustion

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/core/config.py` | MODIFY | Add TriggerConfig, AggregationSettings, aggregations field |
| `src/elspeth/engine/triggers.py` | CREATE | TriggerEvaluator class |
| `src/elspeth/engine/executors.py` | MODIFY | Integrate TriggerEvaluator into AggregationExecutor |
| `src/elspeth/engine/processor.py` | MODIFY | Use engine-controlled triggers |
| `src/elspeth/contracts/results.py` | MODIFY | DELETE trigger field from AcceptResult |
| `src/elspeth/plugins/base.py` | MODIFY | DELETE should_trigger() and reset() |
| `tests/core/test_config_aggregation.py` | CREATE | Tests for TriggerConfig, AggregationSettings |
| `tests/engine/test_triggers.py` | CREATE | Tests for TriggerEvaluator |
| `tests/engine/test_executors.py` | MODIFY | Add trigger integration tests |
| `tests/engine/test_processor.py` | MODIFY | Add trigger integration tests |
| `tests/contracts/test_results.py` | MODIFY | Test trigger field removed |
| `tests/plugins/test_base.py` | MODIFY | Test methods removed |

---

## Dependency Notes

- **Depends on:** WP-05 (TriggerType enum must exist)
- **Unlocks:** WP-14 (Engine Test Rewrites) - partial
- **Risk:** Medium - breaking changes to AcceptResult and BaseAggregation contracts
- **Migration:** Any existing aggregation plugins need to remove should_trigger() and reset() implementations
