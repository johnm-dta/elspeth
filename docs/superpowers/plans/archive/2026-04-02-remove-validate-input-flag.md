# Remove `validate_input` Flag — Unconditional Executor Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the opt-in `validate_input` flag from all plugins, protocols, and base classes so the executor validates every transform/sink input unconditionally — enforcing the Tier 2 trust model.

**Architecture:** The executor already has centralized validation logic that calls `input_schema.model_validate()` on incoming data. Currently this is gated behind `if plugin.validate_input:`. We remove the gate, hoist the deferred `ValidationError` import to module level, and delete the `validate_input` attribute from the entire plugin stack (protocols, base classes, configs, `__init__` methods). Tests that asserted "wrong types pass through" get deleted; tests that asserted "wrong types crash" get simplified to not set the flag.

**Tech Stack:** Python, Pydantic, pluggy, pytest

**Spec:** `docs/superpowers/specs/2026-04-02-remove-validate-input-flag-design.md`

---

### Task 1: Make transform executor validation unconditional

**Files:**
- Modify: `src/elspeth/engine/executors/transform.py:1-4` (imports), `src/elspeth/engine/executors/transform.py:225-237` (validation block)

- [ ] **Step 1: Modify the transform executor**

In `src/elspeth/engine/executors/transform.py`, add `ValidationError` to the module-level imports (line 1-4 area):

```python
from pydantic import ValidationError
```

Then replace the conditional validation block (lines 225-237):

```python
            # --- INPUT VALIDATION (pre-execution) ---
            # Centralized check: if transform has validate_input=True,
            # validate input against its input_schema before calling process().
            if transform.validate_input:
                from pydantic import ValidationError

                try:
                    transform.input_schema.model_validate(input_dict)
                except ValidationError as e:
                    raise PluginContractViolation(
                        f"Transform '{transform.name}' input validation failed: {e}. "
                        f"This indicates an upstream transform/source schema bug."
                    ) from e
```

With the unconditional version:

```python
            # --- INPUT VALIDATION (pre-execution) ---
            # Validate input against input_schema before calling process().
            # Wrong types at a transform boundary are upstream plugin bugs (Tier 2).
            try:
                transform.input_schema.model_validate(input_dict)
            except ValidationError as e:
                raise PluginContractViolation(
                    f"Transform '{transform.name}' input validation failed: {e}. "
                    f"This indicates an upstream transform/source schema bug."
                ) from e
```

- [ ] **Step 2: Run transform executor tests**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_executors.py::TestTransformExecutor -x -q`
Expected: Some failures because mock transforms still set `validate_input = False` and the executor no longer reads it. That's fine — Task 5 fixes the tests.

---

### Task 2: Make sink executor validation unconditional

**Files:**
- Modify: `src/elspeth/engine/executors/sink.py:1-8` (imports), `src/elspeth/engine/executors/sink.py:293-303` (validation block)

- [ ] **Step 1: Modify the sink executor**

In `src/elspeth/engine/executors/sink.py`, add `ValidationError` to the module-level imports:

```python
from pydantic import ValidationError
```

Then replace the conditional validation block (lines 293-303):

```python
                    # Centralized input validation (before sink.write)
                    if sink.validate_input:
                        from pydantic import ValidationError

                        for row in rows:
                            try:
                                sink.input_schema.model_validate(row)
                            except ValidationError as e:
                                raise PluginContractViolation(
                                    f"Sink '{sink.name}' input validation failed: {e}. This indicates an upstream transform/source schema bug."
                                ) from e
```

With the unconditional version:

```python
                    # Centralized input validation (before sink.write).
                    # Wrong types at a sink boundary are upstream plugin bugs (Tier 2).
                    for row in rows:
                        try:
                            sink.input_schema.model_validate(row)
                        except ValidationError as e:
                            raise PluginContractViolation(
                                f"Sink '{sink.name}' input validation failed: {e}. This indicates an upstream transform/source schema bug."
                            ) from e
```

- [ ] **Step 2: Run sink executor tests**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_executors.py::TestSinkExecutor -x -q`
Expected: Some failures from mock sinks still having `validate_input`. Task 5 fixes tests.

---

### Task 3: Remove `validate_input` from protocols and base classes

**Files:**
- Modify: `src/elspeth/contracts/plugin_protocols.py:219-222, 445-448`
- Modify: `src/elspeth/plugins/infrastructure/base.py:161-163, 413-415`

- [ ] **Step 1: Remove from TransformProtocol**

In `src/elspeth/contracts/plugin_protocols.py`, delete lines 219-222:

```python
    # Input validation (centralized in TransformExecutor).
    # When True, executor validates input against input_schema before process().
    # Defaults to False — only enabled via plugin config (validate_input: true).
    validate_input: bool
```

- [ ] **Step 2: Remove from SinkProtocol**

In `src/elspeth/contracts/plugin_protocols.py`, delete lines 445-448:

```python
    # Input validation (centralized in SinkExecutor).
    # When True, executor validates input against input_schema before write().
    # Defaults to False — only enabled via plugin config (validate_input: true).
    validate_input: bool
```

- [ ] **Step 3: Remove from BaseTransform**

In `src/elspeth/plugins/infrastructure/base.py`, delete lines 161-163:

```python
    # Input validation (centralized in TransformExecutor).
    # When True, executor validates input against input_schema before process().
    validate_input: bool = False
```

- [ ] **Step 4: Remove from BaseSink**

In `src/elspeth/plugins/infrastructure/base.py`, delete lines 413-415:

```python
    # Input validation (centralized in SinkExecutor).
    # When True, executor validates input against input_schema before write().
    validate_input: bool = False
```

- [ ] **Step 5: Run mypy on modified files**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/plugin_protocols.py src/elspeth/plugins/infrastructure/base.py --no-error-summary`
Expected: PASS (no type errors from removing the attributes)

---

### Task 4: Remove `validate_input` from all plugins

**Files:**
- Modify: `src/elspeth/plugins/transforms/field_mapper.py:92`
- Modify: `src/elspeth/plugins/transforms/passthrough.py:21-53`
- Modify: `src/elspeth/plugins/sinks/database_sink.py:123`
- Modify: `src/elspeth/plugins/sinks/csv_sink.py:49, 85, 204`
- Modify: `src/elspeth/plugins/sinks/json_sink.py:49, 76, 210`

- [ ] **Step 1: field_mapper.py — remove hardcoded True**

In `src/elspeth/plugins/transforms/field_mapper.py`, delete this line from `__init__` (line 92):

```python
        self.validate_input = True  # Always validate — wrong types are upstream bugs
```

Also remove the `Raises` docstring reference in `process()` (line 153):

```python
        Raises:
            ValidationError: If validate_input=True and row fails schema validation.
                This indicates a bug in the upstream source/transform.
```

Replace with:

```python
        Raises:
            PluginContractViolation: Raised by executor if row fails input schema
                validation. This indicates a bug in the upstream source/transform.
```

- [ ] **Step 2: passthrough.py — remove config field and init assignment**

In `src/elspeth/plugins/transforms/passthrough.py`, delete the `validate_input` field from `PassThroughConfig` (lines 28-31):

```python
    validate_input: bool = Field(
        default=False,
        description="If True, validate input against schema (default: False)",
    )
```

Delete the init assignment (line 53):

```python
        self.validate_input = cfg.validate_input
```

Update the class docstring (line 44) — remove:

```python
        validate_input: If True, validate input against schema (default: False)
```

Update the process() docstring (lines 69-70) — replace:

```python
        Raises:
            ValidationError: If validate_input=True and row fails schema validation.
                This indicates a bug in the upstream source/transform.
```

With:

```python
        Raises:
            PluginContractViolation: Raised by executor if row fails input schema
                validation. This indicates a bug in the upstream source/transform.
```

- [ ] **Step 3: database_sink.py — remove hardcoded True**

In `src/elspeth/plugins/sinks/database_sink.py`, delete line 123:

```python
        self.validate_input = True  # Always validate — wrong types are upstream bugs
```

- [ ] **Step 4: csv_sink.py — remove config field and init assignment**

In `src/elspeth/plugins/sinks/csv_sink.py`:

Delete from `CSVSinkConfig` (line 49):

```python
    validate_input: bool = False  # Optional runtime validation of incoming rows
```

Delete from `CSVSink` class docstring (line 85):

```python
        validate_input: Validate incoming rows against schema (default: False)
```

Delete from `CSVSink.__init__` (line 204):

```python
        self.validate_input = cfg.validate_input
```

Update the `write()` docstring `Raises:` section (line 252-254) — replace:

```python
        Raises:
            ValidationError: If validate_input=True and a row fails validation.
                This indicates a bug in an upstream transform.
```

With:

```python
        Raises:
            PluginContractViolation: Raised by executor if row fails input schema
                validation. This indicates a bug in an upstream transform.
```

- [ ] **Step 5: json_sink.py — remove config field and init assignment**

In `src/elspeth/plugins/sinks/json_sink.py`:

Delete from `JSONSinkConfig` (line 49):

```python
    validate_input: bool = False  # Optional runtime validation of incoming rows
```

Delete from `JSONSink` class docstring (line 76):

```python
        validate_input: Validate incoming rows against schema (default: False)
```

Delete from `JSONSink.__init__` (line 210):

```python
        self.validate_input = cfg.validate_input
```

Update the `write()` docstring `Raises:` section (line 257-259) — replace:

```python
        Raises:
            ValidationError: If validate_input=True and a row fails validation.
                This indicates a bug in an upstream transform.
```

With:

```python
        Raises:
            PluginContractViolation: Raised by executor if row fails input schema
                validation. This indicates a bug in an upstream transform.
```

- [ ] **Step 6: Run mypy on all modified plugin files**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/transforms/field_mapper.py src/elspeth/plugins/transforms/passthrough.py src/elspeth/plugins/sinks/database_sink.py src/elspeth/plugins/sinks/csv_sink.py src/elspeth/plugins/sinks/json_sink.py --no-error-summary`
Expected: PASS

---

### Task 5: Update all tests

**Files:**
- Modify: `tests/unit/engine/test_executors.py`
- Modify: `tests/unit/engine/test_sink_executor_diversion.py`
- Modify: `tests/property/engine/test_sink_executor_diversion_properties.py`
- Modify: `tests/integration/plugins/sinks/test_durability.py`
- Modify: `tests/fixtures/base_classes.py`
- Modify: `tests/unit/plugins/test_protocols.py`
- Modify: `tests/unit/plugins/transforms/test_field_mapper.py`
- Modify: `tests/unit/plugins/transforms/test_passthrough.py`
- Modify: `tests/unit/contracts/transform_contracts/test_passthrough_contract.py`
- Modify: `tests/unit/contracts/sink_contracts/test_csv_sink_contract.py`
- Modify: `tests/property/sinks/test_csv_sink_properties.py`
- Modify: `tests/property/sinks/test_json_sink_properties.py`

- [ ] **Step 1: test_executors.py — update mock factories**

In `tests/unit/engine/test_executors.py`, update `_make_transform()` (lines 193-198):

Remove `"validate_input"` from the MagicMock `spec` list (line 193):

```python
    t = MagicMock(spec=["name", "node_id", "on_error", "declared_output_fields", "_on_start_called", "process"])
```

Delete line 198:

```python
    t.validate_input = False
```

Update `_make_sink()` — delete line 212:

```python
    sink.validate_input = False
```

Update the batch transform factory (around line 4427) — delete:

```python
        t.validate_input = False
```

- [ ] **Step 2: test_executors.py — update transform validation tests**

Update `test_validate_input_rejects_wrong_type` (line 372) — remove `transform.validate_input = True` (line 377). The test stays otherwise identical: it creates a transform with a strict schema, feeds it bad data, and expects `PluginContractViolation`. Rename to `test_unconditional_input_validation_rejects_wrong_type`:

```python
    def test_unconditional_input_validation_rejects_wrong_type(self) -> None:
        """Executor rejects row that fails input schema validation (Tier 2)."""
        recorder = _make_recorder()
        executor = TransformExecutor(recorder, _make_span_factory(), _make_step_resolver())
        transform = _make_transform()
        from elspeth.contracts import PluginSchema

        class StrictSchema(PluginSchema):
            count: int

        transform.input_schema = StrictSchema
        token = _make_token(data={"count": "not_an_int"})
        ctx = make_context()

        with pytest.raises(PluginContractViolation, match="input validation failed"):
            executor.execute_transform(transform, token, ctx)

        transform.process.assert_not_called()
```

Delete `test_validate_input_disabled_passes_wrong_type` entirely (lines 393-410). This test asserted the bug.

- [ ] **Step 3: test_executors.py — update sink validation test**

Update `test_sink_validate_input_rejects_wrong_type` (line 2653) — remove `sink.validate_input = True` (line 2664). Rename to `test_unconditional_sink_input_validation_rejects_wrong_type`:

```python
    def test_unconditional_sink_input_validation_rejects_wrong_type(self) -> None:
        """Executor rejects row that fails sink input schema (Tier 2)."""
        from elspeth.contracts import PluginSchema

        class StrictSinkSchema(PluginSchema):
            count: int

        recorder = _make_recorder()
        executor = SinkExecutor(recorder, _make_span_factory(), run_id="test-run")
        token = _make_token(data={"count": "not_an_int"})
        sink = _make_sink()
        sink.input_schema = StrictSinkSchema
        ctx = make_context()
        pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

        with pytest.raises(PluginContractViolation, match="input validation failed"):
            executor.write(
                sink,
                [token],
                ctx,
                step_in_pipeline=5,
                sink_name="out",
                pending_outcome=pending,
            )

        sink.write.assert_not_called()
```

- [ ] **Step 4: test_sink_executor_diversion.py — remove flag from mock**

In `tests/unit/engine/test_sink_executor_diversion.py`, delete line 49:

```python
    sink.validate_input = False
```

- [ ] **Step 5: test_sink_executor_diversion_properties.py — remove flag from mocks**

In `tests/property/engine/test_sink_executor_diversion_properties.py`, delete line 63:

```python
    sink.validate_input = False
```

Delete line 146:

```python
    sink.validate_input = False
```

- [ ] **Step 6: test_durability.py — remove flag from mock**

In `tests/integration/plugins/sinks/test_durability.py`, delete line 109:

```python
        sink.validate_input = False
```

- [ ] **Step 7: base_classes.py — remove flag from test fixture**

In `tests/fixtures/base_classes.py`, delete line 142:

```python
    validate_input: bool = False
```

- [ ] **Step 8: test_protocols.py — remove flag from mock implementations**

In `tests/unit/plugins/test_protocols.py`, delete these three lines:

Line 172:
```python
            validate_input: bool = False  # Centralized in executor
```

Line 431:
```python
            validate_input: bool = False  # Centralized in executor
```

Line 512:
```python
            validate_input: bool = False  # Centralized in executor
```

- [ ] **Step 9: test_field_mapper.py — update validation tests**

In `tests/unit/plugins/transforms/test_field_mapper.py`, replace `test_validate_input_always_enabled` (written earlier in this session) with a simpler test that doesn't check an attribute:

```python
    def test_no_validate_input_attribute(self) -> None:
        """FieldMapper does not carry a validate_input attribute.

        Input validation is unconditional in the executor — plugins
        no longer control this via a flag.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"mode": "fixed", "fields": ["count: int"]},
                "mapping": {},
            }
        )

        assert not hasattr(transform, "validate_input")
```

Replace `test_validate_input_with_dynamic_schema`:

```python
    def test_dynamic_schema_accepts_any_types(self, ctx: PluginContext) -> None:
        """Dynamic schema imposes no type constraints on input.

        The executor validates unconditionally, but dynamic schemas
        accept everything — validation is a no-op.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"mode": "observed"},
                "mapping": {},
            }
        )

        result = transform.process(make_pipeline_row({"anything": "goes", "count": "string"}), ctx)
        assert result.status == "success"
```

- [ ] **Step 10: test_passthrough.py — replace validation tests**

In `tests/unit/plugins/transforms/test_passthrough.py`, delete these three tests entirely (lines 96-149):

- `test_validate_input_attribute_set_from_config`
- `test_validate_input_disabled_passes_wrong_type`
- `test_validate_input_skipped_for_dynamic_schema`

Replace with:

```python
    def test_no_validate_input_attribute(self) -> None:
        """PassThrough does not carry a validate_input attribute.

        Input validation is unconditional in the executor — plugins
        no longer control this via a flag.
        """
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough(
            {
                "schema": {"mode": "fixed", "fields": ["count: int"]},
            }
        )

        assert not hasattr(transform, "validate_input")
```

- [ ] **Step 11: test_passthrough_contract.py — update fixture and delete flag test**

In `tests/unit/contracts/transform_contracts/test_passthrough_contract.py`:

Update the `transform` fixture (lines 97-105) — remove `"validate_input": True` from the config dict:

```python
    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Create a PassThrough instance with strict schema."""
        return PassThrough(
            {
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str"],
                },
            }
        )
```

Delete `test_strict_passthrough_sets_validate_input_for_executor` entirely (lines 112-121).

- [ ] **Step 12: test_csv_sink_contract.py — delete flag test**

In `tests/unit/contracts/sink_contracts/test_csv_sink_contract.py`, delete the entire `test_strict_schema_sets_validate_input_for_executor` method (lines 379-398).

If the enclosing `TestCSVSinkValidation` class becomes empty, delete the class too.

- [ ] **Step 13: test_csv_sink_properties.py — delete flag test**

In `tests/property/sinks/test_csv_sink_properties.py`, delete `test_csv_sink_validate_input_attribute_set_from_config` entirely (lines 117-133).

- [ ] **Step 14: test_json_sink_properties.py — delete flag test**

In `tests/property/sinks/test_json_sink_properties.py`, delete `test_json_sink_validate_input_attribute_set_from_config` entirely (lines 101-118).

- [ ] **Step 15: Run all tests**

Run: `.venv/bin/python -m pytest tests/unit/engine/ tests/unit/plugins/ tests/unit/contracts/ tests/property/ tests/integration/plugins/sinks/test_durability.py -x -q`
Expected: ALL PASS

---

### Task 6: Verify, commit, and close

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: ALL PASS

- [ ] **Step 2: Run mypy**

Run: `.venv/bin/python -m mypy src/ --no-error-summary`
Expected: PASS (or pre-existing issues only)

- [ ] **Step 3: Run ruff**

Run: `.venv/bin/python -m ruff check src/elspeth/engine/executors/ src/elspeth/contracts/plugin_protocols.py src/elspeth/plugins/ --no-fix`
Expected: PASS

- [ ] **Step 4: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS (the new `from pydantic import ValidationError` in executors is an L0 import into L2 — pydantic is a third-party library, not an ELSPETH layer)

- [ ] **Step 5: Grep for stale references**

Run: `grep -rn "validate_input" src/elspeth/ --include="*.py" | grep -v "config.py" | grep -v "__pycache__"`
Expected: NO output. If any lines remain, they are stale references that need to be removed.

Also check tests:
Run: `grep -rn "validate_input" tests/ --include="*.py" | grep -v "__pycache__"`
Expected: NO output.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "fix: remove validate_input opt-in flag — executor validates unconditionally

The validate_input flag defaulted to False, letting transforms and sinks
silently forward Tier 2 type-contract violations. This contradicted the
trust model where wrong types at a transform/sink boundary are upstream
plugin bugs that must crash immediately.

Remove the flag from protocols, base classes, all plugin configs, and
both executors. Validation is now unconditional — every row is checked
against input_schema before process()/write().

Spec: docs/superpowers/specs/2026-04-02-remove-validate-input-flag-design.md"
```

- [ ] **Step 7: Reopen and close the Filigree issue**

The original bug `elspeth-a55930ba95` was closed prematurely. Use `mcp__filigree__add_comment` to document the systemic fix on that issue, and create a new issue for the systemic change if desired, or just close with an updated comment.
