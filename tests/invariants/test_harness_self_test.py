"""Meta-test for the ADR-009 §Clause 4 forward invariant.

Registers a deliberately mis-annotated fixture transform and asserts the
harness catches it. Without this, a refactor that accidentally empties the
discovery list — or that weakens the forward-invariant assertion itself —
would silently make the harness a no-op. That is the worst kind of
governance theatre: a test that always passes regardless of underlying
reality.

The test drives the live ``test_annotated_transforms_preserve_input_fields``
function rather than reimplementing its assertion logic. This means:

- A regression that weakens the assertion (e.g. removes the ``not dropped``
  check) fails this test, because the live function would no longer raise
  on the deliberately-broken fixture.
- A regression that breaks discovery (e.g. ``_annotated_pass_through_plugins``
  returns the wrong list) fails this test, because the fixture would never
  reach the parametrize.
- A regression that breaks probe-row generation (e.g. ``probe_row()`` only
  generates 0-field or 1-field rows) is detected indirectly: the fixture
  drops fields only when 2+ are present, so a too-narrow strategy would
  cause the assertion never to fire.

Earlier versions of this self-test reimplemented the assertion logic
inline; that version would still pass if the live function's assertion
was broken but the duplicate logic remained correct. The current shape
closes that gap by invoking the live function directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult


class _CaptureMetafunc:
    """Minimal ``pytest.Metafunc`` stand-in for parametrization assertions."""

    def __init__(self, fixturenames: list[str]) -> None:
        self.fixturenames = fixturenames
        self.argnames: str | None = None
        self.argvalues: list[object] | None = None
        self.ids: Any = None

    def parametrize(self, argnames: str, argvalues: list[object], ids: Any = None) -> None:
        self.argnames = argnames
        self.argvalues = argvalues
        self.ids = ids


class _DeliberatelyMisannotatedDropper(BaseTransform):
    """Test fixture: claims ``passes_through_input=True`` but drops a field.

    The harness forward invariant MUST catch this transform and report the
    dropped field. If it doesn't, the harness is broken.
    """

    name = "deliberately-misannotated-dropper"
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    config_model = TransformDataConfig
    passes_through_input = True  # LIE — this transform actually drops fields.

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = TransformDataConfig.from_dict(config, plugin_name=self.name)
        self._schema_config = cfg.schema_config
        self.input_schema, self.output_schema = self._create_schemas(cfg.schema_config, "Dropper")

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {"schema": {"mode": "observed"}}

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Drops any non-first field from the input — triggers a violation
        when the probe row has 2+ fields."""
        data = row.to_dict()
        if not data:
            return TransformResult.success(
                PipelineRow(data, row.contract),
                success_reason={"action": "passthrough-empty"},
            )
        # Keep only the first key; drop both contract entries AND payload
        # entries for the rest. The forward invariant computes
        # `runtime_observed = contract_fields & payload_fields`, so dropping
        # from both is the unambiguous violation case.
        first_key = next(iter(data))
        reduced = {first_key: data[first_key]}
        from elspeth.contracts.schema_contract import SchemaContract

        kept_fields = tuple(fc for fc in row.contract.fields if fc.normalized_name == first_key)
        reduced_contract = SchemaContract(
            mode=row.contract.mode,
            fields=kept_fields,
            locked=row.contract.locked,
        )
        return TransformResult.success(
            PipelineRow(reduced, reduced_contract),
            success_reason={"action": "reduce"},
        )


def test_harness_catches_deliberate_misannotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the harness is healthy, the live forward-invariant function must
    raise ``AssertionError`` when run against ``_DeliberatelyMisannotatedDropper``.

    Drives the live ``test_annotated_transforms_preserve_input_fields``
    function — not a local copy of its assertion — so a regression that
    weakens the live assertion is caught here.

    Mechanism: monkeypatch the plugin manager to return only the fixture,
    then call the live ``@given``-decorated test function directly with the
    fixture as ``_annotated_cls``. Hypothesis will explore probe rows; the
    fixture drops fields whenever the probe has 2+ keys, so the assertion
    fires reliably. ``pytest.raises(AssertionError)`` confirms the live
    assertion still rejects the violation.
    """
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    manager = get_shared_plugin_manager()

    def _patched_get_transforms() -> list[type[BaseTransform]]:
        return [_DeliberatelyMisannotatedDropper]

    monkeypatch.setattr(manager, "get_transforms", _patched_get_transforms)

    # Discovery sanity: the patched manager returns the fixture alone. If
    # this fails, the rest of the test is meaningless — surface that
    # explicitly rather than letting Hypothesis run against the wrong target.
    from tests.invariants.test_pass_through_invariants import (
        _annotated_pass_through_plugins,
        test_annotated_transforms_preserve_input_fields,
    )

    plugins = _annotated_pass_through_plugins()
    assert plugins == [_DeliberatelyMisannotatedDropper], (
        f"monkeypatch failed — plugin manager did not return the fixture alone. Got {[p.__name__ for p in plugins]!r}."
    )

    # Drive the live forward-invariant function. ``@given(row=probe_row())``
    # supplies row values; ``_annotated_cls`` is a regular keyword arg passed
    # through Hypothesis to the inner function. Hypothesis raises the
    # underlying ``AssertionError`` after shrinking to a minimal probe.
    with pytest.raises(AssertionError) as exc_info:
        test_annotated_transforms_preserve_input_fields(
            _annotated_cls=_DeliberatelyMisannotatedDropper,
        )

    # The live assertion message names the offending plugin and the
    # dropped fields. We verify both — a regression that broke the
    # message format (without breaking the assertion itself) would still
    # affect operator triage and is worth catching here.
    message = str(exc_info.value)
    assert "deliberately-misannotated-dropper" in message.lower() or (_DeliberatelyMisannotatedDropper.__name__ in message), (
        f"Live assertion message must identify the offending plugin; got {message!r}"
    )
    assert "dropped" in message.lower(), f"Live assertion message must explain the failure mode ('dropped fields'); got {message!r}"


def test_harness_iterates_registered_declaration_contracts() -> None:
    """ADR-010 §Decision 3: the invariant harness is registry-driven."""
    from elspeth.contracts.declaration_contracts import registered_declaration_contracts
    from tests.invariants.test_pass_through_invariants import (
        _iter_contracts_for_invariant_harness,
    )

    registered = {c.name for c in registered_declaration_contracts()}
    harnessed = {c.name for c in _iter_contracts_for_invariant_harness()}
    assert harnessed == registered, f"Harness-to-registry drift: harnessed={harnessed}, registered={registered}"


def test_collection_guard_accepts_singleton_annotated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lone truthful pass-through transform must still be exercised.

    Regression guard for commit ``08b6d4e2``: once ``BatchReplicate`` stopped
    claiming unconditional pass-through, the live registry legitimately shrank
    to a singleton ``[PassThrough]``. Collection must still parametrize the
    forward invariant instead of crashing on a stale cardinality assumption.
    """
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.plugins.transforms.passthrough import PassThrough
    from tests.invariants.test_pass_through_invariants import pytest_generate_tests

    manager = get_shared_plugin_manager()
    monkeypatch.setattr(manager, "get_transforms", lambda: [PassThrough])

    metafunc = _CaptureMetafunc(["_annotated_cls"])
    pytest_generate_tests(metafunc)

    assert metafunc.argnames == "_annotated_cls"
    assert metafunc.argvalues == [PassThrough]
    assert metafunc.ids is not None
    assert metafunc.ids(PassThrough) == "PassThrough"


def test_backward_invariant_accepts_batch_replicate_mixed_validity_shape() -> None:
    """BatchReplicate's non-pass-through reason must be visible to the live harness.

    ``BatchReplicate`` is non-pass-through because mixed-validity batches can
    quarantine some inputs while still succeeding for the rest. The backward
    invariant must therefore be able to drive a representative batch shape
    rather than only scalar single-row probes, or the live governance harness
    will misclassify the plugin as preserving every input field.
    """
    from elspeth.plugins.transforms.batch_replicate import BatchReplicate
    from tests.invariants.test_pass_through_invariants import (
        test_non_pass_through_transforms_do_drop_fields,
    )

    test_non_pass_through_transforms_do_drop_fields(
        _non_pass_through_cls=BatchReplicate,
    )
