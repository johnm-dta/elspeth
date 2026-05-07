"""Regression tests for the runtime-VAL manifest builder."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import elspeth.engine.executors.declared_output_fields as declared_output_fields_module
from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    DispatchSite,
    ExampleBundle,
    PostEmissionInputs,
    PostEmissionOutputs,
    implements_dispatch_site,
)
from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.engine.executors.pass_through import PassThroughDeclarationContract
from elspeth.engine.orchestrator import prepare_for_run


@pytest.fixture()
def _isolate_runtime_val_registries() -> Iterator[None]:
    """Restore both registries after each test mutates freeze or membership state."""
    import elspeth.contracts.declaration_contracts as dc
    import elspeth.contracts.tier_registry as tr

    saved_dc_registry = list(dc._REGISTRY)
    saved_dc_per_site = {site: list(lst) for site, lst in dc._REGISTRY_BY_SITE.items()}
    saved_dc_frozen = dc._FROZEN

    saved_tr_registry = list(tr._REGISTRY)
    saved_tr_reasons = dict(tr._REASONS)
    saved_tr_frozen = tr._FROZEN

    yield

    dc._REGISTRY.clear()
    dc._REGISTRY.extend(saved_dc_registry)
    for site, lst in saved_dc_per_site.items():
        dc._REGISTRY_BY_SITE[site][:] = lst
    dc._FROZEN = saved_dc_frozen

    tr._REGISTRY.clear()
    tr._REGISTRY.extend(saved_tr_registry)
    tr._REASONS.clear()
    tr._REASONS.update(saved_tr_reasons)
    tr._FROZEN = saved_tr_frozen


def test_build_runtime_val_manifest_requires_frozen_registries(_isolate_runtime_val_registries: None) -> None:
    import elspeth.contracts.declaration_contracts as dc
    import elspeth.contracts.tier_registry as tr

    dc._FROZEN = False
    tr._FROZEN = False

    with pytest.raises(FrameworkBugError, match="frozen"):
        build_runtime_val_manifest()


def test_manifest_records_declaration_contract_implementation_hash(
    _isolate_runtime_val_registries: None,
) -> None:
    import elspeth.contracts.declaration_contracts as dc
    import elspeth.contracts.tier_registry as tr

    dc._FROZEN = False
    tr._FROZEN = False
    prepare_for_run()

    baseline = build_runtime_val_manifest()
    baseline_entry = next(entry for entry in baseline["declaration_contracts"] if entry["name"] == "passes_through_input")

    original_code = PassThroughDeclarationContract.post_emission_check.__code__

    def replacement(self, inputs, outputs):  # type: ignore[no-untyped-def]
        return None

    PassThroughDeclarationContract.post_emission_check.__code__ = replacement.__code__
    try:
        mutated = build_runtime_val_manifest()
    finally:
        PassThroughDeclarationContract.post_emission_check.__code__ = original_code

    mutated_entry = next(entry for entry in mutated["declaration_contracts"] if entry["name"] == "passes_through_input")

    assert baseline_entry["name"] == mutated_entry["name"]
    assert baseline_entry["class_name"] == mutated_entry["class_name"]
    assert baseline_entry["class_module"] == mutated_entry["class_module"]
    assert baseline_entry["dispatch_sites"] == mutated_entry["dispatch_sites"]
    assert baseline_entry["implementation_hash"] != mutated_entry["implementation_hash"]


def test_manifest_records_delegated_declaration_helper_implementation_hash(
    _isolate_runtime_val_registries: None,
) -> None:
    import elspeth.contracts.declaration_contracts as dc
    import elspeth.contracts.tier_registry as tr

    dc._FROZEN = False
    tr._FROZEN = False
    prepare_for_run()

    baseline = build_runtime_val_manifest()
    baseline_entry = next(entry for entry in baseline["declaration_contracts"] if entry["name"] == "declared_output_fields")

    original_code = declared_output_fields_module.verify_declared_output_fields.__code__

    def replacement(
        *,
        declared_output_fields: frozenset[str],
        emitted_rows: object,
        plugin_name: str,
        node_id: str,
        run_id: str,
        row_id: str,
        token_id: str,
    ) -> None:
        del declared_output_fields, emitted_rows, plugin_name, node_id, run_id, row_id, token_id
        return None

    declared_output_fields_module.verify_declared_output_fields.__code__ = replacement.__code__
    try:
        mutated = build_runtime_val_manifest()
    finally:
        declared_output_fields_module.verify_declared_output_fields.__code__ = original_code

    mutated_entry = next(entry for entry in mutated["declaration_contracts"] if entry["name"] == "declared_output_fields")

    assert baseline_entry["name"] == mutated_entry["name"]
    assert baseline_entry["class_name"] == mutated_entry["class_name"]
    assert baseline_entry["class_module"] == mutated_entry["class_module"]
    assert baseline_entry["dispatch_sites"] == mutated_entry["dispatch_sites"]
    assert baseline_entry["implementation_hash"] != mutated_entry["implementation_hash"]


def test_manifest_rejects_source_unavailable_classes(
    monkeypatch: pytest.MonkeyPatch,
    _isolate_runtime_val_registries: None,
) -> None:
    import elspeth.contracts.declaration_contracts as dc
    import elspeth.contracts.runtime_val_manifest as manifest_module
    import elspeth.contracts.tier_registry as tr

    dc._FROZEN = False
    tr._FROZEN = False
    prepare_for_run()

    def source_unavailable(cls: type[object]) -> str:
        raise OSError(f"source unavailable for {cls.__module__}.{cls.__qualname__}")

    monkeypatch.setattr(manifest_module.inspect, "getsource", source_unavailable)

    with pytest.raises(FrameworkBugError, match="source unavailable"):
        build_runtime_val_manifest()


def test_manifest_records_tier_1_implementation_hash(_isolate_runtime_val_registries: None) -> None:
    import elspeth.contracts.declaration_contracts as dc
    import elspeth.contracts.tier_registry as tr
    from elspeth.contracts.tier_registry import tier_1_error

    dc._FROZEN = False
    tr._FROZEN = False

    @tier_1_error(reason="test runtime-val manifest implementation drift", caller_module=__name__)
    class _TempTier1Error(Exception):
        def describe(self) -> str:
            return "before"

    prepare_for_run()

    baseline = build_runtime_val_manifest()
    baseline_entry = next(
        entry for entry in baseline["tier_1_errors"] if entry["class_name"] == "_TempTier1Error" and entry["class_module"] == __name__
    )

    original_code = _TempTier1Error.describe.__code__

    def replacement(self):  # type: ignore[no-untyped-def]
        return "after"

    _TempTier1Error.describe.__code__ = replacement.__code__
    try:
        mutated = build_runtime_val_manifest()
    finally:
        _TempTier1Error.describe.__code__ = original_code

    mutated_entry = next(
        entry for entry in mutated["tier_1_errors"] if entry["class_name"] == "_TempTier1Error" and entry["class_module"] == __name__
    )

    assert baseline_entry["class_name"] == mutated_entry["class_name"]
    assert baseline_entry["class_module"] == mutated_entry["class_module"]
    assert baseline_entry["reason"] == mutated_entry["reason"]
    assert baseline_entry["implementation_hash"] != mutated_entry["implementation_hash"]


def test_manifest_crashes_if_registered_contract_lacks_payload_schema(
    _isolate_runtime_val_registries: None,
) -> None:
    """A corrupted registry entry must not hash ``None`` as its payload schema."""
    import elspeth.contracts.declaration_contracts as dc
    import elspeth.contracts.tier_registry as tr

    class _NoPayloadSchemaContract(DeclarationContract):
        name = "no_payload_schema"

        def applies_to(self, plugin: object) -> bool:
            return False

        @implements_dispatch_site("post_emission_check")
        def post_emission_check(
            self,
            inputs: PostEmissionInputs,
            outputs: PostEmissionOutputs,
        ) -> None:
            raise AssertionError("not reached")

        @classmethod
        def negative_example(cls) -> ExampleBundle:
            raise NotImplementedError

        @classmethod
        def positive_example_does_not_apply(cls) -> ExampleBundle:
            raise NotImplementedError

    contract = _NoPayloadSchemaContract()
    dc._REGISTRY.append(contract)
    dc._REGISTRY_BY_SITE[DispatchSite.POST_EMISSION].append(contract)
    dc._FROZEN = True
    tr._FROZEN = True

    with pytest.raises(AttributeError, match="payload_schema"):
        build_runtime_val_manifest()
