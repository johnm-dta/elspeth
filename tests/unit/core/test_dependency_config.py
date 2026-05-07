"""Tests for dependency, commencement gate, and collection probe config models."""

from __future__ import annotations

from types import MappingProxyType

import pytest
from pydantic import ValidationError

from elspeth.core.dependency_config import (
    CollectionProbeConfig,
    CommencementGateConfig,
    CommencementGateResult,
    DependencyConfig,
    DependencyRunResult,
    PreflightResult,
)


class TestDependencyConfig:
    def test_valid_config(self) -> None:
        config = DependencyConfig(name="index_corpus", settings="./index.yaml")
        assert config.name == "index_corpus"
        assert config.settings == "./index.yaml"

    def test_frozen(self) -> None:
        config = DependencyConfig(name="x", settings="./x.yaml")
        with pytest.raises(ValidationError):
            config.name = "y"  # type: ignore[misc]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            DependencyConfig(name="x", settings="./x.yaml", extra="bad")  # type: ignore[call-arg]

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            DependencyConfig(name="", settings="./x.yaml")

    def test_rejects_empty_settings(self) -> None:
        with pytest.raises(ValidationError):
            DependencyConfig(name="x", settings="")


class TestCommencementGateConfig:
    def test_valid_config(self) -> None:
        config = CommencementGateConfig(
            name="corpus_ready",
            condition="collections['test']['count'] > 0",
        )
        assert config.name == "corpus_ready"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            CommencementGateConfig(name="", condition="True")

    def test_rejects_empty_condition(self) -> None:
        with pytest.raises(ValidationError):
            CommencementGateConfig(name="x", condition="")


class TestCollectionProbeConfig:
    def test_valid_config(self) -> None:
        config = CollectionProbeConfig(
            collection="science-facts",
            provider="chroma",
            provider_config={
                "mode": "persistent",
                "persist_directory": "./chroma_data",
            },
        )
        assert config.collection == "science-facts"
        assert config.provider == "chroma"

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            CollectionProbeConfig(
                collection="x",
                provider="chroma",
                provider_config={},
                extra="bad",  # type: ignore[call-arg]
            )

    def test_provider_config_is_frozen(self) -> None:
        """provider_config must be deep-frozen (review finding #7)."""
        config = CollectionProbeConfig(
            collection="test",
            provider="chroma",
            provider_config={"mode": "persistent", "nested": {"key": "val"}},
        )
        with pytest.raises(TypeError):
            config.provider_config["new_key"] = "bad"  # type: ignore[index]

    def test_empty_provider_config_is_frozen(self) -> None:
        """Empty dict must also be frozen (review finding #3 from fix review)."""
        config = CollectionProbeConfig(
            collection="test",
            provider="chroma",
        )
        with pytest.raises(TypeError):
            config.provider_config["new_key"] = "bad"  # type: ignore[index]

    def test_model_dump_serializes_frozen_provider_config(self) -> None:
        """model_dump() must produce plain dicts from deep-frozen MappingProxyType."""
        config = CollectionProbeConfig(
            collection="science-facts",
            provider="chroma",
            provider_config={"mode": "persistent", "nested": {"key": "val"}},
        )
        dumped = config.model_dump()
        assert isinstance(dumped["provider_config"], dict)
        assert isinstance(dumped["provider_config"]["nested"], dict)
        assert dumped["provider_config"]["mode"] == "persistent"

    def test_model_dump_json_succeeds(self) -> None:
        """model_dump_json() must not raise on deep-frozen provider_config."""
        config = CollectionProbeConfig(
            collection="test",
            provider="chroma",
            provider_config={"mode": "persistent", "persist_directory": "./data"},
        )
        json_str = config.model_dump_json()
        assert "persistent" in json_str

    def test_model_dump_round_trip(self) -> None:
        """model_dump() output must reconstruct an equivalent instance."""
        original = CollectionProbeConfig(
            collection="test",
            provider="chroma",
            provider_config={"mode": "persistent", "nested": {"a": 1}},
        )
        reconstructed = CollectionProbeConfig(**original.model_dump())
        assert reconstructed.collection == original.collection
        assert reconstructed.provider == original.provider
        assert dict(reconstructed.provider_config) == dict(original.provider_config)


class TestDependencyRunResult:
    def test_construction(self) -> None:
        result = DependencyRunResult(
            name="index_corpus",
            run_id="abc-123",
            settings_hash="sha256:deadbeef",
            duration_ms=4520,
            indexed_at="2026-03-25T14:02:33Z",
        )
        assert result.name == "index_corpus"
        assert result.run_id == "abc-123"
        assert result.indexed_at == "2026-03-25T14:02:33Z"

    def test_frozen(self) -> None:
        result = DependencyRunResult(name="x", run_id="y", settings_hash="z", duration_ms=0, indexed_at="t")
        with pytest.raises(AttributeError):
            result.name = "other"  # type: ignore[misc]

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            DependencyRunResult(name="", run_id="y", settings_hash="z", duration_ms=0, indexed_at="t")

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id must not be empty"):
            DependencyRunResult(name="x", run_id="", settings_hash="z", duration_ms=0, indexed_at="t")

    def test_empty_settings_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="settings_hash must not be empty"):
            DependencyRunResult(name="x", run_id="y", settings_hash="", duration_ms=0, indexed_at="t")

    def test_negative_duration_ms_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_ms must be >= 0"):
            DependencyRunResult(name="x", run_id="y", settings_hash="z", duration_ms=-1, indexed_at="t")

    def test_bool_duration_ms_rejected(self) -> None:
        """bool is a subclass of int — require_int rejects it."""
        with pytest.raises(TypeError, match="duration_ms must be int"):
            DependencyRunResult(name="x", run_id="y", settings_hash="z", duration_ms=True, indexed_at="t")


class TestCommencementGateResult:
    def test_construction(self) -> None:
        snapshot = {"collections": {"test": {"count": 10, "reachable": True}}}
        result = CommencementGateResult(
            name="corpus_ready",
            condition="collections['test']['count'] > 0",
            result=True,
            context_snapshot=snapshot,
        )
        assert result.name == "corpus_ready"
        assert result.result is True

    def test_context_snapshot_is_deep_frozen(self) -> None:
        snapshot = {"collections": {"test": {"count": 10}}}
        result = CommencementGateResult(name="x", condition="True", result=True, context_snapshot=snapshot)
        assert isinstance(result.context_snapshot, MappingProxyType)
        # Nested dict should also be frozen
        assert isinstance(result.context_snapshot["collections"], MappingProxyType)

    def test_frozen(self) -> None:
        result = CommencementGateResult(name="x", condition="True", result=True, context_snapshot={})
        with pytest.raises(AttributeError):
            result.name = "other"  # type: ignore[misc]

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            CommencementGateResult(name="", condition="True", result=True, context_snapshot={})

    def test_empty_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="condition must not be empty"):
            CommencementGateResult(name="x", condition="", result=True, context_snapshot={})


class TestPreflightResult:
    def test_construction_with_empty_tuples(self) -> None:
        result = PreflightResult(dependency_runs=(), gate_results=())
        assert result.dependency_runs == ()
        assert result.gate_results == ()

    def test_list_coerced_to_tuple(self) -> None:
        """freeze_fields converts lists to tuples for immutability."""
        result = PreflightResult(
            dependency_runs=[],  # type: ignore[arg-type]
            gate_results=[],  # type: ignore[arg-type]
        )
        assert isinstance(result.dependency_runs, tuple)
        assert isinstance(result.gate_results, tuple)

    def test_frozen(self) -> None:
        result = PreflightResult(dependency_runs=(), gate_results=())
        with pytest.raises(AttributeError):
            result.dependency_runs = ()  # type: ignore[misc]
