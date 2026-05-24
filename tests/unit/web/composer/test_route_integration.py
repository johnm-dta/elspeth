"""Tests for ComposerService route integration -- data transformations.

Tests the data conversions that the route handler performs:
- CompositionStateRecord -> CompositionState (via _state_from_record)
- CompositionState -> CompositionStateData (after compose())
- State version change detection
- ComposerConvergenceError attributes
- YAML endpoint calls generate_yaml on reconstructed state
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType
from uuid import UUID, uuid4

import pytest
from litellm.exceptions import APIError as LiteLLMAPIError
from litellm.exceptions import AuthenticationError as LiteLLMAuthError

from elspeth.web.composer.protocol import ComposerConvergenceError, ComposerResult
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.yaml_generator import generate_yaml
from elspeth.web.execution.schemas import ValidationError, ValidationReadiness, ValidationResult
from elspeth.web.sessions.converters import state_from_record as _state_from_record
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    CompositionStateRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_empty_state(version: int = 1) -> CompositionState:
    """Create an empty CompositionState at the given version."""
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=version,
    )


def _make_populated_state() -> CompositionState:
    """Create a CompositionState with source, node, edge, and output."""
    source = SourceSpec(
        plugin="csv",
        on_success="transform_1",
        options={"path": "/data/input.csv"},
        on_validation_failure="quarantine",
    )
    node = NodeSpec(
        id="transform_1",
        node_type="transform",
        plugin="passthrough",
        input="source_out",
        on_success="sink_out",
        on_error="discard",
        options={"field": "name"},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    edge = EdgeSpec(
        id="e1",
        from_node="source",
        to_node="transform_1",
        edge_type="on_success",
        label=None,
    )
    output = OutputSpec(
        name="main",
        plugin="csv",
        options={"path": "/data/output.csv"},
        on_write_failure="quarantine",
    )
    return CompositionState(
        source=source,
        nodes=(node,),
        edges=(edge,),
        outputs=(output,),
        metadata=PipelineMetadata(name="Test Pipeline", description="A test"),
        version=3,
    )


def _make_state_record(
    state: CompositionState,
    session_id: UUID | None = None,
    state_id: UUID | None = None,
) -> CompositionStateRecord:
    """Create a CompositionStateRecord from a CompositionState.

    Mirrors what the service layer produces when saving state.
    """
    d = state.to_dict()
    return CompositionStateRecord(
        id=state_id or uuid4(),
        session_id=session_id or uuid4(),
        version=state.version,
        sources=d["sources"],
        nodes=d["nodes"],
        edges=d["edges"],
        outputs=d["outputs"],
        metadata_=d["metadata"],
        is_valid=state.validate().is_valid,
        validation_errors=[e.message for e in state.validate().errors] if state.validate().errors else None,
        created_at=datetime.now(UTC),
        derived_from_state_id=None,
    )


# ---------------------------------------------------------------------------
# _state_from_record: CompositionStateRecord -> CompositionState
# ---------------------------------------------------------------------------


class TestStateFromRecord:
    """Tests for _state_from_record round-trip conversion."""

    def test_empty_state_round_trip(self) -> None:
        """Empty state survives record -> CompositionState conversion."""
        original = _make_empty_state()
        record = _make_state_record(original)
        reconstructed = _state_from_record(record)

        assert reconstructed.version == original.version
        assert reconstructed.sources == {}
        assert reconstructed.nodes == ()
        assert reconstructed.edges == ()
        assert reconstructed.outputs == ()
        assert reconstructed.metadata.name == "Untitled Pipeline"
        assert reconstructed.metadata.description == ""

    def test_populated_state_round_trip(self) -> None:
        """Populated state survives record -> CompositionState conversion."""
        original = _make_populated_state()
        record = _make_state_record(original)
        reconstructed = _state_from_record(record)

        assert reconstructed.version == original.version
        source = reconstructed.sources["source"]
        assert source.plugin == "csv"
        assert source.on_success == "transform_1"
        assert len(reconstructed.nodes) == 1
        assert reconstructed.nodes[0].id == "transform_1"
        assert len(reconstructed.edges) == 1
        assert reconstructed.edges[0].id == "e1"
        assert len(reconstructed.outputs) == 1
        assert reconstructed.outputs[0].name == "main"
        assert reconstructed.metadata.name == "Test Pipeline"

    def test_none_metadata_crashes(self) -> None:
        """Tier 1: None metadata_ on a record is database corruption — crash."""
        original = _make_empty_state()
        record = _make_state_record(original)
        record_with_none_meta = CompositionStateRecord(
            id=record.id,
            session_id=record.session_id,
            version=record.version,
            source=None,
            nodes=None,
            edges=None,
            outputs=None,
            metadata_=None,
            is_valid=False,
            validation_errors=None,
            created_at=record.created_at,
            derived_from_state_id=None,
        )
        with pytest.raises(ValueError, match="None metadata_"):
            _state_from_record(record_with_none_meta)

    def test_none_nodes_edges_outputs_map_to_empty(self) -> None:
        """None nodes/edges/outputs on a record map to empty tuples (initial state)."""
        original = _make_empty_state()
        record = _make_state_record(original)
        record_with_none_collections = CompositionStateRecord(
            id=record.id,
            session_id=record.session_id,
            version=record.version,
            source=None,
            nodes=None,
            edges=None,
            outputs=None,
            metadata_={"name": "Untitled Pipeline", "description": ""},
            is_valid=False,
            validation_errors=None,
            created_at=record.created_at,
            derived_from_state_id=None,
        )
        reconstructed = _state_from_record(record_with_none_collections)
        assert reconstructed.nodes == ()
        assert reconstructed.edges == ()
        assert reconstructed.outputs == ()

    def test_frozen_fields_are_thawed_for_reconstruction(self) -> None:
        """Record fields are frozen (MappingProxyType/tuple); reconstruction thaws them."""
        original = _make_populated_state()
        record = _make_state_record(original)

        # Verify the record fields are actually frozen
        assert isinstance(record.metadata_, MappingProxyType)

        # Reconstruction should work despite frozen fields
        reconstructed = _state_from_record(record)
        assert reconstructed.sources["source"].plugin == "csv"


# ---------------------------------------------------------------------------
# CompositionState -> CompositionStateData conversion
# ---------------------------------------------------------------------------


class TestStateToStateData:
    """Tests for CompositionState -> CompositionStateData conversion.

    This mirrors the logic in send_message() for persisting state changes.
    """

    def test_empty_state_to_state_data(self) -> None:
        """Empty state produces a valid CompositionStateData."""
        state = _make_empty_state()
        state_d = state.to_dict()
        validation = state.validate()
        data = CompositionStateData(
            sources=state_d["sources"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=validation.is_valid,
            validation_errors=[e.message for e in validation.errors] if validation.errors else None,
        )
        assert data.sources == {}
        assert not data.is_valid  # No source, no sinks
        assert data.validation_errors is not None

    def test_populated_state_to_state_data(self) -> None:
        """Populated state produces a valid CompositionStateData with correct fields."""
        state = _make_populated_state()
        state_d = state.to_dict()
        validation = state.validate()
        data = CompositionStateData(
            sources=state_d["sources"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=validation.is_valid,
            validation_errors=[e.message for e in validation.errors] if validation.errors else None,
        )
        assert data.sources is not None
        assert data.metadata_ is not None


# ---------------------------------------------------------------------------
# State version change detection
# ---------------------------------------------------------------------------


class TestVersionChangeDetection:
    """Tests for the state version comparison logic in send_message."""

    def test_same_version_skips_persistence(self) -> None:
        """When compose() returns the same version, no state save is needed."""
        initial = _make_empty_state(version=1)
        result = ComposerResult(message="Hello!", state=initial)
        # Version unchanged -> should not persist
        assert result.state.version == initial.version

    def test_incremented_version_triggers_persistence(self) -> None:
        """When compose() returns a higher version, state save is needed."""
        initial = _make_empty_state(version=1)
        updated = initial.with_metadata({"name": "My Pipeline"})
        result = ComposerResult(message="Updated name.", state=updated)
        assert result.state.version != initial.version
        assert result.state.version == 2

    def test_multiple_mutations_accumulate_version(self) -> None:
        """Multiple mutations in one compose() call increment version multiple times."""
        initial = _make_empty_state(version=1)
        s1 = initial.with_metadata({"name": "P1"})
        s2 = s1.with_output(
            OutputSpec(
                name="out",
                plugin="csv",
                options={},
                on_write_failure="quarantine",
            )
        )
        assert s2.version == 3
        assert s2.version != initial.version


# ---------------------------------------------------------------------------
# ComposerResult pairing invariant
# ---------------------------------------------------------------------------


class TestComposerResultPairingInvariant:
    """Enforce the docstring contract between ``runtime_preflight`` and
    ``raw_assistant_content`` mechanically.

    The pairing has two directions:

    1. preflight failed ⇒ raw_assistant_content MUST be set (so the
       original LLM text is recoverable from the audit row).
    2. raw_assistant_content set with passing preflight AND
       message == raw_assistant_content ⇒ rejected (spurious raw set;
       the audit row would falsely imply synthesis happened on a
       verbatim response).

    Note the narrowing on direction (2): synthesis with passing preflight
    IS legitimate when message ≠ raw_assistant_content (the state-claim
    grounding correction case from issue elspeth-c028f7d186 augments the
    happy-path message with an [ELSPETH-SYSTEM] correction suffix).
    The audit-integrity guarantee — no spurious raw_content set on a
    verbatim pass-through — survives the narrowing because the consumer
    discriminator at routes._composer_history_content uses
    ``message.startswith(raw)`` structurally and does not depend on
    *why* synthesis happened.
    """

    @staticmethod
    def _passing_validation() -> ValidationResult:
        return ValidationResult(
            is_valid=True,
            checks=[],
            errors=[],
            readiness=ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[]),
        )

    @staticmethod
    def _failing_validation() -> ValidationResult:
        return ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="boom",
                    suggestion=None,
                    error_code=None,
                ),
            ],
            readiness=ValidationReadiness(authoring_valid=False, execution_ready=False, completion_ready=False, blockers=[]),
        )

    def test_no_preflight_no_raw_content_is_valid(self) -> None:
        """No preflight ran → message is verbatim, raw_content is None."""
        ComposerResult(message="hi", state=_make_empty_state())

    def test_passed_preflight_no_raw_content_is_valid(self) -> None:
        """Preflight passed → message is verbatim, raw_content is None."""
        ComposerResult(
            message="hi",
            state=_make_empty_state(),
            runtime_preflight=self._passing_validation(),
        )

    def test_failed_preflight_with_raw_content_is_valid(self) -> None:
        """Preflight failed → message was replaced, raw_content holds original."""
        ComposerResult(
            message="<synthetic preflight-failure summary>",
            state=_make_empty_state(),
            runtime_preflight=self._failing_validation(),
            raw_assistant_content="The pipeline is complete.",
        )

    def test_spurious_raw_content_no_preflight_is_rejected(self) -> None:
        """raw_assistant_content set with no preflight AND message == raw —
        rejected (verbatim pass-through with raw set spuriously).

        The audit row would falsely imply synthesis happened when the
        message is actually verbatim LLM output.
        """
        with pytest.raises(ValueError, match=r"identical to raw_assistant_content|verbatim"):
            ComposerResult(
                message="hi",
                state=_make_empty_state(),
                runtime_preflight=None,
                raw_assistant_content="hi",
            )

    def test_spurious_raw_content_passed_preflight_is_rejected(self) -> None:
        """raw_assistant_content set with passing preflight AND
        message == raw — rejected (spurious raw set on verbatim
        pass-through).
        """
        with pytest.raises(ValueError, match=r"identical to raw_assistant_content|verbatim"):
            ComposerResult(
                message="hi",
                state=_make_empty_state(),
                runtime_preflight=self._passing_validation(),
                raw_assistant_content="hi",
            )

    def test_grounding_correction_synthesis_with_passed_preflight_is_valid(self) -> None:
        """raw_assistant_content set with passing preflight AND
        message ≠ raw — legitimate (state-claim grounding correction
        from issue elspeth-c028f7d186 augments happy-path prose with
        an [ELSPETH-SYSTEM] correction suffix).
        """
        ComposerResult(
            message="hi\n\n---\n\n[ELSPETH-SYSTEM] correction text",
            state=_make_empty_state(),
            runtime_preflight=self._passing_validation(),
            raw_assistant_content="hi",
        )

    def test_grounding_correction_synthesis_with_no_preflight_is_valid(self) -> None:
        """Same shape as above, with no preflight (the finalizer's
        skip-preflight branch when state.version is unchanged and no
        cached preflight is available — runtime_result is None and the
        grounding check still runs)."""
        ComposerResult(
            message="hi\n\n---\n\n[ELSPETH-SYSTEM] correction text",
            state=_make_empty_state(),
            runtime_preflight=None,
            raw_assistant_content="hi",
        )

    def test_failed_preflight_without_raw_content_is_rejected(self) -> None:
        """runtime_preflight failed but raw_assistant_content is None — forbidden.

        The message field's docstring states "when runtime preflight
        fails, [message] is replaced with a synthetic failure message;
        the original LLM text is preserved in ``raw_assistant_content``".
        A failed preflight without a parked raw_content means either the
        replacement never happened (the user sees a synthetic message
        but the real LLM output is gone) or the replacement happened but
        the original was discarded — either way, a contract violation.
        """
        with pytest.raises(ValueError, match=r"raw_assistant_content|message replacement"):
            ComposerResult(
                message="<synthetic>",
                state=_make_empty_state(),
                runtime_preflight=self._failing_validation(),
                raw_assistant_content=None,
            )


class TestComposerResultRepairTurnsCap:
    """Mechanical cap-assert on ``ComposerResult.repair_turns_used``.

    The compose loop in ``web/composer/service.py`` enforces the bound
    informally via ``_MAX_REPAIR_TURNS = 2`` (``if repair_turns_used >=
    _MAX_REPAIR_TURNS: return False`` before each ``+= 1``). The field
    flows into the audit trail via
    ``composition_states.composer_meta.repair_turns_used`` and is
    consumed by the convergence-suite eval scorer; an out-of-range value
    would land in the legal record and be silently accepted by
    downstream consumers.

    ``Literal[0, 1, 2]`` would have been the static-typed form, but the
    loop mutates a plain ``int`` counter and threads it via
    ``dataclasses.replace`` — mypy cannot narrow ``int`` to a Literal at
    that site. The runtime check in ``__post_init__`` is the fallback
    that mechanically rejects an out-of-range value at the construction
    boundary regardless of how callers obtained the integer. These tests
    pin the bound so future edits cannot relax it silently. Keep
    aligned with ``_MAX_REPAIR_TURNS`` in ``web/composer/service.py``.
    """

    def test_zero_is_valid(self) -> None:
        """First-pass success — no repair turns used."""
        result = ComposerResult(message="hi", state=_make_empty_state(), repair_turns_used=0)
        assert result.repair_turns_used == 0

    def test_one_is_valid(self) -> None:
        """One forced repair turn — within budget."""
        result = ComposerResult(message="hi", state=_make_empty_state(), repair_turns_used=1)
        assert result.repair_turns_used == 1

    def test_two_is_valid_at_cap(self) -> None:
        """Two forced repair turns — exactly at ``_MAX_REPAIR_TURNS``."""
        result = ComposerResult(message="hi", state=_make_empty_state(), repair_turns_used=2)
        assert result.repair_turns_used == 2

    def test_three_is_rejected_above_cap(self) -> None:
        """Three repair turns — above ``_MAX_REPAIR_TURNS``, must be rejected.

        If the compose loop ever forgets to clamp before a ``+= 1`` (or a
        future edit relaxes the loop's pre-increment guard), the audit
        row must not silently record an over-budget count.
        """
        with pytest.raises(ValueError, match=r"repair_turns_used must be 0, 1, or 2"):
            ComposerResult(message="hi", state=_make_empty_state(), repair_turns_used=3)

    def test_negative_is_rejected_below_zero(self) -> None:
        """Negative repair turns — nonsensical, must be rejected.

        The counter starts at 0 and only increments; a negative value
        could only arise from a programming bug threading the wrong
        integer through ``dataclasses.replace``. Crash informatively
        rather than land it in the audit trail.
        """
        with pytest.raises(ValueError, match=r"repair_turns_used must be 0, 1, or 2"):
            ComposerResult(message="hi", state=_make_empty_state(), repair_turns_used=-1)


# ---------------------------------------------------------------------------
# ComposerConvergenceError
# ---------------------------------------------------------------------------


class TestComposerConvergenceError:
    """Tests for ComposerConvergenceError attributes."""

    def test_max_turns_attribute(self) -> None:
        """ComposerConvergenceError carries max_turns for the HTTP response."""
        exc = ComposerConvergenceError(max_turns=5)
        assert exc.max_turns == 5
        assert "5 turns" in str(exc)

    def test_is_exception(self) -> None:
        """ComposerConvergenceError is catchable as Exception."""
        exc = ComposerConvergenceError(max_turns=10)
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# YAML endpoint: generate_yaml on reconstructed state
# ---------------------------------------------------------------------------


class TestYamlGeneration:
    """Tests that generate_yaml works on states reconstructed from records."""

    def test_yaml_from_empty_state(self) -> None:
        """generate_yaml on an empty state produces valid YAML (empty doc)."""
        state = _make_empty_state()
        yaml_str = generate_yaml(state)
        assert isinstance(yaml_str, str)
        # Empty state with no source/sinks -> empty doc
        assert yaml_str.strip() == "{}"

    def test_yaml_from_populated_state_round_trip(self) -> None:
        """generate_yaml on a state reconstructed from a record matches direct generation."""
        original = _make_populated_state()
        direct_yaml = generate_yaml(original)

        record = _make_state_record(original)
        reconstructed = _state_from_record(record)
        reconstructed_yaml = generate_yaml(reconstructed)

        assert direct_yaml == reconstructed_yaml

    def test_yaml_contains_source_plugin(self) -> None:
        """Generated YAML includes the source plugin name."""
        state = _make_populated_state()
        yaml_str = generate_yaml(state)
        assert "csv" in yaml_str
        assert "source:" in yaml_str

    def test_single_default_source_uses_sources_yaml_shape(self) -> None:
        source = SourceSpec(
            plugin="csv",
            on_success="main",
            options={"path": "input.csv"},
            on_validation_failure="quarantine",
        )
        state = CompositionState(
            source=None,
            sources={"source": source},
            nodes=(),
            edges=(),
            outputs=(OutputSpec(name="main", plugin="json", options={"path": "out.json"}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )

        yaml_str = generate_yaml(state)

        assert "sources:" in yaml_str
        assert "source:" in yaml_str

    def test_single_named_source_uses_sources_yaml_shape(self) -> None:
        source = SourceSpec(
            plugin="csv",
            on_success="main",
            options={"path": "orders.csv"},
            on_validation_failure="quarantine",
        )
        state = CompositionState(
            source=None,
            sources={"orders": source},
            nodes=(),
            edges=(),
            outputs=(OutputSpec(name="main", plugin="json", options={"path": "out.json"}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )

        yaml_str = generate_yaml(state)

        assert "sources:" in yaml_str
        assert "orders:" in yaml_str
        assert "\nsource:" not in yaml_str

    def test_yaml_contains_sink(self) -> None:
        """Generated YAML includes the sink name."""
        state = _make_populated_state()
        yaml_str = generate_yaml(state)
        assert "main:" in yaml_str
        assert "sinks:" in yaml_str


# ---------------------------------------------------------------------------
# _is_llm_client_error
# ---------------------------------------------------------------------------


class TestLlmErrorHandling:
    """Tests for LLM error → HTTP status mapping in send_message route."""

    def test_convergence_error_has_max_turns(self) -> None:
        """ComposerConvergenceError carries max_turns for HTTP 422 body."""
        exc = ComposerConvergenceError(max_turns=20)
        assert exc.max_turns == 20

    def test_auth_error_type_available(self) -> None:
        """litellm.exceptions.AuthenticationError is importable for HTTP 502 auth error path."""
        assert LiteLLMAuthError is not None

    def test_api_error_type_available(self) -> None:
        """litellm.exceptions.APIError is importable for HTTP 502 unavailable path."""
        assert LiteLLMAPIError is not None
