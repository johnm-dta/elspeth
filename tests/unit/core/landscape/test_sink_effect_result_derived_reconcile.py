"""Closed carrier proofs for result-derived sink reconciliation."""

from __future__ import annotations

import pytest

from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import SinkEffectAttemptAction, SinkEffectReconcileResult
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.execution.sink_effect_attempt_results import (
    decode_sink_effect_returned_result,
    encode_sink_effect_returned_result,
)

_DESCRIPTOR = ArtifactDescriptor(
    artifact_type="database",
    path_or_uri="database-result:sha256:" + "a" * 64,
    content_hash="b" * 64,
    size_bytes=2,
    metadata={"table": "output", "row_count": 0},
)


def test_reconcile_carrier_round_trips_exact_result_partition() -> None:
    result = SinkEffectReconcileResult.applied(
        _DESCRIPTOR,
        evidence={"marker": "exact"},
        accepted_ordinals=(0, 2),
        diverted_ordinals=(1,),
    )

    encoded = encode_sink_effect_returned_result(result)
    decoded = decode_sink_effect_returned_result(SinkEffectAttemptAction.RECONCILE, canonical_json(encoded))

    assert decoded == result


def test_reconcile_carrier_rejects_partial_or_overlapping_partition() -> None:
    with pytest.raises(ValueError, match="both be present"):
        SinkEffectReconcileResult.applied(
            _DESCRIPTOR,
            evidence={},
            accepted_ordinals=(0,),
        )
    with pytest.raises(ValueError, match="must not overlap"):
        SinkEffectReconcileResult.applied(
            _DESCRIPTOR,
            evidence={},
            accepted_ordinals=(0,),
            diverted_ordinals=(0,),
        )
