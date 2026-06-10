"""Tests for failure-sample enrichment of run-level error messages."""

from __future__ import annotations

from datetime import UTC, datetime

from elspeth.contracts import NodeType
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import tokens_table
from elspeth.web.execution.failure_samples import (
    FailureSample,
    _classify,
    format_failure_samples,
    load_top_failure_samples,
)

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _make_run_with_transform(transform_id: str = "fetch") -> tuple[LandscapeDB, str, str]:
    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="test_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
        node_id="source_test",
        sequence=0,
    )
    factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="web_scrape",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
        node_id=transform_id,
        sequence=1,
    )
    return db, run.run_id, transform_id


def _record_error(
    db: LandscapeDB,
    run_id: str,
    transform_id: str,
    *,
    error: str,
    error_type: str,
    token_id: str,
    row_index: int,
) -> None:
    factory = RecorderFactory(db)
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id="source_test",
        row_index=row_index,
        data={"url": f"row-{row_index}"},
        source_row_index=row_index,
        ingest_sequence=row_index,
    )
    with db.connection() as conn:
        conn.execute(
            tokens_table.insert().values(
                token_id=token_id,
                row_id=row.row_id,
                run_id=run_id,
                step_in_pipeline=0,
                created_at=datetime.now(UTC),
            )
        )
        conn.commit()
    factory.data_flow.record_transform_error(
        ref=TokenRef(token_id=token_id, run_id=run_id),
        transform_id=transform_id,
        row_data={"url": f"row-{row_index}"},
        error_details={"reason": "validation_failed", "error": error, "error_type": error_type},
        destination="discard",
    )


def _record_messageless_error(
    db: LandscapeDB,
    run_id: str,
    transform_id: str,
    *,
    reason: str,
    token_id: str,
    row_index: int,
) -> None:
    """Record an error whose details carry only the required ``reason`` category.

    Exercises the canonical-shape branch where no human-message field
    (``error``/``message``) is present — the honest fallback must keep distinct
    categories in distinct aggregation buckets, not collapse them to ``""``.
    """
    factory = RecorderFactory(db)
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id="source_test",
        row_index=row_index,
        data={"url": f"row-{row_index}"},
        source_row_index=row_index,
        ingest_sequence=row_index,
    )
    with db.connection() as conn:
        conn.execute(
            tokens_table.insert().values(
                token_id=token_id,
                row_id=row.row_id,
                run_id=run_id,
                step_in_pipeline=0,
                created_at=datetime.now(UTC),
            )
        )
        conn.commit()
    factory.data_flow.record_transform_error(
        ref=TokenRef(token_id=token_id, run_id=run_id),
        transform_id=transform_id,
        row_data={"url": f"row-{row_index}"},
        error_details={"reason": reason},
        destination="discard",
    )


class TestLoadTopFailureSamples:
    def test_aggregates_identical_errors_by_count(self) -> None:
        db, run_id, transform_id = _make_run_with_transform()
        for i in range(3):
            _record_error(
                db,
                run_id,
                transform_id,
                error="URL is missing a scheme",
                error_type="SSRFBlockedError",
                token_id=f"tok_{i}",
                row_index=i,
            )

        samples = load_top_failure_samples(db, run_id)

        assert len(samples) == 1
        assert samples[0] == FailureSample(
            transform_id=transform_id,
            error_type="SSRFBlockedError",
            message="URL is missing a scheme",
            count=3,
        )

    def test_orders_by_descending_count(self) -> None:
        db, run_id, transform_id = _make_run_with_transform()
        # 1x rare, 2x medium, 4x dominant
        _record_error(db, run_id, transform_id, error="rare", error_type="A", token_id="t0", row_index=0)
        for i in range(2):
            _record_error(db, run_id, transform_id, error="medium", error_type="B", token_id=f"tm{i}", row_index=10 + i)
        for i in range(4):
            _record_error(db, run_id, transform_id, error="dominant", error_type="C", token_id=f"td{i}", row_index=20 + i)

        samples = load_top_failure_samples(db, run_id, limit=3)

        assert [s.message for s in samples] == ["dominant", "medium", "rare"]
        assert [s.count for s in samples] == [4, 2, 1]

    def test_returns_empty_when_no_errors_recorded(self) -> None:
        db, run_id, _ = _make_run_with_transform()
        assert load_top_failure_samples(db, run_id) == []

    def test_limit_truncates_to_top_n(self) -> None:
        db, run_id, transform_id = _make_run_with_transform()
        for i in range(5):
            _record_error(db, run_id, transform_id, error=f"err-{i}", error_type="E", token_id=f"t{i}", row_index=i)

        samples = load_top_failure_samples(db, run_id, limit=2)
        assert len(samples) == 2

    def test_rejects_invalid_limit(self) -> None:
        db, run_id, _ = _make_run_with_transform()
        try:
            load_top_failure_samples(db, run_id, limit=0)
        except ValueError:
            return
        raise AssertionError("expected ValueError for limit=0")

    def test_messageless_errors_with_distinct_reasons_do_not_merge(self) -> None:
        # Two errors that carry only the required ``reason`` category and no
        # human-message field. Before the Tier-1 honest-read fix the terminal
        # ``""`` default collapsed both into a single ``message=""`` bucket; the
        # distinct categories must now stay distinct so the operator sees that
        # two unrelated failure modes occurred, not one.
        db, run_id, transform_id = _make_run_with_transform()
        _record_messageless_error(db, run_id, transform_id, reason="missing_field", token_id="tm0", row_index=0)
        _record_messageless_error(db, run_id, transform_id, reason="api_error", token_id="ta0", row_index=1)

        samples = load_top_failure_samples(db, run_id, limit=3)

        assert len(samples) == 2
        messages = {s.message for s in samples}
        assert messages == {
            "(no message; category=missing_field)",
            "(no message; category=api_error)",
        }
        # error_type falls back to the required reason category, not "UnknownError".
        assert {s.error_type for s in samples} == {"missing_field", "api_error"}


class TestClassify:
    """Direct coverage of the two-shape discriminator.

    The non-canonical envelope is produced by
    ``DataFlowRepository._canonical_or_recorded_error_details_json`` only when
    canonical serialization of ``error_details`` fails, which cannot be driven
    through the typed recorder API — so it is asserted here against the exact
    envelope shape that writer emits.
    """

    def test_canonical_prefers_error_field(self) -> None:
        details = {"reason": "api_error", "error": "boom", "error_type": "http_error"}
        assert _classify(details) == ("http_error", "boom")

    def test_canonical_falls_back_to_message_field(self) -> None:
        details = {"reason": "api_error", "message": "rate limited"}
        # error_type absent -> falls back to the required reason category.
        assert _classify(details) == ("api_error", "rate limited")

    def test_canonical_messageless_carries_category_distinctly(self) -> None:
        assert _classify({"reason": "missing_field"}) == (
            "missing_field",
            "(no message; category=missing_field)",
        )

    def test_missing_required_reason_raises(self) -> None:
        # Canonical shape with no reason is Tier-1 corruption: KeyError surfaces.
        try:
            _classify({"error_type": "http_error", "error": "boom"})
        except KeyError:
            return
        raise AssertionError("expected KeyError on missing required 'reason'")

    def test_non_canonical_envelope_surfaces_serialization_error(self) -> None:
        # Exact shape emitted by _canonical_or_recorded_error_details_json.
        details = {
            "__non_canonical__": True,
            "repr": "{'reason': 'x', 'val': nan}",
            "serialization_error": "Out of range float values are not JSON compliant",
        }
        error_type, message = _classify(details)
        assert error_type == "NonCanonicalErrorDetails"
        assert "Out of range float values" in message
        assert "repr=" in message

    def test_distinct_non_canonical_envelopes_do_not_merge(self) -> None:
        first = _classify(
            {
                "__non_canonical__": True,
                "repr": "{'a': nan}",
                "serialization_error": "nan error",
            }
        )
        second = _classify(
            {
                "__non_canonical__": True,
                "repr": "{'b': inf}",
                "serialization_error": "inf error",
            }
        )
        assert first != second


class TestFormatFailureSamples:
    def test_empty_samples_yields_empty_string(self) -> None:
        assert format_failure_samples([]) == ""

    def test_single_transform_omits_transform_prefix(self) -> None:
        samples = [
            FailureSample(transform_id="fetch", error_type="SSRFBlockedError", message="missing scheme", count=3),
        ]
        rendered = format_failure_samples(samples)
        assert "[fetch]" not in rendered
        assert "3x SSRFBlockedError: missing scheme" in rendered

    def test_multi_transform_includes_transform_prefix(self) -> None:
        samples = [
            FailureSample(transform_id="fetch", error_type="SSRFBlockedError", message="missing scheme", count=2),
            FailureSample(transform_id="summarise", error_type="LLMError", message="rate limited", count=1),
        ]
        rendered = format_failure_samples(samples)
        assert "[fetch]" in rendered
        assert "[summarise]" in rendered

    def test_truncates_long_messages_with_ellipsis(self) -> None:
        long = "x" * 1000
        samples = [FailureSample(transform_id="fetch", error_type="E", message=long, count=1)]
        rendered = format_failure_samples(samples, message_chars=50)
        assert "…" in rendered
        # Each rendered line is one bullet; cap is on the message portion only,
        # so the line as a whole stays small.
        assert len(rendered) < 200
