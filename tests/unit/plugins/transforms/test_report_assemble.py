"""Tests for ReportAssemble aggregation transform.

Task 2 of the report_assemble plan (TDD red step).  The transform under
test does not yet exist; these tests must fail with ``ModuleNotFoundError``
until Task 3 lands the implementation.

Helper conventions mirror ``test_batch_top_k.py``:

* ``_row(data)`` builds a ``PipelineRow`` carrying an ``OBSERVED``
  ``SchemaContract`` so the row can be passed through a batch-aware
  transform without a pre-declared schema.
* ``_ctx(...)`` attaches a fully-validated :class:`AggregationBatchContext`
  to a real :class:`PluginContext` via ``dataclasses.replace`` (the
  context is frozen).  The batch metadata is what
  ``AggregationExecutor`` would set immediately before invoking the
  batch-aware transform.
* Imports of ``ReportAssemble`` are deferred *inside* each test so that
  pytest collection succeeds and the ``ModuleNotFoundError`` surfaces as
  a per-test failure (RED state) rather than a collection error.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from elspeth.contracts.node_state_context import AggregationBatchContext
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.testing import make_field, make_row
from tests.fixtures.factories import make_context

DYNAMIC_SCHEMA = {"mode": "observed"}


def _row(data: dict[str, Any]):
    """Create a PipelineRow with OBSERVED contract for testing."""
    fields = tuple(
        make_field(
            key,
            type(value) if value is None or type(value) in (str, int, float, bool) else object,
            original_name=key,
            required=False,
            source="inferred",
        )
        for key, value in data.items()
    )
    contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
    return make_row(data, contract=contract)


def _ctx(
    *,
    batch_size: int = 2,
    flush_index: int = 1,
    rows_seen_total: int | None = None,
    row_start: int = 1,
    row_end: int | None = None,
    trigger_type: str = "count",
    batch_id: str = "batch-1",
    is_end_of_source: bool = False,
) -> PluginContext:
    """Build a PluginContext with an AggregationBatchContext attached.

    Defaults are chosen so callers can opt into only the dimensions the
    test cares about.  When ``row_end`` is not supplied, it is derived
    from ``row_start + batch_size - 1`` so the resulting context
    satisfies ``AggregationBatchContext``'s ``row_end >= row_start`` and
    ``rows_seen_total >= row_end`` invariants without each test having to
    spell the arithmetic out.
    """
    if row_end is None:
        row_end = row_start + batch_size - 1
    if rows_seen_total is None:
        rows_seen_total = row_end
    batch = AggregationBatchContext(
        trigger_type=trigger_type,
        batch_id=batch_id,
        batch_size=batch_size,
        flush_index=flush_index,
        rows_seen_total=rows_seen_total,
        row_start=row_start,
        row_end=row_end,
        is_end_of_source=is_end_of_source,
    )
    base = make_context()
    return dataclasses.replace(base, aggregation_batch=batch)


class TestReportAssembleRendering:
    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        assert ReportAssemble.name == "report_assemble"
        assert ReportAssemble.is_batch_aware is True

    def test_collates_lines_with_default_plain_text(self) -> None:
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble({"schema": DYNAMIC_SCHEMA, "text_field": "line"})
        ctx = _ctx(batch_size=2)

        result = transform.process([_row({"line": "alpha"}), _row({"line": "beta"})], ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["report_body"] == "alpha\nbeta"
        assert result.row["report_format"] == "plain_text"

    def test_pagination_columns_reflect_aggregation_batch_context(self) -> None:
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble({"schema": DYNAMIC_SCHEMA, "text_field": "line"})
        # Page 2 of an 80-row count-triggered flush: rows 81..160, no EOS.
        ctx = _ctx(
            batch_size=80,
            flush_index=2,
            rows_seen_total=160,
            row_start=81,
            row_end=160,
            trigger_type="count",
            is_end_of_source=False,
        )
        rows = [_row({"line": f"line-{i}"}) for i in range(80)]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["report_index"] == 2
        assert result.row["line_start"] == 81
        assert result.row["line_end"] == 160
        assert result.row["line_count"] == 80
        assert result.row["lines_seen_total"] == 160
        assert result.row["flush_trigger"] == "count"
        assert result.row["is_end_of_source_report"] is False

    def test_final_partial_page_flagged_end_of_source(self) -> None:
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble({"schema": DYNAMIC_SCHEMA, "text_field": "line"})
        ctx = _ctx(
            batch_size=37,
            flush_index=1,
            rows_seen_total=37,
            row_start=1,
            row_end=37,
            trigger_type="end_of_source",
            is_end_of_source=True,
        )
        rows = [_row({"line": f"line-{i}"}) for i in range(37)]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["is_end_of_source_report"] is True
        assert result.row["line_count"] == 37

    def test_configurable_output_field(self) -> None:
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble({"schema": DYNAMIC_SCHEMA, "text_field": "line", "output_field": "page_text"})
        ctx = _ctx(batch_size=2)

        result = transform.process([_row({"line": "alpha"}), _row({"line": "beta"})], ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["page_text"] == "alpha\nbeta"
        # Default report_body MUST NOT also be populated — the configured
        # output field is authoritative.
        assert "report_body" not in result.row

    def test_markdown_title_prepended(self) -> None:
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble(
            {
                "schema": DYNAMIC_SCHEMA,
                "text_field": "line",
                "format": "markdown",
                "title": "Findings",
            }
        )
        ctx = _ctx(batch_size=2)

        result = transform.process([_row({"line": "alpha"}), _row({"line": "beta"})], ctx)

        assert result.status == "success"
        assert result.row is not None
        body = result.row["report_body"]
        assert isinstance(body, str)
        assert body.startswith("# Findings\n\n")
        assert result.row["report_format"] == "markdown"

    def test_html_fragment_renders_escaped_paragraphs_with_title(self) -> None:
        # html_fragment must escape HTML-special chars in body lines and the
        # title, wrap each line in <p>, and wrap the title in <h1>.
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble(
            {
                "schema": DYNAMIC_SCHEMA,
                "text_field": "line",
                "format": "html_fragment",
                "title": "Headlines & <updates>",
            }
        )
        ctx = _ctx(batch_size=2)

        result = transform.process([_row({"line": "a < b"}), _row({"line": "c > d"})], ctx)

        assert result.status == "success"
        assert result.row is not None
        body = result.row["report_body"]
        assert isinstance(body, str)
        # Title is escaped and wrapped in <h1>
        assert "<h1>Headlines &amp; &lt;updates&gt;</h1>" in body
        # Each body line is escaped and wrapped in <p>
        assert "<p>a &lt; b</p>" in body
        assert "<p>c &gt; d</p>" in body
        assert result.row["report_format"] == "html_fragment"

    def test_join_with_separator_applies_to_html_fragment(self) -> None:
        # Regression: html_fragment previously hardcoded "\n" between <p>
        # elements, silently dropping the configured join_with separator.
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble(
            {
                "schema": DYNAMIC_SCHEMA,
                "text_field": "line",
                "format": "html_fragment",
                "join_with": "\n\n",
            }
        )
        ctx = _ctx(batch_size=2)

        result = transform.process([_row({"line": "alpha"}), _row({"line": "beta"})], ctx)

        assert result.status == "success"
        assert result.row is not None
        body = result.row["report_body"]
        assert body == "<p>alpha</p>\n\n<p>beta</p>"

    def test_join_with_non_default_applies_to_plain_text(self) -> None:
        # Sanity check: join_with applies as documented for plain_text format.
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble(
            {
                "schema": DYNAMIC_SCHEMA,
                "text_field": "line",
                "join_with": "|",
            }
        )
        ctx = _ctx(batch_size=2)

        result = transform.process([_row({"line": "alpha"}), _row({"line": "beta"})], ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["report_body"] == "alpha|beta"

    def test_non_string_row_value_crashes(self) -> None:
        # Plugin contract: text_field MUST be a string by the time it
        # reaches the report assembler.  A non-string value is an
        # upstream validation bug, not a row-data fault, so the transform
        # must crash rather than coerce.
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble({"schema": DYNAMIC_SCHEMA, "text_field": "line"})
        ctx = _ctx(batch_size=1, row_end=1, rows_seen_total=1)

        with pytest.raises(TypeError, match="must be a string"):
            transform.process([_row({"line": 3})], ctx)

    def test_empty_batch_returns_non_retryable_error(self) -> None:
        # An empty batch reaching a batch-aware transform indicates an
        # upstream executor bug (the executor must not flush an empty
        # buffer).  Surface as a non-retryable error rather than emitting
        # an empty report.
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        transform = ReportAssemble({"schema": DYNAMIC_SCHEMA, "text_field": "line"})
        ctx = _ctx(batch_size=1, row_end=1, rows_seen_total=1)

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.retryable is False


class TestReportAssembleConfig:
    def test_output_field_colliding_with_metadata_is_rejected(self) -> None:
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        for reserved in (
            "report_format",
            "report_index",
            "line_start",
            "line_end",
            "line_count",
            "lines_seen_total",
            "flush_trigger",
            "is_end_of_source_report",
        ):
            with pytest.raises(PluginConfigError, match="reserved report metadata"):
                ReportAssemble({"schema": DYNAMIC_SCHEMA, "text_field": "line", "output_field": reserved})

    def test_title_env_ref_placeholder_is_rejected(self) -> None:
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        with pytest.raises(PluginConfigError, match="environment-variable placeholders"):
            ReportAssemble({"schema": DYNAMIC_SCHEMA, "text_field": "line", "title": "${ELSPETH_SECRET_KEY}"})

    def test_join_with_env_ref_placeholder_is_rejected(self) -> None:
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        with pytest.raises(PluginConfigError, match="environment-variable placeholders"):
            ReportAssemble({"schema": DYNAMIC_SCHEMA, "text_field": "line", "join_with": r"${JOINER:-\n}"})


class TestReportAssembleDiscovery:
    def test_register_builtin_plugins_discovers_report_assemble(self) -> None:
        from elspeth.plugins.infrastructure.manager import PluginManager
        from elspeth.plugins.transforms.report_assemble import ReportAssemble

        manager = PluginManager()
        manager.register_builtin_plugins()

        discovered = manager.get_transform_by_name("report_assemble")
        assert discovered is ReportAssemble
