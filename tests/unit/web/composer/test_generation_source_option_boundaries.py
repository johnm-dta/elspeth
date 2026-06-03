"""Trust-boundary honesty tests for the CSV/schema source-option helpers.

These bind the malformed-input invariants claimed by the ``@trust_boundary``
decorators on the CSV-source and schema option extractors in
``elspeth.web.composer.tools.generation``. The ``options`` / ``schema``
mappings arrive inside external / LLM-authored composer source options
(Tier-3, zero-trust). A malformed value must be rejected with ``ValueError``
(offensive validation at the boundary) and never silently coerced. The
companion ``enforce_trust_boundary_honesty`` gate requires each raising-shape
test to exist and to invoke the decorated function directly through
``source_param``.
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.tools.generation import (
    _csv_source_columns,
    _csv_source_delimiter,
    _csv_source_skip_rows,
    _schema_required_fields,
)


def test_csv_source_delimiter_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="must be str when present"):
        _csv_source_delimiter(options={"delimiter": 9})


def test_csv_source_delimiter_rejects_multichar() -> None:
    with pytest.raises(ValueError, match="must be one character"):
        _csv_source_delimiter(options={"delimiter": ",,"})


def test_csv_source_skip_rows_rejects_non_int() -> None:
    with pytest.raises(ValueError, match="must be int when present"):
        _csv_source_skip_rows(options={"skip_rows": "3"})


def test_csv_source_skip_rows_rejects_negative() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        _csv_source_skip_rows(options={"skip_rows": -1})


def test_csv_source_columns_rejects_non_sequence() -> None:
    with pytest.raises(ValueError, match="must be a list of strings"):
        _csv_source_columns(options={"columns": "a,b,c"})


def test_csv_source_columns_rejects_non_string_element() -> None:
    with pytest.raises(ValueError, match=r"columns\[1\] must be str"):
        _csv_source_columns(options={"columns": ["a", 2, "c"]})


def test_schema_required_fields_rejects_non_sequence() -> None:
    with pytest.raises(ValueError, match="must be a list of strings"):
        _schema_required_fields(schema={"required_fields": "id"})


def test_schema_required_fields_rejects_non_string_element() -> None:
    with pytest.raises(ValueError, match=r"required_fields\[0\] must be str"):
        _schema_required_fields(schema={"required_fields": [1, "name"]})
