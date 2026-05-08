"""Unit tests for ``src/elspeth/web/composer/source_inspection.py``.

Covers:

  * Source-kind detection from MIME / filename / content shape.
  * CSV inspection — headers, sample rows, type inference, URL hints,
    headerless detection.
  * JSON inspection — array, single object, wrapped data_key, JSONL
    auto-detection from .json content shape.
  * JSONL inspection — multi-line objects, partial parse failures.
  * Text inspection — single URL → web_scrape hint, multi-line text,
    URL detection.
  * Bounded reads (8 KiB / 100 rows).
  * Redacted identity surfacing without leaking raw content / full hash.
  * derive_extra_column_risk vs declared fixed-schema fields.
  * Frozen-dataclass deep immutability.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from elspeth.web.composer.source_inspection import (
    SourceInspectionFacts,
    derive_extra_column_risk,
    facts_to_dict,
    inspect_blob_content,
)

# --------------------------------------------------------------------------
# Source-kind detection
# --------------------------------------------------------------------------


class TestSourceKindDetection:
    def test_csv_by_mime(self) -> None:
        f = inspect_blob_content(content=b"a,b\n1,2\n", filename="x", mime_type="text/csv")
        assert f.source_kind == "csv"

    def test_csv_by_filename(self) -> None:
        f = inspect_blob_content(content=b"a,b\n1,2\n", filename="data.csv", mime_type="application/octet-stream")
        assert f.source_kind == "csv"

    def test_tsv_filename_treated_as_csv(self) -> None:
        f = inspect_blob_content(content=b"a\tb\n1\t2\n", filename="data.tsv", mime_type="application/octet-stream")
        assert f.source_kind == "csv"

    def test_json_array_by_mime(self) -> None:
        f = inspect_blob_content(content=b'[{"a": 1}]', filename="x.json", mime_type="application/json")
        assert f.source_kind == "json"

    def test_jsonl_by_extension(self) -> None:
        f = inspect_blob_content(
            content=b'{"a": 1}\n{"a": 2}\n',
            filename="x.jsonl",
            mime_type="application/octet-stream",
        )
        assert f.source_kind == "jsonl"

    def test_json_extension_with_jsonl_content_detected_as_jsonl(self) -> None:
        """A .json file whose content is `{...}\n{...}\n` is JSONL."""
        f = inspect_blob_content(content=b'{"a": 1}\n{"a": 2}\n', filename="x.json", mime_type="application/json")
        assert f.source_kind == "jsonl"

    def test_text_by_mime(self) -> None:
        f = inspect_blob_content(content=b"line1\nline2\n", filename="x", mime_type="text/plain")
        assert f.source_kind == "text"

    def test_unknown_mime_and_filename(self) -> None:
        f = inspect_blob_content(content=b"\x00\x01\x02", filename="x.bin", mime_type="application/octet-stream")
        assert f.source_kind == "unknown"
        assert any("unrecognised mime_type" in w for w in f.warnings)


# --------------------------------------------------------------------------
# CSV inspection
# --------------------------------------------------------------------------


class TestCsvInspection:
    def test_headers_extracted(self) -> None:
        f = inspect_blob_content(
            content=b"id,name,price\n1,Alice,9.99\n2,Bob,19.95\n",
            filename="x.csv",
            mime_type="text/csv",
        )
        assert f.observed_headers == ("id", "name", "price")
        assert f.sample_row_count == 2

    def test_inferred_types(self) -> None:
        f = inspect_blob_content(
            content=b"id,name,price,active\n1,Alice,9.99,true\n2,Bob,19.95,false\n",
            filename="x.csv",
            mime_type="text/csv",
        )
        assert f.inferred_types == {
            "id": "int",
            "name": "str",
            "price": "float",
            "active": "bool",
        }

    def test_int_then_float_promotes_to_float(self) -> None:
        f = inspect_blob_content(
            content=b"v\n1\n2\n3.5\n",
            filename="x.csv",
            mime_type="text/csv",
        )
        assert f.inferred_types is not None
        assert f.inferred_types["v"] == "float"

    def test_all_null_column(self) -> None:
        f = inspect_blob_content(
            content=b"id,note\n1,\n2,\n",
            filename="x.csv",
            mime_type="text/csv",
        )
        assert f.inferred_types is not None
        assert f.inferred_types["note"] == "null"

    def test_url_candidates_found_in_data_cells(self) -> None:
        f = inspect_blob_content(
            content=b"name,site\nA,https://example.com\nB,https://example.org\n",
            filename="x.csv",
            mime_type="text/csv",
        )
        assert "https://example.com" in f.url_candidates
        assert "https://example.org" in f.url_candidates

    def test_headerless_warning(self) -> None:
        f = inspect_blob_content(
            content=b"1,2,3\n4,5,6\n",
            filename="x.csv",
            mime_type="text/csv",
        )
        assert any("first row looks like data" in w for w in f.warnings)

    def test_empty_csv(self) -> None:
        f = inspect_blob_content(content=b"", filename="x.csv", mime_type="text/csv")
        assert f.source_kind == "csv"
        assert f.sample_row_count == 0
        assert any("empty" in w for w in f.warnings)

    def test_row_count_capped(self) -> None:
        """200 rows in input → at most 100 sampled (excluding header)."""
        body = b"a\n" + b"\n".join(str(i).encode() for i in range(200)) + b"\n"
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        assert f.sample_row_count <= 100


# --------------------------------------------------------------------------
# JSON / JSONL inspection
# --------------------------------------------------------------------------


class TestJsonInspection:
    def test_json_array_of_objects(self) -> None:
        body = json.dumps([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]).encode()
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")
        assert f.source_kind == "json"
        assert f.observed_headers == ("id", "name")
        assert f.inferred_types == {"id": "int", "name": "str"}
        assert f.sample_row_count == 2

    def test_wrapped_object_warning(self) -> None:
        body = json.dumps({"results": [{"id": 1}, {"id": 2}]}).encode()
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")
        assert any("data_key" in w for w in f.warnings)
        assert f.observed_headers == ("id",)

    def test_jsonl_per_line(self) -> None:
        body = b'{"id": 1, "tag": "x"}\n{"id": 2, "tag": "y"}\n'
        f = inspect_blob_content(content=body, filename="x.jsonl", mime_type="application/x-jsonlines")
        assert f.source_kind == "jsonl"
        assert f.observed_headers == ("id", "tag")
        assert f.sample_row_count == 2

    def test_jsonl_partial_parse_failure_warning(self) -> None:
        body = b'{"id": 1}\nnot-json\n{"id": 2}\n'
        f = inspect_blob_content(content=body, filename="x.jsonl", mime_type="application/jsonl")
        assert any("failed to parse" in w for w in f.warnings)
        # Successful rows still surface
        assert f.sample_row_count == 2

    def test_nested_structures_warning(self) -> None:
        body = json.dumps([{"id": 1, "tags": ["a", "b"]}]).encode()
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")
        assert any("nested structures" in w for w in f.warnings)


# --------------------------------------------------------------------------
# Text inspection
# --------------------------------------------------------------------------


class TestTextInspection:
    def test_single_url_emits_web_scrape_hint(self) -> None:
        f = inspect_blob_content(content=b"https://example.com\n", filename="input.txt", mime_type="text/plain")
        assert f.url_candidates == ("https://example.com",)
        assert any("web_scrape" in w for w in f.warnings)

    def test_multi_line_text_with_urls(self) -> None:
        f = inspect_blob_content(
            content=b"first line\nhttps://a.example\nthird line\n",
            filename="x.txt",
            mime_type="text/plain",
        )
        assert "https://a.example" in f.url_candidates
        assert any("URL(s)" in w for w in f.warnings)

    def test_plain_text_no_url_no_warning(self) -> None:
        f = inspect_blob_content(content=b"line one\nline two\n", filename="x.txt", mime_type="text/plain")
        assert f.url_candidates == ()
        assert f.warnings == ()


# --------------------------------------------------------------------------
# Redacted identity
# --------------------------------------------------------------------------


class TestRedactedIdentity:
    def test_filename_mime_byte_size_present(self) -> None:
        f = inspect_blob_content(content=b"a,b\n1,2\n", filename="orders.csv", mime_type="text/csv")
        assert f.redacted_identity["filename"] == "orders.csv"
        assert f.redacted_identity["mime_type"] == "text/csv"
        assert f.redacted_identity["byte_size"] == "8"

    def test_blob_id_surfaced_when_provided(self) -> None:
        bid = uuid4()
        f = inspect_blob_content(content=b"a\n1\n", filename="x.csv", mime_type="text/csv", blob_id=bid)
        assert f.redacted_identity["blob_id"] == str(bid)

    def test_content_hash_only_prefix(self) -> None:
        f = inspect_blob_content(
            content=b"a\n1\n",
            filename="x.csv",
            mime_type="text/csv",
            content_hash="sha256:0123456789abcdef" + "0" * 48,
        )
        prefix = f.redacted_identity["content_hash_prefix"]
        assert len(prefix) == 8
        assert prefix == "sha256:0"


# --------------------------------------------------------------------------
# Bounded reads
# --------------------------------------------------------------------------


class TestBoundedReads:
    def test_oversized_blob_truncated_to_8_kib(self) -> None:
        # 16 KiB of CSV — only the first 8 KiB should be inspected.
        big = b"a,b\n" + b"1,2\n" * 4096
        f = inspect_blob_content(content=big, filename="x.csv", mime_type="text/csv")
        assert f.byte_range_inspected[1] <= 8 * 1024
        # byte_size in identity reflects the *real* size, not the truncated one.
        assert int(f.redacted_identity["byte_size"]) == len(big)


# --------------------------------------------------------------------------
# Frozen / deep immutability
# --------------------------------------------------------------------------


class TestFrozenInvariants:
    def test_redacted_identity_is_read_only(self) -> None:
        f = inspect_blob_content(content=b"a\n1\n", filename="x.csv", mime_type="text/csv")
        with pytest.raises(TypeError):
            f.redacted_identity["filename"] = "leak"  # type: ignore[index]

    def test_inferred_types_is_read_only(self) -> None:
        f = inspect_blob_content(content=b"a\n1\n", filename="x.csv", mime_type="text/csv")
        assert f.inferred_types is not None
        with pytest.raises(TypeError):
            f.inferred_types["a"] = "str"  # type: ignore[index]

    def test_observed_headers_tuple(self) -> None:
        f = inspect_blob_content(content=b"a,b\n1,2\n", filename="x.csv", mime_type="text/csv")
        assert isinstance(f.observed_headers, tuple)


# --------------------------------------------------------------------------
# derive_extra_column_risk
# --------------------------------------------------------------------------


class TestDeriveExtraColumnRisk:
    def _facts_with_headers(self, headers: tuple[str, ...]) -> SourceInspectionFacts:
        body = (",".join(headers) + "\n" + ",".join("1" for _ in headers) + "\n").encode()
        return inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")

    def test_no_declared_fields_no_risk(self) -> None:
        f = self._facts_with_headers(("id", "name", "price"))
        assert derive_extra_column_risk(f, None) == ()

    def test_all_declared_no_risk(self) -> None:
        f = self._facts_with_headers(("id", "name", "price"))
        declared = ("id: int", "name: str", "price: float")
        assert derive_extra_column_risk(f, declared) == ()

    def test_missing_column_returned(self) -> None:
        f = self._facts_with_headers(("id", "name", "price", "extra"))
        declared = ("id: int", "name: str", "price: float")
        assert derive_extra_column_risk(f, declared) == ("extra",)

    def test_case_insensitive_match(self) -> None:
        f = self._facts_with_headers(("ID", "Name"))
        declared = ("id: int", "name: str")
        assert derive_extra_column_risk(f, declared) == ()

    def test_no_observed_headers_no_risk(self) -> None:
        f = inspect_blob_content(content=b"plain text\n", filename="x.txt", mime_type="text/plain")
        # text source has no observed_headers
        assert derive_extra_column_risk(f, ("text: str",)) == ()


# --------------------------------------------------------------------------
# facts_to_dict
# --------------------------------------------------------------------------


class TestFactsToDict:
    def test_round_trip_shape(self) -> None:
        f = inspect_blob_content(
            content=b"id,name\n1,Alice\n",
            filename="x.csv",
            mime_type="text/csv",
        )
        d = facts_to_dict(f)
        assert d["source_kind"] == "csv"
        assert d["observed_headers"] == ["id", "name"]
        assert d["inferred_types"] == {"id": "int", "name": "str"}
        assert isinstance(d["redacted_identity"], dict)
        assert d["byte_range_inspected"] == [0, len(b"id,name\n1,Alice\n")]

    def test_text_source_dict(self) -> None:
        f = inspect_blob_content(content=b"hello\n", filename="x.txt", mime_type="text/plain")
        d = facts_to_dict(f)
        assert d["observed_headers"] is None
        assert d["inferred_types"] is None
        assert d["url_candidates"] == []


# --------------------------------------------------------------------------
# Tier-3 hostile-input coverage.
# ``inspect_blob_content`` is the Tier-3 boundary at which raw operator-
# uploaded bytes enter the proof step. Per the function's contract,
# partial inspection must always succeed — any unhandled exception
# crashes ``preview_pipeline`` end-to-end. These tests pin that contract
# against a deliberate set of pathological inputs (depth bombs, oversize
# fields, control bytes, mid-codepoint truncation) and a Hypothesis
# property that drives random binary garbage through the entry point.
# --------------------------------------------------------------------------


class TestInspectBlobContentHostileInputs:
    """Pin ``inspect_blob_content``'s "always returns facts" contract."""

    @pytest.mark.parametrize("mime_type", ["text/csv", "application/json", "application/jsonl", "application/octet-stream"])
    def test_empty_bytes_returns_facts(self, mime_type: str) -> None:
        f = inspect_blob_content(content=b"", filename="x", mime_type=mime_type)
        assert isinstance(f, SourceInspectionFacts)
        assert f.byte_range_inspected[1] == 0

    def test_json_deeply_nested_array_does_not_crash(self) -> None:
        """JSON nesting depth bomb (within the 8 KiB peek)."""
        depth = 2000
        payload = ("[" * depth) + "1" + ("]" * depth)
        f = inspect_blob_content(
            content=payload.encode("utf-8"),
            filename="x.json",
            mime_type="application/json",
        )
        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "json"

    def test_json_deeply_nested_object_does_not_crash(self) -> None:
        """JSON depth bomb via nested objects."""
        depth = 1500
        opens = '{"a":' * depth
        closes = "}" * depth
        payload = opens + "1" + closes
        f = inspect_blob_content(
            content=payload.encode("utf-8"),
            filename="x.json",
            mime_type="application/json",
        )
        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "json"

    def test_csv_single_oversize_quoted_field_does_not_crash(self) -> None:
        """CSV with an 8-KiB single quoted field with embedded newlines."""
        big_value = "x" * 7000 + "\n" + "y" * 1000
        body = b'col_a,col_b\n"' + big_value.encode("utf-8") + b'",2\n'
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "csv"

    def test_csv_more_cells_than_headers_does_not_crash(self) -> None:
        """Row has more cells than the header row — must surface as facts.

        Already partially covered in TestCsvInspection but pinned here
        too as part of the hostile-input contract.
        """
        body = b"a,b\n1,2,3,4,5\n6,7,8,9,10\n"
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "csv"
        assert f.observed_headers == ("a", "b")

    def test_csv_fewer_cells_than_headers_does_not_crash(self) -> None:
        """Row has fewer cells than the header row — observed_headers
        unaffected, missing values treated as empty.
        """
        body = b"a,b,c,d\n1\n2,3\n"
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        assert isinstance(f, SourceInspectionFacts)
        assert f.observed_headers == ("a", "b", "c", "d")

    def test_csv_null_bytes_in_headers_does_not_crash(self) -> None:
        """Null bytes / control characters embedded in header bytes."""
        body = b"a\x00b,c\x01d,e\x02f\n1,2,3\n"
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "csv"

    def test_csv_truncated_mid_utf8_codepoint_does_not_crash(self) -> None:
        """Bytes truncated mid-UTF-8 — common at the 8 KiB sample boundary."""
        # ``\xe2\x9c`` is a partial 3-byte UTF-8 sequence (CHECK MARK ✓
        # is U+2713 = e2 9c 93 — drop the last byte).
        body = b"name,marker\nAlice,\xe2\x9c"
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "csv"

    def test_jsonl_garbage_lines_does_not_crash(self) -> None:
        """JSONL with a mix of garbage lines, non-objects, and valid rows."""
        body = b'{"a": 1}\nnot json\n[1, 2, 3]\n{"b": 2}\n\n\n{"c": 3}\n'
        f = inspect_blob_content(content=body, filename="x.jsonl", mime_type="application/jsonl")
        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "jsonl"
        assert any("failed to parse" in w for w in f.warnings)

    def test_non_utf8_binary_garbage_does_not_crash(self) -> None:
        """High-bit binary garbage with no MIME / extension hint."""
        body = bytes(range(256))
        f = inspect_blob_content(content=body, filename="data.bin", mime_type="application/octet-stream")
        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "unknown"

    def test_long_key_payload_object_does_not_crash(self) -> None:
        """JSON with a single long key — checks the inferred-type loop
        does not blow up on the large header.
        """
        long_key = "k" * 4000
        body = b'{"' + long_key.encode() + b'": 1}'
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")
        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "json"


class TestInspectBlobContentHypothesis:
    """Hypothesis-driven property test: random binary garbage at all four
    kind hints must always return ``SourceInspectionFacts`` and never raise.
    """

    def test_random_bytes_never_crash(self) -> None:
        from hypothesis import HealthCheck, given, settings
        from hypothesis import strategies as st

        @given(
            payload=st.binary(max_size=16384),
            mime=st.sampled_from(["text/csv", "application/json", "application/jsonl", "application/octet-stream", "text/plain"]),
            filename=st.sampled_from(["x", "data.csv", "data.json", "data.jsonl", "data.txt", "data.bin"]),
        )
        @settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def _prop(payload: bytes, mime: str, filename: str) -> None:
            facts = inspect_blob_content(content=payload, filename=filename, mime_type=mime)
            assert isinstance(facts, SourceInspectionFacts)

        _prop()
