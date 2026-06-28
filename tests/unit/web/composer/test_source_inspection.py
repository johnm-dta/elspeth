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
from types import MappingProxyType
from uuid import uuid4

import pytest

from elspeth.web.composer.source_inspection import (
    SourceInspectionFacts,
    _declared_field_is_required,
    _declared_field_name,
    derive_extra_column_risk,
    derive_required_header_mismatch_risk,
    facts_to_dict,
    inspect_blob_content,
    inspect_csv_source_content,
)


class TestDeclaredFieldIsRequiredBoundary:
    """Trust-boundary honesty tests for ``_declared_field_is_required``.

    A declared field spec arrives inside external / LLM-authored composer
    source options (Tier-3): it is either a ``str`` (``"name"`` / ``"name?"``)
    or a ``Mapping`` carrying an optional ``required`` flag. A non-bool
    ``required`` flag must be rejected with ``ValueError`` (offensive
    validation at the boundary), never coerced. Binds the invariant claimed
    by the ``@trust_boundary(source_param='field', suppresses=('R1','R5'))``
    decorator.
    """

    def test_rejects_non_bool_required_flag(self) -> None:
        with pytest.raises(ValueError, match="required flag must be bool when present"):
            _declared_field_is_required({"name": "id", "required": "yes"})

    def test_string_spec_optional_suffix(self) -> None:
        # "name:type?" form — the optional marker follows the type after the colon.
        assert _declared_field_is_required("email:str?") is False
        assert _declared_field_is_required("name:str") is True
        # No colon → treated as required (the suffix only applies to the typed form).
        assert _declared_field_is_required("email?") is True

    def test_mapping_spec_bool_required(self) -> None:
        assert _declared_field_is_required({"name": "id", "required": False}) is False
        assert _declared_field_is_required({"name": "id"}) is True


class TestDeclaredFieldName:
    """``_declared_field_name`` over the ``str | Mapping`` field-spec union.

    The string arm and the two legitimate no-name Mapping cases (the
    single-key YAML form ``{"col": "type"}`` carrying no ``name`` key, and an
    empty/whitespace name) return ``None`` so the caller drops the entry. A
    Mapping spec whose ``name`` is *present but not a str* is an upstream-
    validation invariant break — every authoring path is parsed by
    ``FieldDefinition.parse`` / ``_normalize_field_spec`` at the config-
    loading boundary, which raise on a non-str name — so it is asserted
    offensively here rather than silently skipped behind an isinstance guard.
    """

    def test_string_spec_returns_name(self) -> None:
        assert _declared_field_name("id: int") == "id"
        assert _declared_field_name("price: float?") == "price"

    def test_named_mapping_spec_returns_name(self) -> None:
        assert _declared_field_name({"name": "id", "type": "int", "required": True, "nullable": False}) == "id"
        # The JSON-Schema authoring shape (field_type) is also a str-named Mapping.
        assert _declared_field_name(MappingProxyType({"name": "price", "field_type": "float"})) == "price"

    def test_single_key_yaml_form_has_no_name_key(self) -> None:
        # ``{"id": "int"}`` (unquoted YAML form) carries no "name" key — honest
        # absence, not a malformed name; the caller recovers the name elsewhere.
        assert _declared_field_name({"id": "int"}) is None

    def test_empty_name_returns_none(self) -> None:
        assert _declared_field_name({"name": "   "}) is None
        assert _declared_field_name("   : int") is None

    def test_non_str_name_raises(self) -> None:
        with pytest.raises(ValueError, match="declared field spec 'name' must be str when present"):
            _declared_field_name({"name": 123})


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

    def test_tsv_parses_with_tab_delimiter(self) -> None:
        # `.tsv` files dispatch to `_inspect_csv` (kind="csv"), but their
        # content is tab-separated. Without a tab-delimiter override, the
        # whole row collapses into a single column and observed_headers /
        # inferred_types / sample_row_count are derived from malformed
        # row structure. Regression for Codex P2 finding.
        f = inspect_blob_content(
            content=b"id\tname\tprice\n1\tAlice\t9.99\n2\tBob\t19.95\n",
            filename="data.tsv",
            mime_type="application/octet-stream",
        )
        assert f.observed_headers == ("id", "name", "price")
        assert f.sample_row_count == 2
        assert f.inferred_types == {"id": "int", "name": "str", "price": "float"}
        # Non-default delimiter must surface in the audit-visible warnings
        # so the operator/composer LLM sees how the blob was actually parsed.
        assert any("csv_non_default_delimiter" in w and "tab" in w for w in f.warnings), f.warnings

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
        assert f.url_candidates == ("https://example.com", "https://example.org")

    def test_url_candidates_redact_query_values(self) -> None:
        f = inspect_blob_content(
            content=b"name,site\nA,https://example.com/download?sig=SECRET_TOKEN&x=1\n",
            filename="x.csv",
            mime_type="text/csv",
        )
        serialized = facts_to_dict(f)

        assert serialized["url_candidates"] == ["https://example.com"]
        assert "SECRET_TOKEN" not in repr(serialized)

    def test_url_candidates_drop_userinfo_and_path(self) -> None:
        """Userinfo (embedded credentials) and path segments (reset tokens,
        emails, other PII) must NOT survive redaction — only scheme + host
        (+ port). Regression for the credential/PII egress where
        ``_redact_url_candidate`` rebuilt the URL with raw ``netloc`` + ``path``
        and so leaked ``user:pass@`` and ``/reset/<token>`` into the
        tool-result / proof-diagnostic / sessions-DB surfaces.
        """
        f = inspect_blob_content(
            content=(
                b"name,site\n"
                b"A,https://user:s3cr3t@host.example/reset-password/TOK123?sig=Z#frag\n"
                b"B,https://api.example:8443/v1/users/alice@corp.example\n"
            ),
            filename="x.csv",
            mime_type="text/csv",
        )
        serialized = facts_to_dict(f)

        # Host preserved (routing hint); port preserved; everything else gone.
        assert serialized["url_candidates"] == [
            "https://host.example",
            "https://api.example:8443",
        ]
        blob = repr(serialized)
        assert "s3cr3t" not in blob  # userinfo credential
        assert "TOK123" not in blob  # path-embedded token
        assert "reset-password" not in blob  # path segment
        assert "alice@corp.example" not in blob  # PII in path

    def test_url_candidates_malformed_port_does_not_raise(self) -> None:
        """A malformed/out-of-range port must not break the never-raise contract.

        ``urlsplit(...).port`` raises ``ValueError`` for ``h:99999`` / ``h:abc``
        and ``_URL_PATTERN`` matches those. Inspection runs over arbitrary blob
        bytes, so the candidate must degrade gracefully (host kept, bad port
        dropped) rather than propagate a ValueError up through
        ``compute_proof_diagnostics``.
        """
        f = inspect_blob_content(
            content=(b"name,site\nA,http://host.example:99999/p\nB,http://other.example:abc/q\n"),
            filename="x.csv",
            mime_type="text/csv",
        )
        serialized = facts_to_dict(f)
        # No crash; out-of-range / non-numeric ports dropped, hosts kept, paths gone.
        assert serialized["url_candidates"] == ["http://host.example", "http://other.example"]

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

    def test_duplicate_headers_emits_warning(self) -> None:
        """Duplicate CSV headers silently collapse downstream — surface the
        duplication so the operator can rename or use field_mapping."""
        body = b"id,name,name,city\n1,Alice,Smith,NYC\n"
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        msgs = [w for w in f.warnings if "csv_duplicate_headers" in w]
        assert msgs, f.warnings
        # Duplicate name surfaced; the warning lists the offending header.
        assert any("'name'" in w for w in msgs), msgs

    def test_jagged_rows_emits_warning(self) -> None:
        """Rows with cell counts that don't match the header length must
        surface — the inspector silently fabricates ''/drops cells, which
        is operator-observable evidence."""
        body = b"a,b,c\n1,2\n3,4,5,6\n7,8,9\n"
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        msgs = [w for w in f.warnings if "csv_jagged_rows" in w]
        assert msgs, f.warnings
        # Two rows are jagged (one short, one long); the third is clean.
        assert any("2 row" in w for w in msgs), msgs

    def test_clean_csv_no_jagged_warning(self) -> None:
        body = b"a,b,c\n1,2,3\n4,5,6\n"
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        assert not any("csv_jagged_rows" in w for w in f.warnings), f.warnings

    def test_csv_source_content_with_columns_treats_first_record_as_data(self) -> None:
        # Two distinct hosts so the multi-URL detection is still exercised after
        # path-dropping redaction collapses same-host candidates. The per-path
        # segment (``/a``, ``/b``) is intentionally dropped — only scheme + host
        # survives (see test_url_candidates_drop_userinfo_and_path).
        body = b"https://example.com/a\nhttps://example.org/b\n"
        f = inspect_csv_source_content(
            content=body,
            filename="urls.txt",
            mime_type="text/plain",
            delimiter=",",
            skip_rows=0,
            columns=("url",),
        )
        assert f.source_kind == "csv"
        assert f.observed_headers == ("url",)
        assert f.sample_row_count == 2
        assert f.url_candidates == ("https://example.com", "https://example.org")

    def test_replacement_chars_in_csv_emit_warning(self) -> None:
        """Non-UTF-8 bytes get replaced with U+FFFD on decode; surface the
        count so the operator/LLM can declare encoding or treat as binary."""
        # Latin-1 'é' (0xE9) is invalid UTF-8 outside multi-byte sequences.
        body = b"name,city\nM\xe9lanie,Paris\nBob,NYC\n"
        f = inspect_blob_content(content=body, filename="x.csv", mime_type="text/csv")
        assert any("binary_or_non_utf8_content" in w for w in f.warnings), f.warnings

    def test_utf16le_bom_csv_decodes_and_warns_encoding(self) -> None:
        """A UTF-16 LE BOM CSV is accepted as text/csv by the upload sniffer
        (sniff._BOM_CODECS), so inspection must decode it with the matching
        codec — not the BOM-blind UTF-8 path that corrupts the headers into
        ``\\ufffd\\ufffdn\\x00a\\x00...`` garbage. It must ALSO warn that the
        csv source plugin defaults to encoding=utf-8 and will not bind these
        bytes, so the readable headers don't certify a run that fails."""
        content = b"\xff\xfe" + "name,age,city\nAlice,30,London\n".encode("utf-16-le")
        f = inspect_blob_content(content=content, filename="data.csv", mime_type="text/csv")
        # Headers are readable, not BOM-corrupted garbage.
        assert f.observed_headers == ("name", "age", "city"), f.observed_headers
        # An encoding warning names the detected BOM encoding.
        assert any("utf-16-le" in w and ("encoding" in w or "bom" in w.lower()) for w in f.warnings), f.warnings

    def test_utf8_bom_csv_strips_bom_and_warns(self) -> None:
        """A UTF-8 BOM CSV must have its BOM stripped from the first header
        (not retained as ``\\ufeffname``) and must warn that the default
        utf-8 csv source retains the U+FEFF prefix unless encoding=utf-8-sig."""
        content = b"\xef\xbb\xbf" + b"name,age,city\nAlice,30,London\n"
        f = inspect_blob_content(content=content, filename="data.csv", mime_type="text/csv")
        assert f.observed_headers is not None
        assert f.observed_headers[0] == "name", f.observed_headers
        assert any(("encoding" in w or "bom" in w.lower()) for w in f.warnings), f.warnings

    def test_utf32le_bom_csv_decodes_and_warns_encoding(self) -> None:
        """UTF-32 LE BOM (``\\xff\\xfe\\x00\\x00``) shares its first two bytes
        with the UTF-16 LE BOM (``\\xff\\xfe``); the 4-byte marker MUST be
        matched first or the bytes misdecode as UTF-16 garbage. This locks the
        longest-BOM-first ordering of the decode table."""
        content = b"\xff\xfe\x00\x00" + "name,age,city\nAlice,30,London\n".encode("utf-32-le")
        f = inspect_blob_content(content=content, filename="data.csv", mime_type="text/csv")
        assert f.observed_headers == ("name", "age", "city"), f.observed_headers
        assert any("utf-32-le" in w and ("encoding" in w or "bom" in w.lower()) for w in f.warnings), f.warnings


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

    def test_json_rejects_non_finite_constants_without_inferred_float(self) -> None:
        body = b'[{"score": NaN}, {"score": Infinity}]'
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")

        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "json"
        assert any("non-finite" in w for w in f.warnings), f.warnings
        assert f.inferred_types is None or f.inferred_types.get("score") != "float"

    def test_jsonl_rejects_non_finite_lines_without_inferred_float(self) -> None:
        body = b'{"score": NaN}\n{"score": Infinity}\n{"name": "safe"}\n'
        f = inspect_blob_content(content=body, filename="x.jsonl", mime_type="application/jsonl")

        assert isinstance(f, SourceInspectionFacts)
        assert f.source_kind == "jsonl"
        assert any("failed to parse" in w for w in f.warnings), f.warnings
        assert f.sample_row_count == 1
        assert f.inferred_types is not None
        assert "score" not in f.inferred_types

    def test_nested_structures_warning(self) -> None:
        body = json.dumps([{"id": 1, "tags": ["a", "b"]}]).encode()
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")
        assert any("nested structures" in w for w in f.warnings)

    def test_observed_headers_union_preserves_first_seen_order(self) -> None:
        """``observed_headers`` is the first-seen-order union of keys across all
        sampled objects, including keys that only appear in a later object.

        Heterogeneous JSON rows (sparse / superset keys) are valid Tier-3 input.
        The first object fixes the leading order; a key introduced only by a
        later object is appended at the position it is first seen, never
        reordered or dropped. This pins the ordered-dedup union semantics so a
        future refactor of the accumulation idiom cannot silently regress
        ordering or de-duplication.
        """
        body = json.dumps(
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob", "city": "NYC"},
                {"name": "Carol", "id": 3, "country": "AU"},
            ]
        ).encode()
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")
        # id, name from row 1; city first seen in row 2; country first seen in
        # row 3. Repeated keys (id/name in later rows) are de-duplicated and do
        # not perturb the first-seen order.
        assert f.observed_headers == ("id", "name", "city", "country")
        assert f.sample_row_count == 3

    def test_top_level_dict_without_list_value_emits_disambiguation_warning(self) -> None:
        """Wrapped-object detection probed and rejected — operator must see
        that the inspector treated the object as a single row rather than
        finding a wrapped row collection."""
        body = json.dumps({"name": "Alice", "city": "NYC"}).encode()
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")
        assert any("json_top_level_dict_treated_as_single_row" in w for w in f.warnings), f.warnings
        # Behaviour preserved: still treats as single row of one.
        assert f.sample_row_count == 1

    def test_full_content_parse_failure_does_not_claim_truncation(self) -> None:
        """When the blob is small enough to fit in the sample window,
        a parse failure means the document is genuinely malformed — the
        warning must not falsely suggest the sample was truncated."""
        body = b'{"name": "incomplete'  # malformed but small
        assert len(body) < 8 * 1024  # well within the sample window
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")
        msgs = [w for w in f.warnings if "json parse error" in w]
        assert msgs, f.warnings
        # Must NOT claim "truncated" when the sample held the whole body.
        assert not any("truncated" in w for w in msgs), msgs
        assert any("malformed" in w for w in msgs), msgs

    def test_truncated_sample_parse_failure_says_truncated(self) -> None:
        """A blob exceeding the sample window with a parse failure inside
        the truncated peek should surface the truncation context."""
        # 9 KiB of unbalanced array — the 8 KiB peek will mid-document.
        body = ("[" + ("1," * 5000) + '"end"').encode("utf-8")
        assert len(body) > 8 * 1024
        f = inspect_blob_content(content=body, filename="x.json", mime_type="application/json")
        msgs = [w for w in f.warnings if "json parse error" in w]
        assert msgs, f.warnings
        assert any("truncated" in w for w in msgs), msgs


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
        assert f.url_candidates == ("https://a.example",)
        assert any("URL(s)" in w for w in f.warnings)

    def test_plain_text_no_url_no_warning(self) -> None:
        f = inspect_blob_content(content=b"line one\nline two\n", filename="x.txt", mime_type="text/plain")
        assert f.url_candidates == ()
        assert f.warnings == ()

    def test_blank_lines_dropped_emits_warning_with_count(self) -> None:
        # Mixed blank/non-blank input: 3 non-blank lines, 3 blank lines.
        # Body splitlines() yields ["a", "", "b", "", "", "c"] → 3 blanks dropped.
        f = inspect_blob_content(
            content=b"a\n\nb\n\n\nc\n",
            filename="x.txt",
            mime_type="text/plain",
        )
        assert f.sample_row_count == 3
        blank_warnings = [w for w in f.warnings if w.startswith("text_blank_lines_dropped:")]
        assert len(blank_warnings) == 1, f.warnings
        assert "3 blank line(s)" in blank_warnings[0], blank_warnings[0]

    def test_no_blank_lines_no_blank_warning(self) -> None:
        f = inspect_blob_content(
            content=b"alpha\nbeta\ngamma\n",
            filename="x.txt",
            mime_type="text/plain",
        )
        assert f.sample_row_count == 3
        assert not any(w.startswith("text_blank_lines_dropped:") for w in f.warnings), f.warnings


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

    # --- Numeric invariants enforced in __post_init__ -----------------------
    #
    # ``byte_range_inspected`` is a ``(start, end)`` pair where the inspected
    # window is the half-open slice ``content[start:end]``. The semantic
    # contract is ``0 <= start <= end``; equality (``(0, 0)`` for an empty
    # blob) is valid. ``sample_row_count`` records how many rows were actually
    # parsed from the inspected window — never negative. Both invariants land
    # in the audit trail via ``facts_to_dict`` / ``preview_pipeline``'s proof
    # step, so a malformed value would corrupt the audit record. The asserts
    # in ``__post_init__`` make construction fail fast rather than letting the
    # malformed facts propagate.

    def test_byte_range_zero_zero_is_valid_empty_window(self) -> None:
        # Empty blob — both endpoints are zero, half-open slice is empty.
        f = SourceInspectionFacts(
            source_kind="unknown",
            redacted_identity={"filename": "empty", "mime_type": "text/plain", "byte_size": "0"},
            byte_range_inspected=(0, 0),
            sample_row_count=0,
            observed_headers=None,
            inferred_types=None,
            url_candidates=(),
            warnings=(),
        )
        assert f.byte_range_inspected == (0, 0)
        assert f.sample_row_count == 0

    def test_byte_range_normal_window_is_valid(self) -> None:
        f = SourceInspectionFacts(
            source_kind="text",
            redacted_identity={"filename": "x.txt", "mime_type": "text/plain", "byte_size": "100"},
            byte_range_inspected=(0, 100),
            sample_row_count=10,
            observed_headers=None,
            inferred_types=None,
            url_candidates=(),
            warnings=(),
        )
        assert f.byte_range_inspected == (0, 100)
        assert f.sample_row_count == 10

    def test_byte_range_negative_start_raises(self) -> None:
        with pytest.raises(ValueError, match=r"byte_range_inspected must satisfy 0 <= start <= end"):
            SourceInspectionFacts(
                source_kind="unknown",
                redacted_identity={"filename": "x", "mime_type": "text/plain", "byte_size": "0"},
                byte_range_inspected=(-1, 100),
                sample_row_count=0,
                observed_headers=None,
                inferred_types=None,
                url_candidates=(),
                warnings=(),
            )

    def test_byte_range_inverted_pair_raises(self) -> None:
        with pytest.raises(ValueError, match=r"byte_range_inspected must satisfy 0 <= start <= end"):
            SourceInspectionFacts(
                source_kind="unknown",
                redacted_identity={"filename": "x", "mime_type": "text/plain", "byte_size": "0"},
                byte_range_inspected=(50, 10),
                sample_row_count=0,
                observed_headers=None,
                inferred_types=None,
                url_candidates=(),
                warnings=(),
            )

    def test_sample_row_count_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=r"sample_row_count must be non-negative"):
            SourceInspectionFacts(
                source_kind="unknown",
                redacted_identity={"filename": "x", "mime_type": "text/plain", "byte_size": "0"},
                byte_range_inspected=(0, 0),
                sample_row_count=-1,
                observed_headers=None,
                inferred_types=None,
                url_candidates=(),
                warnings=(),
            )


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

    def test_structured_declared_fields_no_risk(self) -> None:
        f = self._facts_with_headers(("id", "name", "price"))
        declared = (
            MappingProxyType({"name": "id", "field_type": "str"}),
            MappingProxyType({"name": "name", "field_type": "str"}),
            MappingProxyType({"name": "price", "field_type": "float"}),
        )
        assert derive_extra_column_risk(f, declared) == ()

    def test_round_trip_dict_form_no_false_extras(self) -> None:
        # The to_dict() bridge in compute_proof_diagnostics feeds field specs
        # in {"name","type","required","nullable"} form. Each carries a str
        # name, so every declared column is matched and no observed header is a
        # false "extra".
        f = self._facts_with_headers(("id", "name", "price"))
        declared = (
            {"name": "id", "type": "int", "required": True, "nullable": False},
            {"name": "name", "type": "str", "required": True, "nullable": False},
            {"name": "price", "type": "float", "required": True, "nullable": False},
        )
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


class TestDeriveRequiredHeaderMismatchRisk:
    def test_required_declared_fields_with_no_header_overlap_returned(self) -> None:
        f = inspect_csv_source_content(
            content=b"https://example.com/a\nhttps://example.com/b\n",
            filename="urls.txt",
            mime_type="text/plain",
            delimiter=",",
            skip_rows=0,
        )
        assert derive_required_header_mismatch_risk(f, ("url: str",)) == ("url",)

    def test_header_overlap_suppresses_risk(self) -> None:
        f = inspect_blob_content(content=b"URL\nhttps://example.com/a\n", filename="x.csv", mime_type="text/csv")
        assert derive_required_header_mismatch_risk(f, ("url: str",)) == ()

    def test_normalized_header_overlap_suppresses_risk(self) -> None:
        f = inspect_blob_content(content=b"Customer ID\n123\n", filename="x.csv", mime_type="text/csv")
        assert derive_required_header_mismatch_risk(f, ("customer_id: str",)) == ()

    def test_field_mapping_overlap_suppresses_risk(self) -> None:
        f = inspect_blob_content(content=b"External ID\n123\n", filename="x.csv", mime_type="text/csv")
        assert (
            derive_required_header_mismatch_risk(
                f,
                ("customer_id: str",),
                field_mapping={"external_id": "customer_id"},
            )
            == ()
        )

    def test_optional_declared_fields_do_not_require_header_overlap(self) -> None:
        f = inspect_csv_source_content(
            content=b"https://example.com/a\nhttps://example.com/b\n",
            filename="urls.txt",
            mime_type="text/plain",
            delimiter=",",
            skip_rows=0,
        )
        assert derive_required_header_mismatch_risk(f, ("url: str?",)) == ()


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


class TestObservedColumnsFromContent:
    """``observed_columns_from_content`` — derive column names from inline source
    content, used to backfill observed_columns when a chat-resolved source left
    them empty (the data is the authority, not the LLM's claim)."""

    def test_json_url_array_yields_url_column(self) -> None:
        from elspeth.web.composer.source_inspection import observed_columns_from_content

        content = b'[{"url": "https://example/a"}, {"url": "https://example/b"}]'
        cols = observed_columns_from_content(content=content, filename="urls.json", mime_type="application/json")
        assert cols == ("url",)

    def test_csv_header_yields_columns(self) -> None:
        from elspeth.web.composer.source_inspection import observed_columns_from_content

        content = b"name,score\nada,42\n"
        cols = observed_columns_from_content(content=content, filename="data.csv", mime_type="text/csv")
        assert cols == ("name", "score")

    def test_no_detectable_columns_yields_empty_tuple(self) -> None:
        from elspeth.web.composer.source_inspection import observed_columns_from_content

        content = b"just some prose with no columns or headers"
        cols = observed_columns_from_content(content=content, filename="note.txt", mime_type="text/plain")
        assert cols == ()


class TestObservedColumnsFromPath:
    """``observed_columns_from_path`` — the bounded-read, path-taking variant.

    ``inspect_blob_content`` already truncates to ``_MAX_BYTES``, so reading the
    whole file at the call site (a guided commit's column backfill) is wasted
    I/O — a multi-hundred-MB blob would be slurped just to recover a header.
    This entry point reads at most ``_MAX_BYTES`` and degrades an unreadable
    file to ``()`` per its contract (the bound stays private to the inspector).
    """

    def test_reads_at_most_max_bytes(self, tmp_path, monkeypatch) -> None:
        import elspeth.web.composer.source_inspection as si

        # A file FAR larger than the inspector's read bound.
        big = tmp_path / "big.csv"
        big.write_bytes(b"id,name,score\n" + b"x,y,z\n" * si._MAX_BYTES)
        assert big.stat().st_size > si._MAX_BYTES

        seen: dict[str, int] = {}
        real = si.observed_columns_from_content

        def _spy(*, content: bytes, filename: str, mime_type: str) -> tuple[str, ...]:
            seen["len"] = len(content)
            return real(content=content, filename=filename, mime_type=mime_type)

        monkeypatch.setattr(si, "observed_columns_from_content", _spy)
        cols = si.observed_columns_from_path(path=big, filename="big.csv", mime_type="text/csv")

        # The fix: the call site hands the inspector a bounded prefix, not the
        # whole file. Without it, ``seen["len"]`` is the full file size.
        assert seen["len"] <= si._MAX_BYTES
        # Behaviour preserved: columns are still detected from the prefix.
        assert cols == ("id", "name", "score")

    def test_unreadable_path_degrades_to_empty(self, tmp_path) -> None:
        from elspeth.web.composer.source_inspection import observed_columns_from_path

        # A directory exists() but cannot be opened/read as a file -> OSError -> ().
        d = tmp_path / "adir"
        d.mkdir()
        assert observed_columns_from_path(path=d, filename="x.csv", mime_type="text/csv") == ()

    def test_missing_path_degrades_to_empty(self, tmp_path) -> None:
        from elspeth.web.composer.source_inspection import observed_columns_from_path

        missing = tmp_path / "nope.csv"
        assert observed_columns_from_path(path=missing, filename="x.csv", mime_type="text/csv") == ()
