"""Tests for detect_mime_type — content-based MIME sniffing.

Verifies the byte-prefix heuristics for data-oriented formats:
- JSON (object and array)
- JSONL (multiple { lines)
- CSV (comma and tab delimited)
- Plain text (valid UTF-8, no markers)
- Binary (null bytes or invalid UTF-8)
- Edge cases (empty, BOM, whitespace)
"""

from __future__ import annotations

from elspeth.web.blobs.sniff import detect_mime_type


class TestJsonDetection:
    """JSON starts with [ or { after optional whitespace."""

    def test_json_array(self) -> None:
        assert detect_mime_type(b'[{"id": 1}]') == "application/json"

    def test_json_object(self) -> None:
        assert detect_mime_type(b'{"key": "value"}') == "application/json"

    def test_json_with_leading_whitespace(self) -> None:
        assert detect_mime_type(b'  \n  {"key": "value"}') == "application/json"

    def test_json_array_with_bom(self) -> None:
        assert detect_mime_type(b"\xef\xbb\xbf" + b"[1, 2, 3]") == "application/json"


class TestJsonlDetection:
    """JSONL: multiple lines starting with {."""

    def test_jsonl_two_lines(self) -> None:
        content = b'{"a": 1}\n{"b": 2}\n'
        assert detect_mime_type(content) == "application/x-jsonlines"

    def test_jsonl_three_lines(self) -> None:
        content = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'
        assert detect_mime_type(content) == "application/x-jsonlines"

    def test_single_json_object_not_jsonl(self) -> None:
        """One { line is JSON, not JSONL."""
        assert detect_mime_type(b'{"only": "one"}') == "application/json"


class TestCsvDetection:
    """CSV: comma or tab in first line, no JSON markers."""

    def test_comma_delimited(self) -> None:
        assert detect_mime_type(b"name,age,city\nAlice,30,London") == "text/csv"

    def test_tab_delimited(self) -> None:
        assert detect_mime_type(b"name\tage\tcity\nAlice\t30\tLondon") == "text/csv"

    def test_csv_with_bom(self) -> None:
        assert detect_mime_type(b"\xef\xbb\xbf" + b"a,b,c\n1,2,3") == "text/csv"

    def test_three_column_multi_row_csv(self) -> None:
        """Multi-line content with consistent 3+ columns is CSV."""
        assert detect_mime_type(b"a,b,c\n1,2,3\n4,5,6") == "text/csv"

    def test_single_line_csv_header_is_ambiguous(self) -> None:
        """Single-line CSV (header only) is ambiguous — returns None so
        caller can fall back to browser-declared MIME.
        """
        assert detect_mime_type(b"name,age,city") is None

    def test_two_column_csv_is_ambiguous(self) -> None:
        """Two-column CSV (1 delimiter per line) below threshold — returns None.

        The browser's declared MIME (e.g. text/csv) is preserved by the caller.
        """
        assert detect_mime_type(b"name,age\nAlice,30") is None

    def test_csv_with_quoted_fields_containing_commas(self) -> None:
        """RFC 4180: commas inside quoted fields must not inflate field count.

        Without quote-aware parsing, '"Smith, John",30,Boston' has 3 raw
        commas vs 2 in the header — causing a false mismatch.
        """
        content = b'name,age,city\n"Smith, John",30,Boston'
        assert detect_mime_type(content) == "text/csv"


class TestPlainText:
    """Plain text: valid UTF-8 without JSON or CSV markers."""

    def test_single_line_with_comma_is_ambiguous(self) -> None:
        """Single-line content with commas is ambiguous — returns None."""
        assert detect_mime_type(b"Hello, world!") is None

    def test_multi_line_prose_with_commas_is_ambiguous(self) -> None:
        """Prose with commas below CSV threshold — returns None."""
        assert detect_mime_type(b"Hello, world!\nGoodbye, world!") is None

    def test_no_delimiters(self) -> None:
        assert detect_mime_type(b"Just some plain text") == "text/plain"

    def test_whitespace_only(self) -> None:
        assert detect_mime_type(b"   \n  \n  ") == "text/plain"


class TestBinaryDetection:
    """Binary: null bytes or invalid UTF-8."""

    def test_null_bytes(self) -> None:
        assert detect_mime_type(b"\x00\x01\x02\x03") == "application/octet-stream"

    def test_invalid_utf8(self) -> None:
        assert detect_mime_type(b"\x80\x81\x82") == "application/octet-stream"

    def test_mixed_text_with_null(self) -> None:
        assert detect_mime_type(b"header\x00binary") == "application/octet-stream"


class TestEdgeCases:
    """Empty content and other edge cases."""

    def test_empty_bytes(self) -> None:
        assert detect_mime_type(b"") is None

    def test_bom_only(self) -> None:
        """BOM with no content after stripping."""
        assert detect_mime_type(b"\xef\xbb\xbf") == "text/plain"

    def test_inconsistent_field_counts_not_csv(self) -> None:
        """Rows with different field counts are not treated as CSV.

        The consistency check requires all rows to have the same field count.
        A header with 3 fields and a data row with 2 means the file is not
        structured CSV — it should be ambiguous (None) so the browser MIME
        is preserved.
        """
        assert detect_mime_type(b"a,b,c\n1,2") is None

    def test_two_column_tab_delimited_is_ambiguous(self) -> None:
        """Two-column tab-delimited is below the 3-field threshold."""
        assert detect_mime_type(b"name\tage\nAlice\t30") is None


# ---------------------------------------------------------------------------
# UTF-16 / UTF-32 BOM detection (elspeth-3e6a7e0cdb)
# ---------------------------------------------------------------------------


class TestBomEncodings:
    """Non-UTF-8 text with BOMs decodes to the right data type rather than
    being misclassified as application/octet-stream because of its NUL bytes."""

    def test_utf16_le_csv(self) -> None:
        content = b"\xff\xfe" + "name,age,city\nAlice,30,London\n".encode("utf-16-le")
        assert detect_mime_type(content) == "text/csv"

    def test_utf16_be_csv(self) -> None:
        content = b"\xfe\xff" + "name,age,city\nAlice,30,London\n".encode("utf-16-be")
        assert detect_mime_type(content) == "text/csv"

    def test_utf16_le_plain_text(self) -> None:
        content = b"\xff\xfe" + "hello world\n".encode("utf-16-le")
        assert detect_mime_type(content) == "text/plain"

    def test_utf16_be_plain_text(self) -> None:
        content = b"\xfe\xff" + "hello world\n".encode("utf-16-be")
        assert detect_mime_type(content) == "text/plain"

    def test_utf16_le_json(self) -> None:
        content = b"\xff\xfe" + '{"key": "value"}'.encode("utf-16-le")
        assert detect_mime_type(content) == "application/json"

    def test_utf32_le_json(self) -> None:
        content = b"\xff\xfe\x00\x00" + '{"k": 1}'.encode("utf-32-le")
        assert detect_mime_type(content) == "application/json"

    def test_utf32_be_csv(self) -> None:
        # 3-column CSV so we clear the sniffer's CSV field floor
        content = b"\x00\x00\xfe\xff" + "a,b,c\n1,2,3\n".encode("utf-32-be")
        assert detect_mime_type(content) == "text/csv"

    def test_utf32_le_bom_not_misread_as_utf16_le(self) -> None:
        """UTF-32 LE BOM starts with the UTF-16 LE BOM.  If BOMs were
        checked shortest-first, a UTF-32 file would decode as garbage
        UTF-16 and return the wrong MIME.  Anchor the dispatch order.
        """
        content = b"\xff\xfe\x00\x00" + "hello\n".encode("utf-32-le")
        assert detect_mime_type(content) == "text/plain"

    def test_bom_with_corrupt_payload_returns_octet_stream(self) -> None:
        """A BOM followed by bytes that don't decode is a deceptive or
        corrupted upload.  The sniffer classifies it as binary so the
        route layer's allowlist check rejects it — a BOM and its
        following bytes are two Tier 3 signals, and disagreeing signals
        do not earn the upload a pass via browser-declared MIME.
        """
        # UTF-16 LE BOM followed by lone high surrogate (invalid)
        assert detect_mime_type(b"\xff\xfe" + b"\x00\xd8") == "application/octet-stream"

    def test_real_binary_still_detected_as_octet_stream(self) -> None:
        """PNG header has no recognised BOM and contains NUL bytes —
        must still hit the binary path, not slip through as None.
        """
        png_magic = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        assert detect_mime_type(png_magic) == "application/octet-stream"
